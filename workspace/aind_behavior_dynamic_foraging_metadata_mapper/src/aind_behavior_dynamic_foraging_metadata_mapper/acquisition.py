import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import git
from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_dynamic_foraging.data_contract.utils import calculate_consumed_water
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic
from aind_behavior_services.rig import Device as AbsDevice
from aind_behavior_services.rig import cameras as abs_camera
from aind_behavior_services.rig import water_valve as abs_water_valve
from aind_behavior_services.session import Session
from aind_behavior_services.utils import get_fields_of_type, utcnow
from aind_data_schema.components.configs import TriggerType
from aind_data_schema.components.measurements import CalibrationFit, FitType, GenericModel, VolumeCalibration
from aind_data_schema.core.acquisition import (
    Acquisition,
    AcquisitionSubjectDetails,
    Code,
    DataStream,
    DetectorConfig,
    PerformanceMetrics,
    StimulusEpoch,
    StimulusModality,
)
from aind_data_schema_models import units
from aind_data_schema_models.modalities import Modality
from clabe.data_mapper import helpers as data_mapper_helpers
from cyclopts import App

logger = logging.getLogger(__name__)

app = App()


@app.default
def acqusition_from_dataset(
    data_directory: Path, repo_path: os.PathLike, end_time: Optional[datetime] = None
) -> Acquisition:
    """
    Create acquisition model for completed session.

    Args:
        data_directory (os.PathLike):
            Path to the directory containing the dataset to analyze. This
            directory is expected to include all required behavioral data files.

        repo_path (os.PathLike):
            Path to github repository.

        end_time: Optional[datetime]:
                End time of acquisition. If None, current time will be used.

    Returns:
        Acquisition:
            Acquisition model for session

    Raises:
        FileNotFoundError:
            If the specified data directory or required files do not exist.

        ValueError:
            If the dataset is malformed or missing required fields for
            computing metrics.
    """
    dataset = df_foraging_dataset(data_directory)
    input_schemas = dataset["Behavior"]["InputSchemas"]
    session_model = Session.model_validate(input_schemas["Session"].data)
    rig_model = AindDynamicForagingRig.model_validate(input_schemas["Rig"].data)
    task_logic_model = AindDynamicForagingTaskLogic.model_validate(input_schemas["TaskLogic"].data)
    repository = git.Repo(repo_path)

    if end_time is None:
        logger.warning("Session end time is not set. Using current time as end time.")
        acquisition_end_time = datetime.now(tz=timezone.utc)

    bonsai_code = _get_bonsai_as_code(repository)
    python_code = _get_python_as_code(repository)

    cameras = data_mapper_helpers.get_cameras(rig_model, exclude_without_video_writer=True)
    camera_configs = [_get_cameras_config(k, v, repository) for k, v in cameras.items()]

    # construct data stream
    modalities: list[Modality] = [getattr(Modality, "BEHAVIOR")]
    if len(camera_configs) > 0:
        modalities.append(getattr(Modality, "BEHAVIOR_VIDEOS"))
    modalities = list(set(modalities))

    active_devices = [
        _device[0]
        for _device in get_fields_of_type(rig_model, AbsDevice, stop_recursion_on_type=False)
        if _device[0] is not None and not isinstance(_device[1], abs_camera.CameraController)
    ]

    data_streams = [
        DataStream(
            stream_start_time=session_model.date,
            stream_end_time=acquisition_end_time,
            code=[bonsai_code, python_code],
            active_devices=active_devices,
            modalities=modalities,
            configurations=camera_configs,
            notes=session_model.notes,
        )
    ]

    # populate behavior epoch
    metrics = dataset["Behavior"]["Metrics"].data
    trainer_state = dataset["Behavior"]["TrainerState"].data
    performance_metrics = PerformanceMetrics(output_parameters=metrics.model_dump())

    stimulus_epoch = StimulusEpoch(
        stimulus_start_time=session_model.date,
        stimulus_end_time=acquisition_end_time,
        stimulus_name="GoCue",
        code=bonsai_code,
        stimulus_modalities=[StimulusModality.AUDITORY],
        performance_metrics=performance_metrics,
        curriculum_status=trainer_state.stage.name,
    )

    # Construct aind-data-schema session
    return Acquisition(
        subject_id=session_model.subject,
        subject_details=_get_subject_details(data_directory),
        instrument_id=rig_model.rig_name,
        acquisition_end_time=acquisition_end_time,
        acquisition_start_time=session_model.date,
        experimenters=session_model.experimenter,
        acquisition_type=session_model.experiment or task_logic_model.name,
        coordinate_system=None,
        data_streams=data_streams,
        calibrations=_get_water_calibration(rig_model),
        stimulus_epochs=[stimulus_epoch],
    )


def _get_subject_details(data_directory: os.PathLike) -> AcquisitionSubjectDetails:
    return AcquisitionSubjectDetails(
        mouse_platform_name="tube",
        reward_consumed_total=calculate_consumed_water(data_directory),
        reward_consumed_unit=units.VolumeUnit.ML,
    )


def _get_water_calibration(rig_model: AindDynamicForagingRig) -> List[VolumeCalibration]:

    water_calibrations = get_fields_of_type(rig_model, abs_water_valve.WaterValveCalibration)
    vol_cal = []
    for device_name, water_calibration in water_calibrations:
        c = water_calibration
        vol_cal.append(
            VolumeCalibration(
                device_name=device_name,
                calibration_date=water_calibration.date if water_calibration.date else utcnow(),
                input=list(c.interval_average.keys()),
                output=list(c.interval_average.values()),
                input_unit=units.TimeUnit.S,
                output_unit=units.VolumeUnit.ML,
                fit=CalibrationFit(
                    fit_type=FitType.LINEAR,
                    fit_parameters=GenericModel.model_validate(c.model_dump()),
                ),
            )
        )
    return vol_cal


def _get_cameras_config(name: str, camera: abs_camera.CameraTypes, repository: git.Repo) -> List[DetectorConfig]:

    if isinstance(camera.video_writer, abs_camera.VideoWriterFfmpeg):
        compression = Code(
            url="https://ffmpeg.org/",
            name="FFMPEG",
            parameters=GenericModel.model_validate(camera.video_writer.model_dump()),
        )
    elif isinstance(camera.video_writer, abs_camera.VideoWriterOpenCv):
        bonsai = _get_bonsai_as_code(repository)
        bonsai.parameters = GenericModel.model_validate(camera.video_writer.model_dump())
        compression = bonsai
    else:
        raise ValueError("Camera does not have a valid video writer configured.")

    camera = DetectorConfig(
        device_name=name,
        exposure_time=getattr(camera, "exposure", -1),
        exposure_time_unit=units.TimeUnit.US,
        trigger_type=TriggerType.EXTERNAL,
        compression=compression(camera.video_writer),
    )

    cameras = data_mapper_helpers.get_cameras(AindDynamicForagingTaskLogic, exclude_without_video_writer=True)

    return list(map(camera, cameras.keys(), cameras.values()))


def _get_bonsai_as_code(repository: git.Repo) -> Code:
    bonsai_folder = Path(Path(repository.working_tree_dir) / "bonsai" / "bonsai.exe").parent
    bonsai_env = data_mapper_helpers.snapshot_bonsai_environment(bonsai_folder / "bonsai.config")
    bonsai_version = bonsai_env.get("Bonsai", "unknown")
    assert isinstance(repository, git.Repo)

    return Code(
        url=repository.remote().url,
        name="Aind.Behavior.DynamicForaging",
        version=repository.head.commit.hexsha,
        language="Bonsai",
        language_version=bonsai_version,
    )


def _get_python_as_code(repository: git.Repo) -> Code:
    v = sys.version_info
    semver = f"{v.major}.{v.minor}.{v.micro}"
    if v.releaselevel != "final":
        semver += f"-{v.releaselevel}.{v.serial}"
    return Code(
        url=repository.remote().url,
        name="aind-behavior-dynamic-foraging",
        version=repository.head.commit.hexsha,
        language="Python",
        language_version=semver,
    )

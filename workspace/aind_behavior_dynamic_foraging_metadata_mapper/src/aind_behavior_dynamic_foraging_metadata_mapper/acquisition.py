import logging
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import git
from aind_behavior_curriculum import TrainerState
from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_dynamic_foraging.data_contract.utils import calculate_consumed_water
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic
from aind_behavior_services.rig import Device as AbsDevice
from aind_behavior_services.rig import cameras as abs_camera
from aind_behavior_services.session import Session
from aind_behavior_services.utils import get_fields_of_type, model_from_json_file
from aind_data_schema.components.configs import TriggerType
from aind_data_schema.components.measurements import GenericModel
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
from clabe.data_mapper.aind_data_schema import AindDataSchemaSessionDataMapper
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class AindAcquisitionDataMapper(AindDataSchemaSessionDataMapper):
    def __init__(
        self,
        data_path: os.PathLike,
        repository_path: os.PathLike,
        session_end_time: Optional[datetime] = None,
        curriculum_repository_path: Optional[os.PathLike] = None,
    ):
        """
        Class to create acquisition model for completed session.

        Args:
            data_path (os.PathLike):
                Path to the directory containing the dataset to analyze. This
                directory is expected to include all required behavioral data files.

            repository_path (os.PathLike):
                Path to github repository.

            session_end_time: Optional[datetime]:
                    End time of acquisition. If None, current time will be used.
        """

        self.data_path = data_path
        self.repository_path = repository_path
        self.session_end_time = session_end_time
        self.curriculum_repository_path = curriculum_repository_path

        self.session_model = model_from_json_file(
            json_path=Path(self.data_path) / "behavior" / "Logs" / "session_output.json", model=Session
        )
        self._mapped: Optional[Acquisition] = None

    def session_schema(self):
        return self.mapped

    @property
    def session_name(self) -> str:
        if self.session_model.session_name is None:
            raise ValueError("Session name is not set in the session model.")
        return self.session_model.session_name

    def map(self) -> Acquisition:
        logger.info("Mapping aind-data-schema Acquisition.")
        try:
            self._mapped = self._map()
            return self._mapped
        except (ValidationError, ValueError, IOError) as e:
            logger.error("Failed to map to aind-data-schema Session. %s", e)
            raise e

    def _map(self) -> Acquisition:
        """
        Create acquisition model for completed session.

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
        dataset = df_foraging_dataset(self.data_path)
        input_schemas = dataset["Behavior"]["InputSchemas"]
        session_model = Session.model_validate(input_schemas["Session"].data)
        rig_model = AindDynamicForagingRig.model_validate(input_schemas["Rig"].data)
        task_logic_model = AindDynamicForagingTaskLogic.model_validate(input_schemas["TaskLogic"].data)
        repository = git.Repo(self.repository_path)
        curriculum_repository = git.Repo(self.curriculum_repository_path or self.repository_path)
        trainer_state = TrainerState.model_validate(dataset["Behavior"]["TrainerState"].data)

        if self.session_end_time is None:
            logger.warning("Session end time is not set. Using current time as end time.")
            acquisition_end_time = datetime.now(tz=timezone.utc)
        else:
            acquisition_end_time = self.session_end_time

        bonsai_code = _get_bonsai_as_code(repository)
        python_code = _get_python_as_code(repository)
        curriculum_code = _get_curriculum_as_code(curriculum_repository, trainer_state)

        cameras = data_mapper_helpers.get_cameras(rig_model, exclude_without_video_writer=True)
        camera_configs = [_get_camera_config(k, v, repository) for k, v in cameras.items()]

        # construct data stream
        modalities: list[Modality.ONE_OF] = [getattr(Modality, "BEHAVIOR")]
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
                code=[bonsai_code, python_code, curriculum_code],
                active_devices=active_devices,
                modalities=modalities,
                configurations=camera_configs,
                notes=session_model.notes,
            )
        ]

        # populate behavior epoch
        metrics = dataset["Behavior"]["Metrics"].data
        trial_outcomes = dataset["Behavior"]["SoftwareEvents"]["TrialOutcome"].data["data"].iloc
        rewarded = sum(to["is_rewarded"] for to in trial_outcomes)
        water = calculate_consumed_water(self.data_path)
        performance_metrics = PerformanceMetrics(
            reward_consumed_during_epoch=None if not water else Decimal(str(water)),
            reward_consumed_unit=units.VolumeUnit.ML,
            trials_total=trial_outcomes[:].shape[0],
            trials_finished=metrics.unignored_trials_per_session[-1],
            trials_rewarded=rewarded,
            output_parameters=metrics.model_dump(),
        )

        stimulus_epoch = StimulusEpoch(
            stimulus_start_time=session_model.date,
            stimulus_end_time=acquisition_end_time,
            stimulus_name="GoCue",
            code=bonsai_code,
            stimulus_modalities=[StimulusModality.AUDITORY],
            performance_metrics=performance_metrics,
            training_protocol_name=str(trainer_state.curriculum.name),
        )

        # Construct aind-data-schema session
        return Acquisition(
            subject_id=session_model.subject,
            subject_details=_get_subject_details(self.data_path),
            instrument_id=rig_model.rig_name,
            acquisition_end_time=acquisition_end_time,
            acquisition_start_time=session_model.date,
            experimenters=session_model.experimenter,
            acquisition_type=session_model.experiment or task_logic_model.name,
            coordinate_system=None,
            data_streams=data_streams,
            stimulus_epochs=[stimulus_epoch],
        )


def _get_subject_details(data_path: os.PathLike) -> AcquisitionSubjectDetails:
    water = calculate_consumed_water(data_path)
    return AcquisitionSubjectDetails(
        mouse_platform_name="tube",
        reward_consumed_total=None if not water else Decimal(str(water)),
        reward_consumed_unit=units.VolumeUnit.ML,
    )


def _get_camera_config(name: str, camera: abs_camera.CameraTypes, repository: git.Repo) -> DetectorConfig:

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

    return DetectorConfig(
        device_name=name,
        exposure_time=getattr(camera, "exposure", -1),
        exposure_time_unit=units.TimeUnit.US,
        trigger_type=TriggerType.EXTERNAL,
        compression=compression,
    )


def _get_curriculum_as_code(repository: git.Repo, trainer_state: TrainerState) -> Code:

    return Code(
        url=repository.remote().url,
        commit_hash=repository.head.commit.hexsha,
        name=trainer_state.curriculum.pkg_location,
        version=trainer_state.curriculum.version,
        language="aind-behavior-curriculum",
    )


def _get_bonsai_as_code(repository: git.Repo) -> Code:
    bonsai_folder = Path(Path(repository.working_tree_dir) / ".bonsai" / "bonsai.exe").parent
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

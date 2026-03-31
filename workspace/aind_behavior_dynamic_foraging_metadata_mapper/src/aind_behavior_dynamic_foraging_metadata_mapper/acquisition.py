import os
from pathlib import Path
from datetime import datetime, timezone
from cyclopts import App

from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_behavior_services.session import Session
from aind_data_schema.components.configs import TriggerType
from aind_data_schema.components.connections import Connection
from aind_data_schema.components.identifiers import Software
from aind_data_schema.core.acquisition import (
    Acquisition,
    Code,
    DataStream,
    DetectorConfig,
    PerformanceMetrics,
    StimulusEpoch,
    StimulusModality,
)
from aind_data_schema_models.modalities import Modality

app = App()

@app.default
def acqusition_from_dataset(
    data_directory: Path,
) -> Acquisition:
    """
    Create acquisition model for completed session.

    Args:
        data_directory (os.PathLike):
            Path to the directory containing the dataset to analyze. This
            directory is expected to include all required behavioral data files.

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
    software_events = dataset["Behavior"]["SoftwareEvents"]
    software_events.load_all()

    input_schemas = dataset["Behavior"]["InputSchemas"]

    # extract info from session model
    session = Session.model_validate(input_schemas["Session"].data)
    acquisition_start_time = session.date
    subject_id = session.subject
    experimenter = session.experimenter
    notes = session.notes

    # extract info from rig model
    rig = AindDynamicForagingRig.model_validate(input_schemas["Rig"].data)

    instrument_id = os.getenv("aibs_comp_id", "unknown")
    acquisition_end_time = datetime.now(tz=timezone.utc)

    # populate camera data stream
    cam_configs = []
    active_devices = ["BehaviorBoard"]
    connections = []
    for name, camera in rig.triggered_camera_controller.cameras.items():
        cam_configs.append(
            DetectorConfig(
                device_name=name,
                exposure_time=camera.exposure,
                trigger_type=TriggerType.EXTERNAL,
                crop_offset_x=camera.region_of_interest.x,
                crop_offset_y=camera.region_of_interest.y,
                crop_width=camera.region_of_interest.width,
                crop_height=camera.region_of_interest.height,
            )
        )
        # TODO: compression
        active_devices.append(name)
        connections.append(Connection(source_device="BehaviorBoard", target_device=name))

    data_stream = DataStream(
        stream_start_time=acquisition_start_time,
        stream_end_time=acquisition_end_time,
        modalities=[Modality.BEHAVIOR_VIDEOS],
        code=[
            Code(
                url=r"https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging/blob/feat-adding-curriculum/src/aind_behavior_dynamic_foraging/rig.py",
                parameters=rig.model_dump(),
                core_dependency=Software(name="bonsai"),
            )
        ],
        active_devices=active_devices,
        configurations=cam_configs,
        connections=connections,
    )

    # populate behavior epoch
    metrics = dataset["Behavior"]["PreviousMetrics"].data
    trainer_state = dataset["Behavior"]["TrainerState"].data
    performance_metrics = PerformanceMetrics(output_parameters=metrics)

    stimulus_epoch = StimulusEpoch(
        stimulus_start_time=acquisition_start_time,
        stimulus_end_time=acquisition_end_time,
        stimulus_name="GoCue",
        code=Code(
            url=r"https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging/tree/feat-adding-curriculum",
            parameters=input_schemas["TaskLogic"].data.model_dump(),
            core_dependency=Software(name="bonsai"),
        ),
        stimulus_modalities=[StimulusModality.AUDITORY],
        performance_metrics=performance_metrics,
        curriculum_status=trainer_state.stage.name,
    )

    acq =  Acquisition(
        subject_id=subject_id,
        instrument_id=instrument_id,
        experimenters=experimenter,
        acquisition_start_time=acquisition_start_time,
        acquisition_end_time=acquisition_end_time,
        acquisition_type="DynamicForaging",
        notes=notes,
        data_streams=[data_stream],
        stimulus_epochs=[stimulus_epoch],
    )

    print(acq)
    return acq

from pathlib import Path

from aind_behavior_services.session import Session
from contraqctor.contract import Dataset, DataStreamCollection
from contraqctor.contract.camera import Camera
from contraqctor.contract.harp import (
    DeviceYmlByFile,
    HarpDevice,
)
from contraqctor.contract.json import PydanticModel, SoftwareEvents
from contraqctor.contract.mux import MapFromPaths

from .. import __semver__
from ..rig import AindDynamicForagingRig
from ..task_logic import AindDynamicForagingTaskLogic


def make_dataset(
    root_path: Path,
    name: str = "DynamicForagingDataset",
    description: str = "A Dynamic Foraging dataset",
    version: str = __semver__,
) -> Dataset:
    """
    Creates a Dataset object for the Dynamic Foraging experiment.
    This function constructs a hierarchical representation of the data streams collected
    during an experiment, including hardware device data, software events,
    and configuration files.
    Parameters
    ----------
    root_path : Path
        Path to the root directory containing the dataset
    name : str, optional
        Name of the dataset, defaults to "DynamicForagingDataset"
    description : str, optional
        Description of the dataset, defaults to "A Dynamic Foraging dataset"
    version : str, optional
        Version of the dataset, defaults to the package version (This is also the version of the experiment)
    Returns
    -------
    Dataset
        A Dataset object containing a hierarchical representation of all data streams
        from the Dynamic Foraging experiment, including:
        - Harp device data (behavior, manipulator, etc.)
        - Harp device commands
        - Software events
        - Log files
        - Configuration schemas (rig, task logic, session)
    """

    root_path = Path(root_path)
    return Dataset(
        name=name,
        version=version,
        description=description,
        data_streams=[
            DataStreamCollection(
                name="Behavior",
                description="Data from the Behavior modality",
                data_streams=[
                    HarpDevice(
                        name="HarpBehavior",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/Behavior.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    HarpDevice(
                        name="HarpManipulator",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/StepperDriver.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    HarpDevice(
                        name="HarpSniffDetector",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/SniffDetector.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    HarpDevice(
                        name="HarpLickometerRight",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/LickometerRight.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    HarpDevice(
                        name="HarpLickometerLeft",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/LickometerLeft.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    HarpDevice(
                        name="HarpClockGenerator",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/ClockGenerator.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    HarpDevice(
                        name="HarpEnvironmentSensor",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/EnvironmentSensor.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    HarpDevice(
                        name="HarpSoundCard",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/SoundCard.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    DataStreamCollection(
                        name="HarpCommands",
                        description="Commands sent to Harp devices",
                        data_streams=[
                            HarpDevice(
                                name="HarpBehavior",
                                reader_params=HarpDevice.make_params(
                                    path=root_path / "behavior/HarpCommands/Behavior.harp",
                                    device_yml_hint=DeviceYmlByFile(),
                                ),
                            ),
                            HarpDevice(
                                name="HarpManipulator",
                                reader_params=HarpDevice.make_params(
                                    path=root_path / "behavior/HarpCommands/StepperDriver.harp",
                                    device_yml_hint=DeviceYmlByFile(),
                                ),
                            ),
                            HarpDevice(
                                name="HarpSniffDetector",
                                reader_params=HarpDevice.make_params(
                                    path=root_path / "behavior/HarpCommands/SniffDetector.harp",
                                    device_yml_hint=DeviceYmlByFile(),
                                ),
                            ),
                            HarpDevice(
                                name="HarpLickometerLeft",
                                reader_params=HarpDevice.make_params(
                                    path=root_path / "behavior/HarpCommands/LickometerLeft.harp",
                                    device_yml_hint=DeviceYmlByFile(),
                                ),
                            ),
                            HarpDevice(
                                name="HarpLickometerRight",
                                reader_params=HarpDevice.make_params(
                                    path=root_path / "behavior/HarpCommands/LickometerRight.harp",
                                    device_yml_hint=DeviceYmlByFile(),
                                ),
                            ),
                            HarpDevice(
                                name="HarpClockGenerator",
                                reader_params=HarpDevice.make_params(
                                    path=root_path / "behavior/HarpCommands/ClockGenerator.harp",
                                    device_yml_hint=DeviceYmlByFile(),
                                ),
                            ),
                            HarpDevice(
                                name="HarpEnvironmentSensor",
                                reader_params=HarpDevice.make_params(
                                    path=root_path / "behavior/HarpCommands/EnvironmentSensor.harp",
                                    device_yml_hint=DeviceYmlByFile(),
                                ),
                            ),
                            HarpDevice(
                                name="HarpSoundCard",
                                reader_params=HarpDevice.make_params(
                                    path=root_path / "behavior/HarpCommands/SoundCard.harp",
                                    device_yml_hint=DeviceYmlByFile(),
                                ),
                            ),
                        ],
                    ),
                    DataStreamCollection(
                        name="SoftwareEvents",
                        description="Software events generated by the workflow. The timestamps of these events are low precision and should not be used to align to physiology data.",
                        data_streams=[
                            SoftwareEvents(
                                name="TrialGeneratorSpec",
                                description="An event emitted with the specification for the trial generator.",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/TrialGeneratorSpec.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="QuiescentPeriod",
                                description="An event emitted at the start of the quiescent period.",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/QuiscentPeriod.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="Response",
                                description="An event emitted when a response is registered (timestamp, null = no choice, true = right, false = left).",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/Response.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="ResponsePeriod",
                                description="An event emitted at the start of the response period.",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/ResponsePeriod.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="IsRightTriggerQuickRetract",
                                description="An event emitted when the quick retract logic was triggered (true = right, false = left).",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/IsRightTriggerQuickRetract.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="DeliverSecondaryReinforcer",
                                description="An event emitted when a secondary reinforcer is triggered. It serializes the information about the secondary reinforcer.",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/DeliverSecondaryReinforcer.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="RewardConsumptionPeriod",
                                description="An event emitted at the start of the reward consumption period.",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/RewardConsumptionPeriod.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="TrialOutcome",
                                description="An event emitted with the outcome of the trial. It serializes the information about the outcome (reward, choice and trial specifications).",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/TrialOutcome.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="ItiPeriod",
                                description="An event emitted at the start of the inter-trial interval.",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/ItiPeriod.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="RngSeed",
                                description="An event emitted with the random seed used for trial generation.",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/RngSeed.json"
                                ),
                            ),
                            SoftwareEvents(
                                name="EndExperiment",
                                description="An event emitted when the experiment ends.",
                                reader_params=SoftwareEvents.make_params(
                                    root_path / "behavior/SoftwareEvents/EndExperiment.json"
                                ),
                            ),
                        ],
                    ),
                    DataStreamCollection(
                        name="InputSchemas",
                        description="Configuration files for the behavior rig, task_logic and session.",
                        data_streams=[
                            PydanticModel(
                                name="Rig",
                                reader_params=PydanticModel.make_params(
                                    model=AindDynamicForagingRig,
                                    path=root_path / "behavior/Logs/rig_output.json",
                                ),
                            ),
                            PydanticModel(
                                name="TaskLogic",
                                reader_params=PydanticModel.make_params(
                                    model=AindDynamicForagingTaskLogic,
                                    path=root_path / "behavior/Logs/tasklogic_output.json",
                                ),
                            ),
                            PydanticModel(
                                name="Session",
                                reader_params=PydanticModel.make_params(
                                    model=Session,
                                    path=root_path / "behavior/Logs/session_output.json",
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            MapFromPaths(
                name="BehaviorVideos",
                description="Data from BehaviorVideos modality",
                reader_params=MapFromPaths.make_params(
                    paths=root_path / "behavior-videos",
                    include_glob_pattern=["*"],
                    inner_data_stream=Camera,
                    inner_param_factory=lambda camera_name: Camera.make_params(
                        path=root_path / "behavior-videos" / camera_name
                    ),
                ),
            ),
        ],
    )

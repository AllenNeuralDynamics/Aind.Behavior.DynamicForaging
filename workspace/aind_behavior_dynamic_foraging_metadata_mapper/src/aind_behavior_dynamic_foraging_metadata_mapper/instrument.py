from pathlib import Path
from datetime import date
from cyclopts import App

from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_data_schema.components.connections import Connection
from aind_data_schema.components.coordinates import Axis, AxisName, CoordinateSystem, Direction, Origin
from aind_data_schema.components.devices import (
    AnatomicalRelative,
    Camera,
    CameraAssembly,
    CameraTarget,
    DataInterface,
    HarpDevice,
    HarpDeviceType,
    Lens,
    MotorizedStage,
    SizeUnit,
)
from aind_data_schema.core.instrument import Instrument
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization

app = App()

@app.default
def instrument_from_dataset(
    data_directory: Path,
) -> Instrument:
    """
    Create Instrument model for completed session.

    Args:
        data_directory (os.PathLike):
            Path to the directory containing the dataset to analyze. This
            directory is expected to include all required behavioral data files.

    Returns:
        Instrument:
            Instrument model for session

    Raises:
        FileNotFoundError:
            If the specified data directory or required files do not exist.

        ValueError:
            If the dataset is malformed or missing required fields for
            computing metrics.
    """

    dataset = df_foraging_dataset(data_directory)
    input_schemas = dataset["Behavior"]["InputSchemas"]
    rig = AindDynamicForagingRig.model_validate(input_schemas["Rig"].data)

    components = []
    connections = []

    # cameras
    for name, cam in rig.triggered_camera_controller.cameras.items():
        camera = Camera(
            name=name,
            serial_number=cam.serial_number,
            manufacturer=Organization.SPINNAKER,
            data_interface=DataInterface.COAX,
        )
        assembly = CameraAssembly(
            name=f"{name}Assembly",
            camera=camera,
            target=CameraTarget.BODY if "Body" in name else CameraTarget.FACE,
            lens=Lens(name="Lens A", manufacturer=Organization.FUJINON),
            relative_position=[AnatomicalRelative.RIGHT if "Body" in name else AnatomicalRelative.SUPERIOR],
        )
        components.append(assembly)

    # behavior board
    components.append(
        HarpDevice(
            name="BehaviorBoard",
            harp_device_type=HarpDeviceType.BEHAVIOR,
            serial_number=rig.harp_behavior.serial_number,
            manufacturer=Organization.CHAMPALIMAUD,
            is_clock_generator=False,
        )
    )

    # clock generator
    components.append(
        HarpDevice(
            name="ClockGenerator",
            harp_device_type=HarpDeviceType.WHITERABBIT,
            serial_number=rig.harp_clock_generator.serial_number,
            is_clock_generator=True,
        )
    )

    # sound card
    components.append(
        HarpDevice(
            name="SoundCard",
            harp_device_type=HarpDeviceType.SOUNDCARD,
            serial_number=rig.harp_sound_card.serial_number,
            manufacturer=Organization.CHAMPALIMAUD,
            is_clock_generator=False,
        )
    )

    # optional harp devices
    if rig.harp_lickometer_left:
        components.append(
            HarpDevice(
                name="LickometerLeft",
                harp_device_type=HarpDeviceType.LICKETYSPLIT,
                serial_number=rig.harp_lickometer_left.serial_number,
                is_clock_generator=False,
            )
        )
    if rig.harp_lickometer_right:
        components.append(
            HarpDevice(
                name="LickometerRight",
                serial_number=rig.harp_lickometer_right.serial_number,
                harp_device_type=HarpDeviceType.LICKETYSPLIT,
                is_clock_generator=False,
            )
        )
    if rig.harp_sniff_detector:
        components.append(
            HarpDevice(
                name="SniffDetector",
                harp_device_type=HarpDeviceType.SNIFFDETECTOR,
                serial_number=rig.harp_sniff_detector.serial_number,
                is_clock_generator=False,
            )
        )
    if rig.harp_environment_sensor:
        components.append(
            HarpDevice(
                name="EnvironmentSensor",
                harp_device_type=HarpDeviceType.ENVIRONMENTSENSOR,
                serial_number=rig.harp_environment_sensor.serial_number,
                is_clock_generator=False,
            )
        )

    # manipulator
    components.append(MotorizedStage(name="Manipulator", serial_number=rig.manipulator.serial_number, travel=0.0))

    # connections
    for name in rig.triggered_camera_controller.cameras:
        connections.append(
            Connection(
                source_device="BehaviorBoard",
                target_device=name,
            )
        )

    inst =  Instrument(
        instrument_id=rig.rig_name,
        modification_date=date.today(),
        modalities=[Modality.BEHAVIOR, Modality.BEHAVIOR_VIDEOS],
        coordinate_system=CoordinateSystem(
            name="RigCoordinateSystem",
            origin=Origin.ORIGIN,
            axes=[
                Axis(name=AxisName.X, direction=Direction.LR),
                Axis(name=AxisName.Y, direction=Direction.FB),
                Axis(name=AxisName.Z, direction=Direction.DU),
            ],
            axis_unit=SizeUnit.MM,
        ),
        components=components,
        connections=connections,
    )
    print(inst.model_dump_json(indent=3))
    return inst

from datetime import date
import os

from aind_data_schema.core.instrument import Instrument
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_data_schema.components.devices import (
    Camera,
    CameraAssembly,
    CameraTarget,
    HarpDevice,
    MotorizedStage,
    Speaker,
    Computer,
)
from aind_data_schema.components.connections import Connection
from aind_data_schema.components.coordinates import CoordinateSystem
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization


def instrument_from_dataset(
    data_directory: os.PathLike,
) -> Instrument:
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
    
    rig = AindDynamicForagingRig.model_validate(input_schemas["Rig"].data)
    
    components = []
    connections = []

    # --- Triggered cameras wrapped in CameraAssembly (required for BEHAVIOR_VIDEOS) ---
    for name, cam in rig.triggered_camera_controller.cameras.items():
        camera = Camera(
            name=name,
            serial_number=cam.serial_number,
            # manufacturer=Organization.FLIR,  # TODO
            # model="Blackfly S BFS-U3-16S2M",  # TODO
        )
        assembly = CameraAssembly(
            name=f"{name}Assembly",
            camera=camera,
            target=CameraTarget.BODY,  # TODO: adjust per camera (FACE, SIDE, etc.)
            # lens=Lens(...),  # TODO if needed
        )
        components.append(assembly)

    # --- Monitoring cameras (optional) ---
    if rig.monitoring_camera_controller:
        for name, cam in rig.monitoring_camera_controller.cameras.items():
            camera = Camera(
                name=name,
                serial_number=getattr(cam, "serial_number", None),
                # manufacturer=...,  # TODO
            )
            assembly = CameraAssembly(
                name=f"{name}Assembly",
                camera=camera,
                target=CameraTarget.OTHER,  # TODO: requires notes on Instrument
            )
            components.append(assembly)

    # --- Harp behavior board ---
    components.append(
        HarpDevice(
            name="BehaviorBoard",
            serial_number=rig.harp_behavior.serial_number,
            who_am_i=rig.harp_behavior.who_am_i,
            port_name=rig.harp_behavior.port_name,
            # manufacturer=Organization.HARP_TECH,  # TODO
        )
    )

    # --- Harp clock generator ---
    components.append(
        HarpDevice(
            name="ClockGenerator",
            serial_number=rig.harp_clock_generator.serial_number,
            who_am_i=rig.harp_clock_generator.who_am_i,
            port_name=rig.harp_clock_generator.port_name,
            is_clock_generator=True,
        )
    )

    # --- Harp sound card ---
    components.append(
        HarpDevice(
            name="SoundCard",
            serial_number=rig.harp_sound_card.serial_number,
            who_am_i=rig.harp_sound_card.who_am_i,
            port_name=rig.harp_sound_card.port_name,
        )
    )

    # --- Optional harp devices ---
    if rig.harp_lickometer_left:
        components.append(HarpDevice(
            name="LickometerLeft",
            serial_number=rig.harp_lickometer_left.serial_number,
            who_am_i=rig.harp_lickometer_left.who_am_i,
            port_name=rig.harp_lickometer_left.port_name,
        ))
    if rig.harp_lickometer_right:
        components.append(HarpDevice(
            name="LickometerRight",
            serial_number=rig.harp_lickometer_right.serial_number,
            who_am_i=rig.harp_lickometer_right.who_am_i,
            port_name=rig.harp_lickometer_right.port_name,
        ))
    if rig.harp_sniff_detector:
        components.append(HarpDevice(
            name="SniffDetector",
            serial_number=rig.harp_sniff_detector.serial_number,
            who_am_i=rig.harp_sniff_detector.who_am_i,
            port_name=rig.harp_sniff_detector.port_name,
        ))
    if rig.harp_environment_sensor:
        components.append(HarpDevice(
            name="EnvironmentSensor",
            serial_number=rig.harp_environment_sensor.serial_number,
            who_am_i=rig.harp_environment_sensor.who_am_i,
            port_name=rig.harp_environment_sensor.port_name,
        ))

    # --- Manipulator (no dedicated type, use MotorizedStage) ---
    components.append(
        MotorizedStage(
            name="Manipulator",
            serial_number=rig.manipulator.serial_number,
            # manufacturer=...,  # TODO
            # model="AindManipulator",  # TODO
        )
    )

    # --- Connections: BehaviorBoard triggers cameras ---
    for name in rig.triggered_camera_controller.cameras:
        connections.append(Connection(
            source_device="BehaviorBoard",
            target_device=name,
        ))

    return Instrument(
        instrument_id=instrument_id,
        modification_date=date.today(),  # TODO: use actual last-modified date
        modalities=[Modality.BEHAVIOR_VIDEOS],  # TODO: add others if applicable
        coordinate_system=CoordinateSystem(
            name="RigCoordinateSystem",  # TODO: fill in properly
            # origin=...,
            # axes=...,
        ),
        components=components,
        connections=connections,
        # location="447",  # TODO: room/lab location
        # notes="...",     # Required if any CameraTarget.OTHER is used
    )
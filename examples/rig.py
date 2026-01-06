import datetime
import os

import aind_behavior_services.rig as rig
from aind_behavior_services.calibration.aind_manipulator import (
    AindManipulatorCalibration,
    AindManipulatorCalibrationInput,
    AindManipulatorCalibrationOutput,
    AindManipulatorDevice,
    Axis,
    AxisConfiguration,
    ManipulatorPosition,
)
from aind_behavior_services.calibration.water_valve import (
    Measurement,
    WaterValveCalibration,
    WaterValveCalibrationInput,
)
from aind_behavior_services.rig.harp import (
    HarpBehavior,
    HarpLicketySplit,
    HarpSniffDetector,
    HarpWhiteRabbit,
)

from aind_behavior_dynamic_foraging.rig import (
    AindDynamicForagingRig,
    RigCalibration,
)

manipulator_calibration = AindManipulatorCalibration(
    output=AindManipulatorCalibrationOutput(),
    input=AindManipulatorCalibrationInput(
        full_step_to_mm=(ManipulatorPosition(x=0.010, y1=0.010, y2=0.010, z=0.010)),
        axis_configuration=[
            AxisConfiguration(axis=Axis.Y1, min_limit=-0.01, max_limit=25),
            AxisConfiguration(axis=Axis.Y2, min_limit=-0.01, max_limit=25),
            AxisConfiguration(axis=Axis.X, min_limit=-0.01, max_limit=25),
            AxisConfiguration(axis=Axis.Z, min_limit=-0.01, max_limit=25),
        ],
        homing_order=[Axis.Y1, Axis.Y2, Axis.X, Axis.Z],
        initial_position=ManipulatorPosition(y1=0, y2=0, x=0, z=0),
    ),
)

water_valve_input = WaterValveCalibrationInput(
    measurements=[
        Measurement(valve_open_interval=0.2, valve_open_time=0.01, water_weight=[0.6, 0.6], repeat_count=200),
        Measurement(valve_open_interval=0.2, valve_open_time=0.02, water_weight=[1.2, 1.2], repeat_count=200),
    ]
)
water_valve_calibration = WaterValveCalibration(
    input=water_valve_input, output=water_valve_input.calibrate_output(), date=datetime.datetime.now()
)

video_writer = rig.cameras.VideoWriterFfmpeg(frame_rate=120, container_extension="mp4")

rig = AindDynamicForagingRig(
    rig_name="test_rig",
    triggered_camera_controller=rig.cameras.CameraController[rig.cameras.SpinnakerCamera](
        frame_rate=120,
        cameras={
            "FaceCamera": rig.cameras.SpinnakerCamera(
                serial_number="SerialNumber", binning=1, exposure=5000, gain=0, video_writer=video_writer
            ),
            "SideCamera": rig.cameras.SpinnakerCamera(
                serial_number="SerialNumber", binning=1, exposure=5000, gain=0, video_writer=video_writer
            ),
        },
    ),
    monitoring_camera_controller=None,
    harp_behavior=HarpBehavior(port_name="COM3"),
    harp_lickometer_left=HarpLicketySplit(port_name="COM5"),
    harp_lickometer_right=HarpLicketySplit(port_name="COM10"),
    harp_clock_generator=HarpWhiteRabbit(port_name="COM6"),
    harp_sniff_detector=HarpSniffDetector(port_name="COM7"),
    manipulator=AindManipulatorDevice(port_name="COM9", calibration=manipulator_calibration),
    calibration=RigCalibration(
        water_valve_left=water_valve_calibration,
        water_valve_right=water_valve_calibration,
    ),
)


def main(path_seed: str = "./local/{schema}.json"):
    os.makedirs(os.path.dirname(path_seed), exist_ok=True)
    models = [rig]

    for model in models:
        with open(path_seed.format(schema=model.__class__.__name__), "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

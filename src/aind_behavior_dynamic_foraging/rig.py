# Import core types
from typing import Literal, Optional

import aind_behavior_services.rig as rig
from aind_behavior_services.calibration import aind_manipulator, water_valve
from pydantic import BaseModel, Field

from aind_behavior_dynamic_foraging import __semver__


class RigCalibration(BaseModel):
    """Container class for calibration models. In a future release these will be moved to the respective devices"""

    water_valve_left: water_valve.WaterValveCalibration = Field(description="Left water valve calibration")
    water_valve_right: water_valve.WaterValveCalibration = Field(description="Right water valve calibration")


class AindDynamicForagingRig(rig.AindBehaviorRigModel):
    version: Literal[__semver__] = __semver__
    triggered_camera_controller: rig.cameras.CameraController[rig.cameras.SpinnakerCamera] = Field(
        description="Required camera controller to triggered cameras."
    )
    monitoring_camera_controller: Optional[rig.cameras.CameraController[rig.cameras.WebCamera]] = Field(
        default=None, description="Optional camera controller for monitoring cameras."
    )
    harp_behavior: rig.harp.HarpBehavior = Field(description="Harp behavior")
    harp_lickometer_left: rig.harp.HarpLicketySplit = Field(description="Harp left lickometer")
    harp_lickometer_right: rig.harp.HarpLicketySplit = Field(description="Harp right lickometer")
    harp_clock_generator: rig.harp.HarpWhiteRabbit = Field(description="Harp clock generator")

    harp_sniff_detector: Optional[rig.harp.HarpSniffDetector] = Field(
        default=None, description="Harp sniff detector"
    )
    harp_environment_sensor: Optional[rig.harp.HarpEnvironmentSensor] = Field(
        default=None, description="Harp environment sensor"
    )
    manipulator: aind_manipulator.AindManipulatorDevice = Field(description="Manipulator")
    calibration: RigCalibration = Field(description="Calibration models")
    harp_sound_card: rig.harp.HarpSoundCard = Field(description="Harp sound card")
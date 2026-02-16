# Import core types
from typing import Literal, Optional

import aind_behavior_services.rig as rig
from aind_behavior_services.rig import aind_manipulator, cameras, harp, water_valve
from pydantic import BaseModel, Field

from aind_behavior_dynamic_foraging import __semver__


class RigCalibration(BaseModel):
    """Container class for calibration models. In a future release these will be moved to the respective devices"""

    water_valve_left: water_valve.WaterValveCalibration = Field(description="Left water valve calibration")
    water_valve_right: water_valve.WaterValveCalibration = Field(description="Right water valve calibration")


class AindDynamicForagingRig(rig.Rig):
    version: Literal[__semver__] = __semver__
    triggered_camera_controller: cameras.CameraController[cameras.SpinnakerCamera] = Field(
        description="Required camera controller to triggered cameras."
    )
    monitoring_camera_controller: Optional[cameras.CameraController[cameras.WebCamera]] = Field(
        default=None, description="Optional camera controller for monitoring cameras."
    )
    harp_behavior: harp.HarpBehavior = Field(description="Harp behavior")
    harp_lickometer_left: Optional[harp.HarpLicketySplit] = Field(
        default=None,
        description="Harp left lickometer. If null, the rig will use the harp_behavior DIPort0 for lick detection.",
    )
    harp_lickometer_right: Optional[harp.HarpLicketySplit] = Field(
        default=None,
        description="Harp right lickometer. If null, the rig will use the harp_behavior DIPort1 for lick detection.",
    )
    harp_clock_generator: harp.HarpWhiteRabbit = Field(description="Harp clock generator")
    harp_sound_card: harp.HarpSoundCard = Field(description="Harp sound card")
    harp_sniff_detector: Optional[harp.HarpSniffDetector] = Field(default=None, description="Harp sniff detector")
    harp_environment_sensor: Optional[harp.HarpEnvironmentSensor] = Field(
        default=None, description="Harp environment sensor"
    )
    manipulator: aind_manipulator.AindManipulator = Field(description="Manipulator")
    calibration: RigCalibration = Field(description="Calibration models")

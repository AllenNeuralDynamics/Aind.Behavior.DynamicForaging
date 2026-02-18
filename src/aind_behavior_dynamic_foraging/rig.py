from typing import Annotated, Literal, Optional

import aind_behavior_services.rig as rig
from aind_behavior_services.rig import aind_manipulator, cameras, harp, water_valve
from pydantic import BaseModel, Field, model_validator

from aind_behavior_dynamic_foraging import __semver__


class RigCalibration(BaseModel):
    """Container class for calibration models. In a future release these will be moved to the respective devices"""

    water_valve_left: water_valve.WaterValveCalibration = Field(description="Left water valve calibration")
    water_valve_right: water_valve.WaterValveCalibration = Field(description="Right water valve calibration")


_SoundIndex = Annotated[int, Field(ge=3, le=31, description="Index of the sound to play on the sound card")]


class Waveform(BaseModel):
    """Model for a waveform to be played on the sound card."""

    waveform_type: Literal["sine", "white_noise"] = Field(default="sine", description="Type of the waveform")
    index: _SoundIndex = Field(description="Index of the waveform on the sound card")
    duration: float = Field(description="Duration of the waveform in seconds", gt=0)
    frequency: Optional[float] = Field(
        default=None, description="Frequency of the sine wave in Hz. Required if waveform_type is 'sine'."
    )

    @model_validator(mode="after")
    def validate_waveform(self: "Waveform"):
        if self.waveform_type == "sine" and self.frequency is None:
            raise ValueError("Frequency must be provided for sine waveforms.")
        return self


class SoundCardCalibration(BaseModel):
    """Calibration model for the sound card. Contains the waveforms to play for each cue."""

    go_cue: Waveform = Field(
        default=Waveform(index=3, duration=0.1, frequency=7500),
        description="Waveform to play for go cue",
        validate_default=True,
    )
    cs_plus: Waveform = Field(
        default=Waveform(index=4, duration=0.1, frequency=13000),
        description="Waveform to play for CS+ cue",
        validate_default=True,
    )
    cs_minus: Waveform = Field(
        default=Waveform(waveform_type="white_noise", index=5, duration=0.1),
        description="Waveform to play for CS- cue",
        validate_default=True,
    )


class DynamicForagingSoundCard(harp.HarpSoundCard):
    """A calibrated sound card for the dynamic foraging rig. This is a subclass of the HarpSoundCard that includes the sound card calibration."""

    calibration: SoundCardCalibration = Field(
        default=SoundCardCalibration(), description="Sound card calibration", validate_default=True
    )


class AindDynamicForagingRig(rig.Rig):
    version: Literal[__semver__] = __semver__
    triggered_camera_controller: cameras.CameraController[cameras.SpinnakerCamera] = Field(
        description="Required camera controller to triggered cameras."
    )
    monitoring_camera_controller: Optional[cameras.CameraController[cameras.WebCamera]] = Field(
        default=None, description="Optional camera controller for monitoring cameras."
    )
    harp_behavior: harp.HarpBehavior = Field(description="Harp behavior")
    harp_lickometer_left: harp.HarpLicketySplit = Field(description="Harp left lickometer")
    harp_lickometer_right: harp.HarpLicketySplit = Field(description="Harp right lickometer")
    harp_clock_generator: harp.HarpWhiteRabbit = Field(description="Harp clock generator")
    harp_sound_card: DynamicForagingSoundCard = Field(description="Harp sound card")
    harp_sniff_detector: Optional[harp.HarpSniffDetector] = Field(default=None, description="Harp sniff detector")
    harp_environment_sensor: Optional[harp.HarpEnvironmentSensor] = Field(
        default=None, description="Harp environment sensor"
    )
    manipulator: aind_manipulator.AindManipulator = Field(description="Manipulator")
    calibration: RigCalibration = Field(description="Calibration models")

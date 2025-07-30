from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union
from enum import Enum


class UnRetractSpeed(Enum):
    SLOW = 0
    NORMAL = 1
    FAST = 2


STAGE_NAMES = Literal["newscale", "AIND"]


class AutoStop(BaseModel):
    ignore_win: int = Field(default=30, title="Window of trials to check ignored responses")
    ignore_ratio_threshold: float = Field(default=.8,
                                          title="Threshold for acceptable ignored trials within window.",
                                          ge=0, le=1)
    max_trial: int = Field(default=1000, title="Maximal number of trials")
    max_time: int = Field(default=75, title="Maximal session time (min)")
    min_time: int = Field(default=30, title="Minimum session time (min)")


class StageSpecs(BaseModel):
    stage_name: STAGE_NAMES = Field(title="Name of stage used")
    rig_name: str = Field(title="Name of rig used")
    step_size: Optional[float] = Field(default=None, title="Stage step size")
    x: Optional[float] = Field(default=None, title="X position of stage")
    y: Optional[float] = Field(default=None, title="Y position of stage")
    z: Optional[float] = Field(default=None, title="Z position of stage")


class LickSpoutRetractionSpecs(BaseModel):
    wait_time: float = Field(default=3, description="Wait time in seconds before lick spout is un-retracted.")
    un_retract_speed: UnRetractSpeed = Field(default=UnRetractSpeed.NORMAL,
                                             description="Speed of lick spout retraction")

class LickSpoutMovement(BaseModel):
    trial_interval: int = Field(50, description="Trial interval to evaluate position.")
    bias_lower_threshold: float = Field(default=.3, description="Value which lick spout will move towards origin if "
                                                                "bias drops below.")
    bias_upper_threshold: float = Field(default=.7, description="Value which lick spout will move away from origin if "
                                                                "bias goes above.")
    range_um: float = Field(default=300, description="+/- range lick spout can travel in um")
    step_size_um: float = Field(default=50, description="Step size for moving lick spout if bias is outside thresholds")

class WaterReward(BaseModel):
    trial_interval: int = Field(50, description="Trial interval to evaluate reward.")
    bias_upper_threshold: float = Field(default=.7, description="Value which water will be given if bias exceeds")
    n_choices: int = Field(20, description="Last N choices to evaluate if all are on the lowest probability side")
    volume_ul: int = Field(5, description="Volume in ul to deliver")

class BiasCorrection(BaseModel):
    trial_buffer: int = Field(default=20, description="Buffer between water and lickspout movement to avoid over "
                                                      "correction.")
    lick_spout_movement: Optional[LickSpoutMovement] = Field(default=LickSpoutMovement(),
                                                             description="Lick spout movement to correct for bias.")
    water_reward: Optional[WaterReward] = Field(default=WaterReward(),
                                                    description="Water reward to correct for bias.")

class OperationalControl(BaseModel):
    name: Literal["OperationalControl"] = Field(default="OperationalControl", frozen=True)
    auto_stop: AutoStop = Field(default=AutoStop(), description="Parameters describing auto stop.")
    stage_specs: Optional[StageSpecs] = Field(default=None, description="Stage positions related to session.")
    lick_spout_retraction_specs: LickSpoutRetractionSpecs = Field(default=LickSpoutRetractionSpecs(),
                                                                  description="Lick spout retraction settings"
                                                                              "related to session.")
    bias_correction: BiasCorrection = Field(default=BiasCorrection(),
                                            description="Lick spout movement to correct for"
                                                        " bias.")

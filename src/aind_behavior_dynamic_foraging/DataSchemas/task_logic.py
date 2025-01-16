from __future__ import annotations

from enum import Enum
from typing import Literal

from aind_behavior_services.task_logic import AindBehaviorTaskLogicModel, TaskParameters
from pydantic import Field

__version__ = "0.1.0"


class AdvancedBlockMode(str, Enum):
    ''' Modes for advanced block '''
    OFF = "off"
    NOW = "now"
    ONCE = "once"


class AutoWaterMode(str, Enum):
    ''' Modes for auto water '''
    NATURAL = "Natural"
    BOTH = "Both"
    HIGH_PRO = "High pro"

class AindDynamicForagingTaskParameters(TaskParameters):
    # --- Critical training parameters ---
    # Reward probability
    BaseRewardSum: float = Field(..., title="Sum of p_reward")
    RewardFamily: int = Field(..., title="Reward family")  # Should be explicit here
    RewardPairsN: int = Field(..., title="Number of pairs")  # Should be explicit here

    UncoupledReward: str = Field("0.1,0.3,0.7", title="Uncoupled reward")  # For uncoupled tasks only

    # Randomness
    Randomness: str = Field('Exponential', title="Randomness mode")  # Exponential by default

    # Block length
    BlockMin: int = Field(..., title="Block length (min)")
    BlockMax: int = Field(..., title="Block length (max)")
    BlockBeta: int = Field(..., title="Block length (beta)")
    BlockMinReward: int = Field(1, title="Minimal rewards in a block to switch")

    # Delay period
    DelayMin: float = Field(..., title="Delay period (min) ")
    DelayMax: float = Field(..., title="Delay period (max) ")
    DelayBeta: float = Field(..., title="Delay period (beta)")

    # Reward delay
    RewardDelay: float = Field(..., title="Reward delay (sec)")

    # Auto water
    AutoReward: bool = Field(..., title="Auto reward switch")
    AutoWaterType: AutoWaterMode = Field(AutoWaterMode.NATURAL, title="Auto water mode")
    Multiplier: float = Field(..., title="Multiplier for auto reward")
    Unrewarded: int = Field(..., title="Number of unrewarded trials before auto water")
    Ignored: int = Field(..., title="Number of ignored trials before auto water")

    # ITI
    ITIMin: float = Field(..., title="ITI (min)")
    ITIMax: float = Field(..., title="ITI (max)")
    ITIBeta: float = Field(..., title="ITI (beta)")
    ITIIncrease: float = Field(0.0, title="ITI increase")  # TODO: not implemented in the GUI??

    # Response time
    ResponseTime: float = Field(..., title="Response time")
    RewardConsumeTime: float = Field(..., title="Reward consume time",
                                     description="Time of the no-lick period before trial end")
    StopIgnores: int = Field(..., title="Number of ignored trials before stop")

    # Auto block
    AdvancedBlockAuto: AdvancedBlockMode = Field(..., title="Auto block mode")
    SwitchThr: float = Field(..., title="Switch threshold for auto block")
    PointsInARow: int = Field(..., title="Points in a row for auto block")

    # Auto stop
    MaxTrial: int = Field(..., title="Maximal number of trials")
    MaxTime: int = Field(..., title="Maximal session time (min)")

    # Reward size
    RightValue_volume: float = Field(3.00, title="Right reward size (uL)")
    LeftValue_volume: float = Field(3.00, title="Left reward size (uL)")

    # Warmup
    warmup: str = Field('off', title="Warmup master switch")
    warm_min_trial: int = Field(50, title="Warmup finish criteria: minimal trials")
    warm_max_choice_ratio_bias: float = Field(0.1, title="Warmup finish criteria: maximal choice ratio bias from 0.5")
    warm_min_finish_ratio: float = Field(0.8, title="Warmup finish criteria: minimal finish ratio")
    warm_windowsize: int = Field(20, title="Warmup finish criteria: window size to compute the bias and ratio")


class AindDynamicForagingTaskLogic(AindBehaviorTaskLogicModel):
    version: Literal[__version__] = __version__
    name: Literal["AindDynamicForaging"] = Field(default="AindDynamicForaging", description="Name of the task logic", frozen=True)
    task_parameters: AindDynamicForagingTaskParameters = Field(..., description="Parameters of the task logic")

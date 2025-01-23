from __future__ import annotations

from enum import Enum
from typing import Literal

from aind_behavior_services.task_logic import AindBehaviorTaskLogicModel, TaskParameters
from pydantic import Field

__version__ = "0.0.0"


class AdvancedBlockMode(str, Enum):
    """ Modes for advanced block """

    OFF = "off"
    NOW = "now"
    ONCE = "once"


class AutoWaterMode(str, Enum):
    """Modes for auto water """
    NATURAL = "Natural"
    BOTH = "Both"
    HIGH_PRO = "High pro"

class AindDynamicForagingTaskParameters(TaskParameters):
    # --- Critical training parameters ---
    name:  Literal["Aind Dynamic Foraging Task Parameters"] = "Aind Dynamic Foraging Task Parameters"
    # Reward probability
    base_reward_sum: float = Field(..., title="Sum of p_reward")
    reward_family: int = Field(..., title="Reward family")  # Should be explicit here
    reward_pairs_n: int = Field(..., title="Number of pairs")  # Should be explicit here

    uncoupled_reward: str = Field("0.1,0.3,0.7", title="Uncoupled reward")  # For uncoupled tasks only

    # Randomness
    randomness: str = Field('Exponential', title="Randomness mode")  # Exponential by default

    # Block length
    block_min: int = Field(..., title="Block length (min)")
    block_max: int = Field(..., title="Block length (max)")
    block_beta: int = Field(..., title="Block length (beta)")
    block_min_reward: int = Field(1, title="Minimal rewards in a block to switch")

    # Delay period
    delay_min: float = Field(..., title="Delay period (min) ")
    delay_max: float = Field(..., title="Delay period (max) ")
    delay_beta: float = Field(..., title="Delay period (beta)")

    # Reward delay
    reward_delay: float = Field(..., title="Reward delay (sec)")

    # Auto water
    auto_reward: bool = Field(..., title="Auto reward switch")
    auto_water_type: AutoWaterMode = Field(AutoWaterMode.NATURAL, title="Auto water mode")
    multiplier: float = Field(..., title="Multiplier for auto reward")
    unrewarded: int = Field(..., title="Number of unrewarded trials before auto water")
    ignored: int = Field(..., title="Number of ignored trials before auto water")

    # ITI
    iti_min: float = Field(..., title="ITI (min)")
    iti_max: float = Field(..., title="ITI (max)")
    iti_beta: float = Field(..., title="ITI (beta)")
    iti_increase: float = Field(0.0, title="ITI increase")  # TODO: not implemented in the GUI??

    # Response time
    response_time: float = Field(..., title="Response time")
    reward_consume_time: float = Field(..., title="Reward consume time",
                                     description="Time of the no-lick period before trial end")

    auto_stop_ignore_win: int = Field(..., title="Window of trials to check ignored responses")
    auto_stop_ignore_ratio_threshold: float = Field(..., title="Threshold for acceptable ignored trials within window.")

    # Auto block
    advanced_block_auto: AdvancedBlockMode = Field(..., title="Auto block mode")
    switch_thr: float = Field(..., title="Switch threshold for auto block")
    points_in_a_row: int = Field(..., title="Points in a row for auto block")

    # Auto stop
    max_trial: int = Field(..., title="Maximal number of trials")
    max_time: int = Field(..., title="Maximal session time (min)")

    # Reward size
    right_value_volume: float = Field(3.00, title="Right reward size (uL)")
    left_value_volume: float = Field(3.00, title="Left reward size (uL)")

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

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

from aind_behavior_services.task_logic import AindBehaviorTaskLogicModel, TaskParameters


__version__ = "0.1.0"

advanced_block_autos = Literal["now", "once"]
RANDOMNESSES = Literal["Exponential", "Even"]

class BlockParameters(BaseModel):
    # Block length
    min: int = Field(default=20, title="Block length (min)")
    max: int = Field(default=60, title="Block length (max)")
    beta: int = Field(default=20, title="Block length (beta)")
    min_reward: int = Field(default=1, title="Minimal rewards in a block to switch")


class RewardProbability(BaseModel):
    base_reward_sum: float = Field(default=.8, title="Sum of p_reward")
    family: int = Field(default=1, title="Reward family")  # Should be explicit here
    pairs_n: int = Field(default=1, title="Number of pairs")  # Should be explicit here


class DelayPeriod(BaseModel):
    min: float = Field(default=0.0, title="Delay period (min) ")
    max: float = Field(default=1.0, title="Delay period (max) ")
    beta: float = Field(default=1.0, title="Delay period (beta)")


class AutoWaterMode(str, Enum):
    """Modes for auto water """
    NATURAL = "Natural"
    BOTH = "Both"
    HIGH_PRO = "High pro"


class AutoWater(BaseModel):
    auto_water_type: AutoWaterMode = Field(default=AutoWaterMode.NATURAL, title="Auto water mode")
    multiplier: float = Field(default=.8, title="Multiplier for auto reward")
    unrewarded: int = Field(default=200, title="Number of unrewarded trials before auto water")
    ignored: int = Field(default=100, title="Number of ignored trials before auto water")
    include_reward: bool = Field(default=False, description="Include auto water in total rewards.")


class InterTrialInterval(BaseModel):
    min: float = Field(default=1.0, title="ITI (min)")
    max: float = Field(default=8.0, title="ITI (max)")
    beta: float = Field(default=2.0, title="ITI (beta)")
    increase: float = Field(default=0.0, title="ITI increase")  # TODO: not implemented in the GUI??


class ResponseTime(BaseModel):
    response_time: float = Field(default=1.0, title="Response time")
    reward_consume_time: float = Field(default=3.0, title="Reward consume time",
                                       description="Time of the no-lick period before trial end")


class AutoStop(BaseModel):
    ignore_win: int = Field(default=30, title="Window of trials to check ignored responses")
    ignore_ratio_threshold: float = Field(default=.8,
                                          title="Threshold for acceptable ignored trials within window.",
                                          ge=0, le=1)
    max_trial: int = Field(default=1000, title="Maximal number of trials")
    max_time: int = Field(default=120, title="Maximal session time (min)")
    min_time: int = Field(default=30, title="Minimum session time (min)")


class AutoBlock(BaseModel):
    advanced_block_auto: Literal["now", "once"] = Field(default="now", title="Auto block mode")
    switch_thr: float = Field(default=.5, title="Switch threshold for auto block")
    points_in_a_row: int = Field(default=5, title="Points in a row for auto block")


class RewardSize(BaseModel):
    right_value_volume: float = Field(default=3.00, title="Right reward size (uL)")
    left_value_volume: float = Field(default=3.00, title="Left reward size (uL)")


class Warmup(BaseModel):
    min_trial: int = Field(default=50, title="Warmup finish criteria: minimal trials")
    max_choice_ratio_bias: float = Field(default=0.1,
                                         title="Warmup finish criteria: maximal choice ratio bias from 0.5")
    min_finish_ratio: float = Field(default=0.8, title="Warmup finish criteria: minimal finish ratio")
    windowsize: int = Field(default=20, title="Warmup finish criteria: window size to compute the bias and ratio")

class RewardN(BaseModel):
    min_reward: int = Field(default=60, description="The minimum reward amount to enter the next block.")
    initial_inactive_trials: int = Field(default=2, description="Initial N trials of the active side where no "
                                                                 "bait will be be given.")

class AindDynamicForagingTaskParameters(TaskParameters):
    block_parameters: BlockParameters = Field(default=BlockParameters(),
                                              description="Parameters describing block conditions.")
    reward_probability: RewardProbability = Field(default=RewardProbability(),
                                                  description="Parameters describing reward_probability.")
    uncoupled_reward: Optional[list[float]] = Field(default=[0.1, 0.3, 0.7], title="Uncoupled reward", min_length=3,
                                                    max_length=3)  # For uncoupled tasks only
    randomness: RANDOMNESSES = Field(default="Exponential", title="Randomness mode")
    delay_period: DelayPeriod = Field(default=DelayPeriod(), description="Parameters describing delay period.")
    reward_delay: float = Field(default=0, title="Reward delay (sec)")
    auto_water: Optional[AutoWater] = Field(default=None, description="Parameters describing auto water.")
    inter_trial_interval: InterTrialInterval = Field(default_factory=InterTrialInterval, validate_default=True,
                                                     description="Parameters describing iti.")
    response_time: ResponseTime = Field(default=ResponseTime(), description="Parameters describing response time.")
    auto_stop: AutoStop = Field(default=AutoStop(), description="Parameters describing auto stop.")
    auto_block: Optional[AutoBlock] = Field(default=None,
                                            description="Parameters describing auto advancement to next block.")
    reward_size: RewardSize = Field(default=RewardSize(), description="Parameters describing reward size.")
    warmup: Optional[Warmup] = Field(default=None, description="Parameters describing warmup.")
    no_response_trial_addition: bool = Field(default=True,
                                             description="Add one trial to the block length on both lickspouts.")
    reward_n: Optional[RewardN] = Field(default=RewardN())


class AindDynamicForagingTaskLogic(AindBehaviorTaskLogicModel):
    version: Literal[__version__] = __version__
    name: Literal["AindDynamicForaging"] = Field(default="AindDynamicForaging", description="Name of the task logic",
                                                 frozen=True)
    task_parameters: AindDynamicForagingTaskParameters = Field(..., description="Parameters of the task logic")

    # @model_validator(mode="after")
    # def _reward_n_and_uncouple_check(self):
    #     """Check that reward_n and uncouple_reward do not both exist"""
    #     has_reward_n = self.task_parameters.reward_n is not None
    #     has_uncoupled_reward = self.task_parameters.uncoupled_reward is not None
    #
    #     if has_reward_n and has_uncoupled_reward:
    #         raise ValueError("Invalid task with both reward_n and uncoupled_reward defined.")
    #
    #     return self
from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Self, Union
from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator

from aind_behavior_services.task_logic import AindBehaviorTaskLogicModel, TaskParameters
from pydantic import Field

__version__ = "0.1.0"

class _NeuralExperimentTypeBase(BaseModel):
    experiment_type: str


class Fiber_Photometry(_NeuralExperimentTypeBase):
    experiment_type: Literal["Fiber_Photometry"] = "Fiber_Photometry"
    # more parameters to define experiment
    # mode
    # recording type
    # baseline time

class Optogenetics(_NeuralExperimentTypeBase):
    experiment_type: Literal["Optogenetics"] = "Optogenetics"
    # more parameters to define experiment
    # laser -> list of class that defines lasers

class Ephys(_NeuralExperimentTypeBase):
    experiment_type: Literal["Ephys"] = "Ephys"
    # more parameters to define experiment
    # probes -> list of class that defines probes
    
class Behavior(_NeuralExperimentTypeBase):
    experiment_type: Literal["Behavior"] = "Behavior"
    # more parameters to define experiment if any

class NeuralExperimentTypes(RootModel):
    root: Annotated[
        Union[
            Behavior,
            Fiber_Photometry,
            Optogenetics,
            Ephys,
    ],
        Field(discriminator="experiment_type"),
    ]

class BlockParameters(BaseModel):
    # Block length
    block_min: int = Field(..., title="Block length (min)")
    block_max: int = Field(..., title="Block length (max)")
    block_beta: int = Field(..., title="Block length (beta)")
    block_min_reward: int = Field(1, title="Minimal rewards in a block to switch")

class RewardProbability(BaseModel):
    base_reward_sum: float = Field(..., title="Sum of p_reward")
    reward_family: int = Field(..., title="Reward family")  # Should be explicit here
    reward_pairs_n: int = Field(..., title="Number of pairs")  # Should be explicit here

class DelayPeriod(BaseModel):
    delay_min: float = Field(..., title="Delay period (min) ")
    delay_max: float = Field(..., title="Delay period (max) ")
    delay_beta: float = Field(..., title="Delay period (beta)")

class AutoWaterMode(str, Enum):
    """Modes for auto water """
    NATURAL = "Natural"
    BOTH = "Both"
    HIGH_PRO = "High pro"

class AutoWater(BaseModel):
    auto_reward: bool = Field(..., title="Auto reward switch")
    auto_water_type: AutoWaterMode = Field(AutoWaterMode.NATURAL, title="Auto water mode")
    multiplier: float = Field(..., title="Multiplier for auto reward")
    unrewarded: int = Field(..., title="Number of unrewarded trials before auto water")
    ignored: int = Field(..., title="Number of ignored trials before auto water")

class InterTrialInterval(BaseModel):
    iti_min: float = Field(..., title="ITI (min)")
    iti_max: float = Field(..., title="ITI (max)")
    iti_beta: float = Field(..., title="ITI (beta)")
    iti_increase: float = Field(0.0, title="ITI increase")  # TODO: not implemented in the GUI??

class ResponseTime(BaseModel):
    response_time: float = Field(..., title="Response time")
    reward_consume_time: float = Field(..., title="Reward consume time",
                                       description="Time of the no-lick period before trial end")

class AutoStop(BaseModel):
    auto_stop_ignore_win: int = Field(..., title="Window of trials to check ignored responses")
    auto_stop_ignore_ratio_threshold: float = Field(..., title="Threshold for acceptable ignored trials within window.")
    max_trial: int = Field(..., title="Maximal number of trials")
    max_time: int = Field(..., title="Maximal session time (min)")

class AdvancedBlockMode(str, Enum):
    """ Modes for advanced block """

    OFF = "off"
    NOW = "now"
    ONCE = "once"

class AutoBlock(BaseModel):
    advanced_block_auto: AdvancedBlockMode = Field(..., title="Auto block mode")
    switch_thr: float = Field(..., title="Switch threshold for auto block")
    points_in_a_row: int = Field(..., title="Points in a row for auto block")

class RewardSize(BaseModel):
    right_value_volume: float = Field(3.00, title="Right reward size (uL)")
    left_value_volume: float = Field(3.00, title="Left reward size (uL)")

class Warmup(BaseModel):
    warmup_state: str = Field('off', title="Warmup master switch")
    warm_min_trial: int = Field(50, title="Warmup finish criteria: minimal trials")
    warm_max_choice_ratio_bias: float = Field(0.1, title="Warmup finish criteria: maximal choice ratio bias from 0.5")
    warm_min_finish_ratio: float = Field(0.8, title="Warmup finish criteria: minimal finish ratio")
    warm_windowsize: int = Field(20, title="Warmup finish criteria: window size to compute the bias and ratio")

class AindDynamicForagingTaskParameters(TaskParameters):

    neural_experiments: List[NeuralExperimentTypes] = Field(..., description="List of modalities included in session.")
    block_parameters: BlockParameters = Field(..., description="Parameters describing block conditions.")
    reward_probability: RewardProbability = Field(..., description="Parameters describing reward_probability.")
    uncoupled_reward: list[float] = Field([0.1,0.3,0.7], title="Uncoupled reward", min_length=3, max_length=3)  # For uncoupled tasks only
    randomness: str = Field('Exponential', title="Randomness mode")  # Exponential by default
    delay_period: DelayPeriod = Field(..., description="Parameters describing delay period.")
    reward_delay: float = Field(..., title="Reward delay (sec)")
    auto_water: AutoWater = Field(..., description="Parameters describing auto water.")
    inter_trial_interval: InterTrialInterval = Field(..., description="Parameters describing iti.")
    response_time: ResponseTime = Field(..., description="Parameters describing response time.")
    auto_stop: AutoStop = Field(..., description="Parameters describing auto stop.")
    auto_block: AutoBlock = Field(..., description="Parameters describing auto advancement to next block.")
    reward_size: RewardSize = Field(..., description="Parameters describing reward size.")
    warmup: Warmup = Field(..., description="Parameters describing warmup.")

class AindDynamicForagingTaskLogic(AindBehaviorTaskLogicModel):
    version: Literal[__version__] = __version__
    name: Literal["AindDynamicForaging"] = Field(default="AindDynamicForaging", description="Name of the task logic", frozen=True)
    task_parameters: AindDynamicForagingTaskParameters = Field(..., description="Parameters of the task logic")
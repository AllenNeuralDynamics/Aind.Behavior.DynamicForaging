from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Self, Union

import aind_behavior_services.task_logic.distributions as distributions
from aind_behavior_services.task_logic import AindBehaviorTaskLogicModel, TaskParameters
from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator

__version__ = "0.5.0"


def scalar_value(value: float) -> distributions.Scalar:
    """
    Helper function to create a scalar value distribution for a given value.

    Args:
        value (float): The value of the scalar distribution.

    Returns:
        distributions.Scalar: The scalar distribution type.
    """
    return distributions.Scalar(distribution_parameters=distributions.ScalarDistributionParameter(value=value))


class Size(BaseModel):
    width: float = Field(default=0, description="Width of the texture")
    height: float = Field(default=0, description="Height of the texture")


class Vector2(BaseModel):
    x: float = Field(default=0, description="X coordinate of the point")
    y: float = Field(default=0, description="Y coordinate of the point")


class Vector3(BaseModel):
    x: float = Field(default=0, description="X coordinate of the point")
    y: float = Field(default=0, description="Y coordinate of the point")
    z: float = Field(default=0, description="Z coordinate of the point")


# Updaters
class NumericalUpdaterOperation(str, Enum):
    NONE = "None"
    OFFSET = "Offset"
    GAIN = "Gain"
    SET = "Set"
    OFFSETPERCENTAGE = "OffsetPercentage"


class NumericalUpdaterParameters(BaseModel):
    initial_value: float = Field(default=0.0, description="Initial value of the parameter")
    increment: float = Field(default=0.0, description="Value to increment the parameter by")
    decrement: float = Field(default=0.0, description="Value to decrement the parameter by")
    minimum: float = Field(default=0.0, description="Minimum value of the parameter")
    maximum: float = Field(default=0.0, description="Minimum value of the parameter")


class NumericalUpdater(BaseModel):
    operation: NumericalUpdaterOperation = Field(
        default=NumericalUpdaterOperation.NONE, description="Operation to perform on the parameter"
    )
    parameters: NumericalUpdaterParameters = Field(
        default=NumericalUpdaterParameters(), description="Parameters of the updater"
    )


class OperantLogic(BaseModel):
    is_operant: bool = Field(default=True, description="Will the trial implement operant logic")
    time_to_collect_reward: float = Field(
        default=100000, ge=0, description="Time(s) the animal has to collect the reward"
    )


class PowerFunction(BaseModel):
    function_type: Literal["PowerFunction"] = "PowerFunction"
    minimum: float = Field(default=0, description="Minimum value of the function")
    maximum: float = Field(default=1, description="Maximum value of the function")
    a: float = Field(default=1, description="Coefficient a of the function: value = a * pow(b, c * x) + d")
    b: float = Field(
        default=2.718281828459045, description="Coefficient b of the function: value = a * pow(b, c * x) + d"
    )
    c: float = Field(default=-1, description="Coefficient c of the function: value = a * pow(b, c * x) + d")
    d: float = Field(default=0, description="Coefficient d of the function: value = a * pow(b, c * x) + d")


class LinearFunction(BaseModel):
    function_type: Literal["LinearFunction"] = "LinearFunction"
    minimum: float = Field(default=0, description="Minimum value of the function")
    maximum: float = Field(default=9999, description="Maximum value of the function")
    a: float = Field(default=1, description="Coefficient a of the function: value = a * x + b")
    b: float = Field(default=0, description="Coefficient b of the function: value = a * x + b")


class ConstantFunction(BaseModel):
    function_type: Literal["ConstantFunction"] = "ConstantFunction"
    value: float = Field(default=1, description="Value of the function")


class LookupTableFunction(BaseModel):
    function_type: Literal["LookupTableFunction"] = "LookupTableFunction"
    lut_keys: List[float] = Field(..., description="List of keys of the lookup table", min_length=1)
    lut_values: List[float] = Field(..., description="List of values of the lookup table", min_length=1)

    @model_validator(mode="after")
    def _validate_lut(self) -> Self:
        if len(self.lut_keys) != len(self.lut_values):
            raise ValueError("The number of keys and values must be the same.")
        return self


class RewardFunction(RootModel):
    root: Annotated[
        Union[ConstantFunction, LinearFunction, PowerFunction, LookupTableFunction],
        Field(discriminator="function_type"),
    ]


class DepletionRule(str, Enum):
    ON_REWARD = "OnReward"
    ON_CHOICE = "OnChoice"
    ON_TIME = "OnTime"
    ON_DISTANCE = "OnDistance"


class PatchRewardFunction(BaseModel):
    amount: RewardFunction = Field(
        default=ConstantFunction(value=1),
        description="Determines the amount of reward to be delivered. The value is in microliters",
        validate_default=True,
    )
    probability: RewardFunction = Field(
        default=ConstantFunction(value=1),
        description="Determines the probability that a reward will be delivered",
        validate_default=True,
    )
    available: RewardFunction = Field(
        default=LinearFunction(minimum=0, a=-1, b=5),
        description="Determines the total amount of reward available left in the patch. The value is in microliters",
        validate_default=True,
    )
    depletion_rule: DepletionRule = Field(default=DepletionRule.ON_CHOICE, description="Depletion")


class RewardSpecification(BaseModel):
    operant_logic: Optional[OperantLogic] = Field(default=None, description="The optional operant logic of the reward")
    delay: distributions.Distribution = Field(
        default=scalar_value(0),
        description="The optional distribution where the delay to reward will be drawn from",
        validate_default=True,
    )
    reward_function: PatchRewardFunction = Field(
        default=PatchRewardFunction(), description="Reward function of the patch."
    )


class PatchStatistics(BaseModel):
    label: str = Field(default="", description="Label of the patch")
    state_index: int = Field(default=0, ge=0, description="Index of the state")
    reward_specification: Optional[RewardSpecification] = Field(
        default=None, description="The optional reward specification of the patch"
    )


class EnvironmentStatistics(BaseModel):
    patches: List[PatchStatistics] = Field(default_factory=list, description="List of patches", min_items=1)

class PositionControl(BaseModel):
    gain: Vector3 = Field(default=Vector3(x=1, y=1, z=1), description="Gain of the position control.")
    initial_position: Vector3 = Field(default=Vector3(x=0, y=2.56, z=0), description="Gain of the position control.")
    frequency_filter_cutoff: float = Field(
        default=0.5,
        ge=0,
        le=100,
        description="Cutoff frequency (Hz) of the low-pass filter used to filter the velocity signal.",
    )

class AudioControl(BaseModel):
    duration: float = Field(default=0.2, ge=0, description="Duration")
    frequency: float = Field(default=1000, ge=100, description="Frequency")


class OperationControl(BaseModel):
    position_control: PositionControl = Field(
        default=PositionControl(), description="Control of the position", validate_default=True
    )
    audio_control: AudioControl = Field(
        default=AudioControl(), description="Control of the audio", validate_default=True
    )


class TaskMode(str, Enum):
    DEBUG = "DEBUG"
    HABITUATION = "HABITUATION"
    FORAGING = "FORAGING"


class TaskModeSettingsBase(BaseModel):
    task_mode: TaskMode = Field(default=TaskMode.FORAGING, description="Stage of the task")


class HabituationSettings(TaskModeSettingsBase):
    task_mode: Literal[TaskMode.HABITUATION] = TaskMode.HABITUATION


class ForagingSettings(TaskModeSettingsBase):
    task_mode: Literal[TaskMode.FORAGING] = TaskMode.FORAGING


class TaskModeSettings(RootModel):
    root: Annotated[Union[HabituationSettings, ForagingSettings], Field(discriminator="task_mode")]


class _BlockEndConditionBase(BaseModel):
    condition_type: str


class BlockEndConditionDuration(_BlockEndConditionBase):
    condition_type: Literal["Duration"] = "Duration"
    value: distributions.Distribution = Field(..., description="Time after which the block ends.")


class BlockEndConditionChoice(_BlockEndConditionBase):
    condition_type: Literal["Choice"] = "Choice"
    value: distributions.Distribution = Field(..., description="Number of choices after which the block ends.")


class BlockEndConditionReward(_BlockEndConditionBase):
    condition_type: Literal["Reward"] = "Reward"
    value: distributions.Distribution = Field(..., description="Number of rewards after which the block ends.")



class BlockEndCondition(RootModel):
    root: Annotated[
        Union[
            BlockEndConditionDuration,
            BlockEndConditionChoice,
            BlockEndConditionReward,
        ],
        Field(discriminator="condition_type"),
    ]


class Block(BaseModel):
    environment_statistics: EnvironmentStatistics = Field(..., description="Statistics of the environment")
    end_conditions: List[BlockEndCondition] = Field(
        [], description="List of end conditions that must be true for the block to end."
    )


class _BlockAdvanceConditionBase(BaseModel):
    condition_type: str

class BlockAdvanceConditionSwitchThreshold(_BlockAdvanceConditionBase):
    condition_type: Literal["SwitchThreshold"] = "SwitchThreshold"
    value: distributions.Distribution = Field(..., description="?")

class BlockAdvanceConditionPointsInARow(_BlockAdvanceConditionBase):
    condition_type: Literal["PointsInARow"] = "PointsInARow"
    value: distributions.Distribution = Field(..., description="?")

class BlockAdvanceCondition(RootModel):
    root: Annotated[
        Union[
            BlockAdvanceConditionSwitchThreshold,
            BlockAdvanceConditionPointsInARow
        ],
        Field(discriminator="condition_type"),
    ]

class AutoAdvance(BaseModel):
    is_auto_advance: bool = Field(default=True, description="Will the block be auto advanced")
    advance_conditions: List[BlockAdvanceCondition] = Field(
        [], description="List of advance conditions that must be true for the block to advance."
    )

class BlockStructure(BaseModel):
    blocks: List[Block] = Field(..., description="Statistics of the environment", min_length=1)
    sampling_mode: Literal["Random", "Sequential"] = Field("Sequential", description="Sampling mode of the blocks.")
    auto_advance: AutoAdvance = Field(..., description="Auto advancement of blocks")


class AindDynamicForagingTaskParameters(TaskParameters):
    updaters: Dict[str, NumericalUpdater] = Field(default_factory=dict, description="List of numerical updaters")
    environment: BlockStructure = Field(..., description="Statistics of the environment")
    task_mode_settings: TaskModeSettings = Field(
        default=ForagingSettings(), description="Settings of the task stage", validate_default=True
    )
    operation_control: OperationControl = Field(..., description="Control of the operation")


class AindDynamicForagingTaskLogic(AindBehaviorTaskLogicModel):
    version: Literal[__version__] = __version__
    name: Literal["AindDynamicForaging"] = Field(default="AindDynamicForaging", description="Name of the task logic", frozen=True)
    task_parameters: AindDynamicForagingTaskParameters = Field(..., description="Parameters of the task logic")

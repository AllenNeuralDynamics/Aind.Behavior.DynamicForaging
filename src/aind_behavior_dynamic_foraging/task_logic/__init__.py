from typing import Literal, Optional

from aind_behavior_services.task import Task, TaskParameters
from pydantic import BaseModel, Field

from aind_behavior_dynamic_foraging import (
    __semver__,
)

from . import trial_models as trial_models
from .trial_generators import IntegrationTestTrialGeneratorSpec, TrialGeneratorSpec


class RewardSize(BaseModel):
    right_value_volume: float = Field(default=3.00, title="Right reward size (uL)")
    left_value_volume: float = Field(default=3.00, title="Left reward size (uL)")


# ==================== MAIN TASK LOGIC CLASSES ====================


class AindDynamicForagingTaskParameters(TaskParameters):
    """
    Complete parameter specification for the AIND Dynamic Foraging task.

    This class contains all configurable parameters for the Dynamic Foraging task,
    including environment structure, task mode settings, operation control,
    and numerical updaters for dynamic parameter modification.
    """

    reward_size: RewardSize = Field(default=RewardSize(), description="Parameters describing reward size.")
    lick_spout_retraction: Optional[bool] = Field(default=False, description="Lick spout retraction enabled.")
    trial_generator: TrialGeneratorSpec = Field(
        default=IntegrationTestTrialGeneratorSpec(),
        description="Trial generator model for generating trials in the task.",
        validate_default=True,
    )


class AindDynamicForagingTaskLogic(Task):
    """
    Main task logic model for the AIND Dynamic Foraging task.

    This is the top-level class that encapsulates the complete task logic
    specification for the dynamic foraging behavioral experiment.
    It includes all task parameters, environment specifications, and control settings.
    """

    version: Literal[__semver__] = __semver__
    name: Literal["AindDynamicForaging"] = Field(
        default="AindDynamicForaging", description="Name of the task logic", frozen=True
    )
    task_parameters: AindDynamicForagingTaskParameters = Field(description="Parameters of the task logic")


# We must rebuild these models after imports to resolve the forward reference in TrialGeneratorCompositeSpec
# See https://docs.pydantic.dev/latest/errors/usage_errors/#class-not-fully-defined for details
AindDynamicForagingTaskParameters.model_rebuild()
AindDynamicForagingTaskLogic.model_rebuild()

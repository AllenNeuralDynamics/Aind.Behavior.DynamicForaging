import logging
from typing import Literal

from aind_behavior_services.task.distributions import Distribution, Scalar
from pydantic import BaseModel, Field

from .base_coupled_trial_generator import (
    BaseCoupledTrialGenerator,
    BaseCoupledTrialGeneratorSpec,
)

logger = logging.getLogger(__name__)


class CoupledWarmupTrialGenerationEndConditions(BaseModel):
    min_trial: int = Field(default=50, ge=0, description="Minimum trials in generator.")
    max_choice_bias: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Maximum allowed deviation from 50/50 choice ratio to end trial generation.",
    )
    min_response_rate: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Minimum fraction of trials with a choice (non-ignored) to end trial generation.",
    )
    evaluation_window: int = Field(
        default=20, ge=0, description="Number of most recent trials to evaluate the end criteria."
    )


class CoupledWarmupTrialGeneratorSpec(BaseCoupledTrialGeneratorSpec):
    type: Literal["CoupledWarmupTrialGenerator"] = "CoupledWarmupTrialGenerator"

    block_len: Distribution = Field(
        default=Scalar(value=1),
        description="Distribution describing block length.",
    )

    trial_generation_end_parameters: CoupledWarmupTrialGenerationEndConditions = Field(
        default=CoupledWarmupTrialGenerationEndConditions(), description="Conditions to end trial generation."
    )
    is_baiting: Literal[True] = Field(
        default=True, description="Whether uncollected rewards carry over to the next trial."
    )

    def create_generator(self) -> "CoupledWarmupTrialGeneratorSpec":
        return CoupledWarmupTrialGenerator(self)


class CoupledWarmupTrialGenerator(BaseCoupledTrialGenerator):
    spec: CoupledWarmupTrialGeneratorSpec

    def _are_end_conditions_met(self) -> bool:
        """
        Check if end conditions are met to stop session
        """

        end_conditions = self.spec.trial_generation_end_parameters
        win = end_conditions.evaluation_window
        choice_history = self.is_right_choice_history[-win:] if win > 0 else self.is_right_choice_history

        choice_len = len(choice_history)
        left_choices = choice_history.count(False)
        right_choices = choice_history.count(True)
        unignored = left_choices + right_choices

        finish_ratio = 0 if choice_len == 0 else (unignored) / choice_len
        choice_ratio = 0 if unignored == 0 else right_choices / (unignored)
        if (
            len(self.is_right_choice_history) >= end_conditions.min_trial
            and finish_ratio >= end_conditions.min_response_rate
            and abs(choice_ratio - 0.5) <= end_conditions.max_choice_bias
        ):
            logger.debug(
                "Warmup trial generation end conditions met: "
                f"total trials={len(self.is_right_choice_history)}, "
                f"finish ratio={finish_ratio}, "
                f"choice bias={abs(choice_ratio - 0.5)}"
            )
            return True

        logger.debug(
            "Warmup trial generation end conditions are not met: "
            f"total trials={len(self.is_right_choice_history)}, "
            f"finish ratio={finish_ratio}, "
            f"choice bias={abs(choice_ratio - 0.5)}"
        )
        return False

    def _is_block_switch_allowed(self) -> True:
        """
        Warmup switches block every update

        Returns:
            True
        """

        return True

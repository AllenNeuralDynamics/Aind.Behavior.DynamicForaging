import logging
from typing import Literal

from aind_behavior_services.task.distributions import (
    ExponentialDistribution,
    ExponentialDistributionParameters,
    TruncationParameters,
)
from pydantic import BaseModel, Field

from ..trial_models import TrialOutcome
from .block_based_trial_generator import (
    BlockBasedTrialGenerator,
    BlockBasedTrialGeneratorSpec,
)

logger = logging.getLogger(__name__)


class WarmupTrialGenerationEndConditions(BaseModel):
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


class WarmupTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
    type: Literal["WarmupTrialGenerator"] = "WarmupTrialGenerator"

    block_len: ExponentialDistribution = Field(
        ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1),
            truncation_parameters=TruncationParameters(min=1, max=1),
        ),
        description="Distribution describing block length.",
    )

    trial_generation_end_parameters: WarmupTrialGenerationEndConditions = Field(
        default=WarmupTrialGenerationEndConditions(), description="Conditions to end trial generation."
    )
    min_block_reward: Literal[1] = Field(1, title="Minimal rewards in a block to switch")
    is_baiting: Literal[True] = Field(
        default=True, description="Whether uncollected rewards carry over to the next trial."
    )

    def create_generator(self) -> "WarmupTrialGenerator":
        return WarmupTrialGenerator(self)


class WarmupTrialGenerator(BlockBasedTrialGenerator):
    spec: WarmupTrialGeneratorSpec

    def _are_end_conditions_met(self) -> bool:
        """
        Check if end conditions are met to stop session
        """

        end_conditions = self.spec.trial_generation_end_parameters
        choice_history = self.is_right_choice_history

        choice_len = len(choice_history)
        left_choices = choice_history.count(False)
        right_choices = choice_history.count(True)
        unignored = left_choices + right_choices

        finish_ratio = 0 if choice_len == 0 else (unignored) / choice_len
        choice_ratio = 0 if unignored == 0 else right_choices / (unignored)

        if (
            choice_len >= end_conditions.min_trial
            and finish_ratio >= end_conditions.min_response_rate
            and abs(choice_ratio - 0.5) <= end_conditions.max_choice_bias
        ):
            logger.debug(
                "Warmup trial generation end conditions met: "
                f"total trials={choice_len}, "
                f"finish ratio={finish_ratio}, "
                f"choice bias={abs(choice_ratio - 0.5)}"
            )
            return True

        return False

    def update(self, outcome: TrialOutcome) -> None:
        """
        Warmup switches block every update

        :param outcome: trial outcome of previous trial
        """

        logger.info(f"Updating Warmup trial generator with trial outcome of {outcome}")

        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)
        self.trials_in_block += 1

        if self.spec.is_baiting:
            if outcome.is_right_choice:
                logger.debug("Resesting right bait.")
                self.is_right_baited = False
            elif not outcome.is_right_choice:
                logger.debug("Resesting left bait.")
                self.is_left_baited = False

        # warmup switches block each choice
        logger.info("Switching block.")
        self.trials_in_block = 0
        self.block = self.generate_next_block(
            reward_families=self.spec.reward_family,
            reward_family_index=self.spec.reward_probability_parameters.family,
            reward_pairs_n=self.spec.reward_probability_parameters.pairs_n,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            current_block=self.block,
            block_len=self.spec.block_len,
        )
        self.block_history.append(self.block)

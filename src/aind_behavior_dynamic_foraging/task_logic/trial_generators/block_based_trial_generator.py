import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
from aind_behavior_services.task.distributions import (
    Distribution,
    ExponentialDistribution,
    ExponentialDistributionParameters,
    TruncationParameters,
)
from aind_behavior_services.task.distributions_utils import draw_sample
from pydantic import BaseModel, Field

from ..trial_models import Trial
from ._base import BaseTrialGeneratorSpecModel, ITrialGenerator, TrialOutcome

logger = logging.getLogger(__name__)


class AutoWaterParameters(BaseModel):
    min_ignored_trials: int = Field(
        default=3, ge=0, description="Minimum consecutive ignored trials before auto water is triggered."
    )
    min_unrewarded_trials: int = Field(
        default=3, ge=0, description="Minimum consecutive unrewarded trials before auto water is triggered."
    )
    reward_fraction: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Fraction of full reward volume delivered during auto water (0=none, 1=full).",
    )


class RewardProbabilityParameters(BaseModel):
    """Defines the reward probability structure for a dynamic foraging task.

    Reward probabilities are defined as pairs (p_left, p_right) normalized by
    base_reward_sum. Pairs are drawn from a family representing a difficulty level:

        Family 1:   [[8, 1], [6, 1], [3, 1], [1, 1]]
        Family 2:  [[8, 1], [1, 1]]
        Family 3:  [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]
        Family 4:  [[6, 1], [3, 1], [1, 1]]

    """

    base_reward_sum: float = Field(
        default=0.8,
        description="Total reward probability shared between the two sides. Each reward pair is normalized to sum to this value.",
    )
    reward_pairs: list[list[float, float]] = Field(
        default=[[8, 1]],
        description="List of (left, right) reward ratio pairs to sample from during block transitions. ",
    )


class Block(BaseModel):
    p_right_reward: Optional[float] = Field(ge=0, le=1, description="Reward probability for right side during block.")
    p_left_reward: Optional[float] = Field(ge=0, le=1, description="Reward probability for left side during block.")
    right_length: int = Field(ge=0, description="Minimum number of trials in block.")
    left_length: int = Field(ge=0, description="Minimum number of trials in block.")


class BlockBasedTrialGeneratorSpec(BaseTrialGeneratorSpecModel):
    type: Literal["BlockBasedTrialGenerator"] = "BlockBasedTrialGenerator"

    quiescent_duration: Distribution = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1),
            truncation_parameters=TruncationParameters(min=0, max=1),
        ),
        description="Distribution describing the quiescence period before trial starts (in seconds). Each lick resets the timer.",
    )

    response_duration: float = Field(default=1.0, ge=0, description="Duration after go cue for animal response.")

    reward_consumption_duration: float = Field(
        default=3.0,
        ge=0,
        description="Duration of reward consumption before transition to ITI (in seconds).",
    )

    inter_trial_interval_duration: Distribution = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 2),
            truncation_parameters=TruncationParameters(min=1, max=8),
        ),
        description="Distribution describing the inter-trial interval (in seconds).",
    )

    block_len: Distribution = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 20),
            truncation_parameters=TruncationParameters(min=20, max=60),
        ),
        description="Distribution describing block length.",
    )

    min_block_reward: int = Field(default=1, ge=0, title="Minimal rewards in a block to switch")

    kernel_size: int = Field(default=2, description="Kernel to evaluate choice fraction.")
    reward_probability_parameters: RewardProbabilityParameters = Field(
        default=RewardProbabilityParameters(),
        description="Parameters defining the reward probability structure.",
        validate_default=True,
    )

    autowater_parameters: Optional[AutoWaterParameters] = Field(
        default=AutoWaterParameters(),
        validate_default=True,
        description="Auto water settings. If set, free water is delivered when the animal exceeds the ignored or unrewarded trial thresholds.",
    )

    is_baiting: bool = Field(default=False, description="Whether uncollected rewards carry over to the next trial.")

    def create_generator(self) -> "BlockBasedTrialGenerator":
        return BlockBasedTrialGenerator(self)


class BlockBasedTrialGenerator(ITrialGenerator, ABC):
    """Abstract trial generator for block-based dynamic foraging tasks.

    Manages block transitions, baiting logic, and trial generation. Subclasses
    must implement `_are_end_conditions_met` to define session termination logic.

    Attributes:
        spec: The specification used to configure this generator.
        is_right_choice_history: Record of whether each trial was a right choice.
            None indicates no choice was made (e.g. missed trial).
        reward_history: Record of whether each trial resulted in a reward.
        is_left_baited: Whether the left port currently has a baited reward.
        is_right_baited: Whether the right port currently has a baited reward.

    """

    def __init__(self, spec: BlockBasedTrialGeneratorSpec) -> None:
        """Initializes the generator and generates the first block.

        Args:
            spec: The BlockBasedTrialGenerator defining task parameters.
        """

        self.spec = spec
        self.is_right_choice_history: list[bool | None] = []
        self.reward_history: list[bool] = []
        self.is_left_baited: bool = False
        self.is_right_baited: bool = False
        self.block: Block

    def update(self, outcome: TrialOutcome | str):
        """Updates generator state from the previous trial outcome. Records choice and reward history and manages baiting state.
        Args:
            outcome: The TrialOutcome from the most recently completed trial.
        """
        logger.debug("Updating trial generator.")
        if isinstance(outcome, str):
            outcome = TrialOutcome.model_validate_json(outcome)

        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)

        if self.spec.is_baiting:
            if outcome.is_right_choice:
                logger.debug("Resesting right bait.")
                self.is_right_baited = False
            elif outcome.is_right_choice is False:
                logger.debug("Resesting left bait.")
                self.is_left_baited = False

    def next(self) -> Trial | None:
        """Generates the next trial in the session.

        Checks end conditions, samples timing parameters, and applies baiting
        logic if enabled. Returns None if the session should end.

        Returns:
            The next Trial, or None if end conditions are met.
        """
        logger.info("Generating next trial.")

        # check end conditions
        if self._are_end_conditions_met():
            logger.info("Trial generator end conditions met.")
            return

        # determine iti and quiescent period duration
        iti = draw_sample(self.spec.inter_trial_interval_duration)
        quiescent = draw_sample(self.spec.quiescent_duration)

        # determine baiting
        if self.spec.is_baiting:
            random_numbers = np.random.random(2)

            self.is_left_baited = self.block.p_left_reward > random_numbers[0] or self.is_left_baited
            logger.debug(f"Left baited: {self.is_left_baited}")

            self.is_right_baited = self.block.p_right_reward > random_numbers[1] or self.is_right_baited
            logger.debug(f"Right baited: {self.is_right_baited}")

        # determine autowater
        if self._are_autowater_conditions_met():
            is_right_autowater = True if self.block.p_right_reward > self.block.p_left_reward else False

        return Trial(
            p_reward_left=1 if (self.is_left_baited and self.spec.is_baiting) else self.block.p_left_reward,
            p_reward_right=1 if (self.is_right_baited and self.spec.is_baiting) else self.block.p_right_reward,
            reward_consumption_duration=self.spec.reward_consumption_duration,
            response_deadline_duration=self.spec.response_duration,
            quiescence_period_duration=quiescent,
            inter_trial_interval_duration=iti,
            is_auto_response_right=is_right_autowater,
        )

    def _are_autowater_conditions_met(self) -> bool:
        """Checks whether autowater should be given.

        Returns:
            True if autowater conditions are met, False otherwise.
        """

        if self.spec.autowater_parameters is None:  # autowater disabled
            return False

        min_ignore = self.spec.autowater_parameters.min_ignored_trials
        min_unreward = self.spec.autowater_parameters.min_unrewarded_trials

        is_ignored = [choice is None for choice in self.is_right_choice_history]
        if all(is_ignored[-min_ignore:]):
            return True

        is_unrewarded = [not reward for reward in self.reward_history]
        if all(is_unrewarded[-min_unreward:]):
            return True

        return False

    @abstractmethod
    def _are_end_conditions_met(self) -> bool:
        """Checks whether the session should end.

        Returns:
            True if end conditions are met and no further trials should be
            generated, False otherwise.
        """
        pass

    def _generate_next_block(*args, **kwargs) -> Block:
        """Abstract method. Subclasses must implement their own block switching logic.

        Returns:
            A new Block with sampled reward probabilities and length.
        """

        pass

    @abstractmethod
    def _is_block_switch_allowed(self) -> bool:
        """Determines whether all criteria are met to switch to the next block.

        Returns:
            True if all switch criteria are satisfied, False otherwise.
        """

        pass

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
        p_left_reward: Current probability of reward on left side
        p_right_reward: Current probability of reward on right side
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

        if self.spec.is_baiting:
            random_numbers = np.random.random(2)

            self.is_left_baited = self.block.p_left_reward > random_numbers[0] or self.is_left_baited
            logger.debug(f"Left baited: {self.is_left_baited}")

            self.is_right_baited = self.block.p_right_reward > random_numbers[1] or self.is_right_baited
            logger.debug(f"Right baited: {self.is_right_baited}")

        return Trial(
            p_reward_left=1 if (self.is_left_baited and self.spec.is_baiting) else self.block.p_left_reward,
            p_reward_right=1 if (self.is_right_baited and self.spec.is_baiting) else self.block.p_right_reward,
            reward_consumption_duration=self.spec.reward_consumption_duration,
            response_deadline_duration=self.spec.response_duration,
            quiescence_period_duration=quiescent,
            inter_trial_interval_duration=iti,
        )

    @abstractmethod
    def _are_end_conditions_met(self) -> bool:
        """Checks whether the session should end.

        Returns:
            True if end conditions are met and no further trials should be
            generated, False otherwise.
        """
        pass

    @abstractmethod
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

import logging
import random
from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import numpy as np
from aind_behavior_services.task.distributions import (
    Distribution,
    ExponentialDistribution,
    ExponentialDistributionParameters,
    TruncationParameters,
    UniformDistribution,
)
from aind_behavior_services.task.distributions_utils import draw_sample
from pydantic import BaseModel, Field

from ..trial_models import Trial
from ._base import BaseTrialGeneratorSpecModel, ITrialGenerator

logger = logging.getLogger(__name__)


class RewardProbabilityParameters(BaseModel):
    """Defines the reward probability structure for a dynamic foraging task.

    Reward probabilities are defined as pairs (p_left, p_right) normalized by
    base_reward_sum. Pairs are drawn from a family representing a difficulty level:

        Family 0:   [[8, 1], [6, 1], [3, 1], [1, 1]]
        Family 1:  [[8, 1], [1, 1]]
        Family 2:  [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]
        Family 3:  [[6, 1], [3, 1], [1, 1]]

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
    p_right_reward: float = Field(ge=0, le=1, description="Reward probability for right side during block.")
    p_left_reward: float = Field(ge=0, le=1, description="Reward probability for left side during block.")
    min_length: int = Field(ge=0, description="Minimum number of trials in block.")


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
        block_history: Record of all completed blocks.
        block: The currently active block.
        trials_in_block: Number of trials elapsed in the current block.
        is_left_baited: Whether the left port currently has a baited reward.
        is_right_baited: Whether the right port currently has a baited reward.
    """

    def __init__(self, spec: BlockBasedTrialGeneratorSpec) -> None:
        """Initializes the generator and generates the first block.

        Args:
            spec: The BlockBasedTrialGeneratorSpec defining task parameters.
        """

        self.spec = spec
        self.is_right_choice_history: list[bool | None] = []
        self.reward_history: list[bool] = []
        self.block_history: list[Block] = []
        self.block: Block = self._generate_next_block(
            reward_pairs=self.spec.reward_probability_parameters.reward_pairs,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_len=self.spec.block_len,
        )
        self.trials_in_block = 0
        self.is_left_baited: bool = False
        self.is_right_baited: bool = False

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

        p_reward_left = self.block.p_left_reward
        p_reward_right = self.block.p_right_reward

        if self.spec.is_baiting:
            random_numbers = np.random.random(2)

            is_left_baited = self.block.p_left_reward > random_numbers[0] or self.is_left_baited
            logger.debug(f"Left baited: {is_left_baited}")
            p_reward_left = 1 if is_left_baited else p_reward_left

            is_right_baited = self.block.p_right_reward > random_numbers[1] or self.is_right_baited
            logger.debug(f"Right baited: {is_left_baited}")
            p_reward_right = 1 if is_right_baited else p_reward_right

        return Trial(
            p_reward_left=p_reward_left,
            p_reward_right=p_reward_right,
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

    def _generate_next_block(
        self,
        reward_pairs: list[list[float, float]],
        base_reward_sum: float,
        block_len: Union[UniformDistribution, ExponentialDistribution],
        current_block: Optional[None] = None,
    ) -> Block:
        """Generates the next block, avoiding repeating the current block's side bias.

        Normalizes reward pairs by base_reward_sum, mirrors them to create a full
        pool, optionally excludes the current block's probabilities and high-reward
        side, then randomly samples the next block.

        Args:
            reward_pairs: List of (left, right) reward ratio pairs to draw from.
            base_reward_sum: Total reward probability to normalize each pair to.
            block_len: Distribution from which to sample the next block length.
            current_block: The currently active block, used to avoid repeating the
                same reward probabilities or high-reward side. Defaults to None.

        Returns:
            A new Block with sampled reward probabilities and length.
        """

        logger.info("Generating next block.")

        # determine candidate reward pairs
        reward_prob = np.array(reward_pairs, dtype=float)
        reward_prob /= reward_prob.sum(axis=1, keepdims=True)
        reward_prob *= float(base_reward_sum)
        logger.info(f"Candidate reward pairs normalized and scaled: {reward_prob.tolist()}")

        # create pool including all reward probabiliteis and mirrored pairs
        reward_prob_pool = np.vstack([reward_prob, np.fliplr(reward_prob)])

        if current_block:  # exclude previous block if history exists
            logger.info("Excluding previous block reward probability.")
            last_block_reward_prob = [current_block.p_right_reward, current_block.p_left_reward]

            # remove blocks identical to last block
            reward_prob_pool = reward_prob_pool[np.any(reward_prob_pool != last_block_reward_prob, axis=1)]
            logger.debug(f"Pool after removing identical to last block: {reward_prob_pool.tolist()}")

            # remove blocks with same high-reward side (if last block had a clear high side)
            if last_block_reward_prob[0] != last_block_reward_prob[1]:
                high_side_last = last_block_reward_prob[0] > last_block_reward_prob[1]
                high_side_pool = reward_prob_pool[:, 0] > reward_prob_pool[:, 1]
                reward_prob_pool = reward_prob_pool[high_side_pool != high_side_last]
                logger.debug(f"Pool after removing same high-reward side: {reward_prob_pool.tolist()}")

        # remove duplicates
        reward_prob_pool = np.unique(reward_prob_pool, axis=0)
        logger.debug(f"Final reward probability pool after removing duplicates: {reward_prob_pool.tolist()}")

        # randomly pick next block reward probability
        p_right_reward, p_left_reward = reward_prob_pool[random.choice(range(reward_prob_pool.shape[0]))]
        logger.info(f"Selected next block reward probabilities: right={p_right_reward}, left={p_left_reward}")

        # randomly pick block length
        next_block_len = round(draw_sample(block_len))
        logger.info(f"Selected next block length: {next_block_len}")

        return Block(
            p_right_reward=p_right_reward,
            p_left_reward=p_left_reward,
            min_length=next_block_len,
        )

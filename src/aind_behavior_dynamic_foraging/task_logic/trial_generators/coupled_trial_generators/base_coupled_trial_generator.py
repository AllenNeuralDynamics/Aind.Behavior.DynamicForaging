import logging
import random
from typing import Annotated, List, Literal, Optional

import numpy as np
from aind_behavior_services.task.distributions import Distribution
from aind_behavior_services.task.distributions_utils import draw_sample
from pydantic import BaseModel, Field

from ...trial_models import TrialOutcome
from ..block_based_trial_generator import Block, BlockBasedTrialGenerator, BlockBasedTrialGeneratorSpec

logger = logging.getLogger(__name__)

RewardPair = Annotated[
    List[float], Field(min_length=2, max_length=2, description="Reward ratio pairs ordered left, right.")
]


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
    reward_pairs: list[RewardPair] = Field(
        default=[[8, 1]],
        description="List of (left, right) reward ratio pairs to sample from during block transitions. ",
    )


class BaseCoupledTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
    type: Literal["BaseCoupledTrialGenerator"] = "BaseCoupledTrialGenerator"

    reward_probability_parameters: RewardProbabilityParameters = Field(
        default=RewardProbabilityParameters(),
        description="Parameters defining the reward probability structure.",
        validate_default=True,
    )

    def create_generator(self) -> "BaseCoupledTrialGenerator":
        return BaseCoupledTrialGenerator(self)


class BaseCoupledTrialGenerator(BlockBasedTrialGenerator):
    spec: BaseCoupledTrialGeneratorSpec

    def __init__(self, spec: BaseCoupledTrialGeneratorSpec) -> None:
        """Initializes the generator and generates the first block.

        Args:
            spec: The BaseCoupledTrialGeneratorSpec defining task parameters.
        """

        super().__init__(spec)

        self.block: Block = self._generate_next_block(
            reward_pairs=self.spec.reward_probability_parameters.reward_pairs,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_length=self.spec.block_length,
        )
        self.p_right_reward = self.block.p_right_reward
        self.p_left_reward = self.block.p_left_reward
        self.block_history = []
        self.trials_in_block = 0

    def update(self, outcome: TrialOutcome | str) -> None:
        """
        Records choice and reward history, manages baiting state, optionally extends
        the block on no response, and triggers a block switch if all switch criteria
        are satisfied.
        :param outcome: trial outcome of previous trial
        """
        super().update(outcome)

        self.trials_in_block += 1

        if self._is_block_switch_allowed():
            logger.info("Switching block.")
            self.trials_in_block = 0
            self.block = self._generate_next_block(
                reward_pairs=self.spec.reward_probability_parameters.reward_pairs,
                base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
                current_block=self.block,
                block_length=self.spec.block_length,
            )
            self.block_history.append(self.block)

    @staticmethod
    def _generate_next_block(
        reward_pairs: list[list[float, float]],
        base_reward_sum: float,
        block_length: Distribution,
        current_block: Optional[Block] = None,
    ) -> Block:
        """Generates the next block, avoiding repeating the current block's side bias.

        Normalizes reward pairs by base_reward_sum, mirrors them to create a full
        pool, optionally excludes the current block's probabilities and high-reward
        side, then randomly samples the next block.

        Args:
            reward_pairs: List of (left, right) reward ratio pairs to draw from.
            base_reward_sum: Total reward probability to normalize each pair to.
            block_length: Distribution from which to sample the next block length.
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
        logger.info("Candidate reward pairs normalized and scaled: %s" % reward_prob.tolist())

        # create pool including all reward probabiliteis and mirrored pairs
        reward_prob_pool = np.vstack([reward_prob, np.fliplr(reward_prob)])

        if current_block:  # exclude previous block if history exists
            logger.info("Excluding previous block reward probability.")
            last_block_reward_prob = [current_block.p_right_reward, current_block.p_left_reward]

            # remove blocks identical to last block
            reward_prob_pool = reward_prob_pool[np.any(reward_prob_pool != last_block_reward_prob, axis=1)]
            logger.debug("Pool after removing identical to last block: %s" % reward_prob_pool.tolist())

            # remove blocks with same high-reward side (if last block had a clear high side)
            if last_block_reward_prob[0] != last_block_reward_prob[1]:
                high_side_last = last_block_reward_prob[0] > last_block_reward_prob[1]
                high_side_pool = reward_prob_pool[:, 0] > reward_prob_pool[:, 1]
                reward_prob_pool = reward_prob_pool[high_side_pool != high_side_last]
                logger.debug("Pool after removing same high-reward side: %s" % reward_prob_pool.tolist())

        # remove duplicates
        reward_prob_pool = np.unique(reward_prob_pool, axis=0)
        logger.debug("Final reward probability pool after removing duplicates: %s" % reward_prob_pool.tolist())

        # randomly pick next block reward probability
        p_right_reward, p_left_reward = reward_prob_pool[random.choice(range(reward_prob_pool.shape[0]))]
        logger.info("Selected next block reward probabilities: right=%s, left=%s" % (p_right_reward, p_left_reward))

        # randomly pick block length
        next_block_length = round(draw_sample(block_length))
        logger.info("Selected next block length: %s" % next_block_length)

        return Block(
            p_right_reward=p_right_reward,
            p_left_reward=p_left_reward,
            right_length=next_block_length,
            left_length=next_block_length,
        )

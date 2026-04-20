import logging
from typing import Literal, Optional
import numpy as np
import random
from pydantic import BaseModel, Field

from aind_behavior_services.task.distributions import Distribution, Scalar
from aind_behavior_services.task.distributions_utils import draw_sample

from ..trial_models import TrialOutcome
from .block_based_trial_generator import (
    BlockBasedTrialGenerator,
    BlockBasedTrialGeneratorSpec,
    Block
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

class CoupledWarmupTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
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
    reward_probability_parameters: RewardProbabilityParameters = Field(
        default=RewardProbabilityParameters(),
        description="Parameters defining the reward probability structure.",
        validate_default=True,
    )

    def create_generator(self) -> "CoupledWarmupTrialGenerator":
        return CoupledWarmupTrialGenerator(self)


class CoupledWarmupTrialGenerator(BlockBasedTrialGenerator):
    spec: CoupledWarmupTrialGeneratorSpec

    def __init__(self, spec: CoupledWarmupTrialGeneratorSpec) -> None:
        """Initializes the generator and generates the first block.

        Args:
            spec: The CoupledWarmupTrialGeneratorSpec defining task parameters.
        """

        super().__init__(spec)

        self.block: Block = self._generate_next_block(
            reward_pairs=self.spec.reward_probability_parameters.reward_pairs,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_len=self.spec.block_len,
        )

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

    def update(self, outcome: TrialOutcome | str) -> None:
        """
        Records choice and reward history, manages baiting state, optionally extends
        the block on no response, and triggers a block switch if all switch criteria
        are satisfied.
        :param outcome: trial outcome of previous trial
        """

        logger.debug(f"Updating trial generator.")

        if isinstance(outcome, str):
            outcome = TrialOutcome.model_validate_json(outcome)

        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)
        self.trials_in_block += 1

        if self.spec.is_baiting:
            if outcome.is_right_choice:
                logger.debug("Resesting right bait.")
                self.is_right_baited = False
            elif outcome.is_right_choice is False:
                logger.debug("Resesting left bait.")
                self.is_left_baited = False

        switch_block = self._is_block_switch_allowed()

        if switch_block:
            logger.info("Switching block.")
            self.trials_in_block = 0
            self.block = self._generate_next_block(
                reward_pairs=self.spec.reward_probability_parameters.reward_pairs,
                base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
                current_block=self.block,
                block_len=self.spec.block_len,
            )
            self.block_history.append(self.block)

    def _is_block_switch_allowed(self) -> True:

        """
        Warmup switches block every update

        Returns:
            True
        """

        return True

    @staticmethod
    def _generate_next_block(
        reward_pairs: list[list[float, float]],
        base_reward_sum: float,
        block_len: Distribution,
        current_block: Optional[Block] = None,
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
import logging
import random
from typing import Literal, Optional, Union

import numpy as np
from aind_behavior_services.task.distributions import (
    ExponentialDistribution,
    ExponentialDistributionParameters,
    TruncationParameters,
    UniformDistribution,
)
from aind_behavior_services.task.distributions_utils import draw_sample
from pydantic import BaseModel, Field

from ..trial_models import Trial, TrialOutcome
from ._base import BaseTrialGeneratorSpecModel, ITrialGenerator

logger = logging.getLogger(__name__)


class TrialGenerationEndConditions(BaseModel):
    min_trial: int = Field(default=50, description="Minimum trials in generator.")
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
    evaluation_window: int = Field(default=20, description="Number of most recent trials to evaluate the end criteria.")


class RewardProbabilityParameters(BaseModel):
    base_reward_sum: float = 1
    family: int = 3
    pairs_n: int = 1


class Block(BaseModel):
    right_reward_prob: float
    left_reward_prob: float
    min_length: int


class WarmupTrialGeneratorSpec(BaseTrialGeneratorSpecModel):
    type: Literal["WarmupTrialGenerator"] = "WarmupTrialGenerator"

    quiescent_duration_distribution: Union[UniformDistribution, ExponentialDistribution] = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1),
            truncation_parameters=TruncationParameters(min=0, max=1),
        ),
        description="Duration of the quiescence period before trial starts (in seconds). Each lick resets the timer.",
    )
    response_duration: float = Field(default=1.0, description="Duration after go cue for animal response.")

    reward_consumption_duration: float = Field(
        default=3.0,
        description="Duration of reward consumption before transition to ITI (in seconds).",
    )
    inter_trial_interval_duration_distribution: Union[UniformDistribution, ExponentialDistribution] = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 2),
            truncation_parameters=TruncationParameters(min=1, max=8),
        ),
        description="Duration of the inter-trial interval (in seconds).",
    )

    block_len_distribution: ExponentialDistribution = ExponentialDistribution(
        distribution_parameters=ExponentialDistributionParameters(rate=1),
        truncation_parameters=TruncationParameters(min=1, max=2),
    )

    trial_generation_end_parameters: TrialGenerationEndConditions = Field(
        default=TrialGenerationEndConditions(), description="Conditions to end trial generation."
    )
    min_block_reward: int = 1
    reward_probability_parameters: RewardProbabilityParameters = Field(default=RewardProbabilityParameters())
    reward_family: list = [
        [[8, 1], [6, 1], [3, 1], [1, 1]],
        [[8, 1], [1, 1]],
        [
            [1, 0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
        ],
        [[6, 1], [3, 1], [1, 1]],
    ]

    baiting: bool = True

    def create_generator(self) -> "WarmupTrialGenerator":
        return WarmupTrialGenerator(self)


class WarmupTrialGenerator(ITrialGenerator):
    def __init__(self, spec: WarmupTrialGeneratorSpec) -> None:
        """"""
        self.spec = spec
        self.is_right_choice_history: list[bool | None] = []
        self.reward_history: list[bool] = []
        self.block_history: list[Block] = []
        self.block: Block = self.generate_next_block(
            reward_families=self.spec.reward_family,
            reward_family_index=self.spec.reward_probability_parameters.family,
            reward_pairs_n=self.spec.reward_probability_parameters.pairs_n,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_len_distribution=self.spec.block_len_distribution,
        )
        self.trials_in_block = 0

        self.is_left_baited: bool = False
        self.is_right_baited: bool = False

    def next(self) -> Trial | None:
        """
        Generate next trial

        """
        logger.info("Generating next trial.")

        # check end conditions
        if self.are_end_conditions_met(self.spec.trial_generation_end_parameters, self.is_right_choice_history):
            logger.info("Trial generator end conditions met.")
            return

        # determine iti and quiescent period duration
        iti = draw_sample(self.spec.inter_trial_interval_duration_distribution)
        quiescent = draw_sample(self.spec.quiescent_duration_distribution)

        p_reward_left = self.block.left_reward_prob
        p_reward_right = self.block.right_reward_prob

        if self.spec.baiting:
            random_numbers = np.random.random(2)

            is_left_baited = self.block.left_reward_prob > random_numbers[0] or self.is_left_baited
            logger.debug(f"Left baited: {is_left_baited}")
            p_reward_left = 1 if is_left_baited else p_reward_left

            is_right_baited = self.block.right_reward_prob > random_numbers[1] or self.is_right_baited
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

    @staticmethod
    def are_end_conditions_met(end_conditions: TrialGenerationEndConditions, choice_history: list[bool | None]) -> bool:
        """

        Check if end conditions are met to stop session

        :param end_conditons: conditions to be met for trial generation to stop

        """

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
        Check if block should switch, generate next block if necessary, and  generate next trial

        :param outcome: trial outcome of previous trial
        """

        logger.info(f"Updating Warmup trial generator with trial outcome of {outcome}")

        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)
        self.trials_in_block += 1

        if self.spec.baiting:
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
            block_len_distribution=self.spec.block_len_distribution,
        )
        self.block_history.append(self.block)

    def generate_next_block(
        self,
        reward_families: list,
        reward_family_index: int,
        reward_pairs_n: int,
        base_reward_sum: float,
        block_len_distribution: Union[UniformDistribution, ExponentialDistribution],
        current_block: Optional[None] = None,
    ) -> Block:
        """
        Generate the next block for a warmup task.

        :param reward_families: Description
        :param reward_family_index: Description
        :param reward_pairs_n: Description
        :param base_reward_sum: Description
        :param current_block: Description
        :param block_len_distribution: Description
        """

        logger.info("Generating next block.")

        # determine candidate reward pairs
        reward_pairs = reward_families[reward_family_index][:reward_pairs_n]
        reward_prob = np.array(reward_pairs, dtype=float)
        reward_prob /= reward_prob.sum(axis=1, keepdims=True)
        reward_prob *= float(base_reward_sum)
        logger.info(f"Candidate reward pairs normalized and scaled: {reward_prob.tolist()}")

        # create pool including all reward probabiliteis and mirrored pairs
        reward_prob_pool = np.vstack([reward_prob, np.fliplr(reward_prob)])

        if current_block:  # exclude previous block if history exists
            logger.info("Excluding previous block reward probability.")
            last_block_reward_prob = [current_block.right_reward_prob, current_block.left_reward_prob]

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
        right_reward_prob, left_reward_prob = reward_prob_pool[random.choice(range(reward_prob_pool.shape[0]))]
        logger.info(f"Selected next block reward probabilities: right={right_reward_prob}, left={left_reward_prob}")

        # randomly pick block length
        next_block_len = round(draw_sample(block_len_distribution))
        logger.info(f"Selected next block length: {next_block_len}")

        return Block(
            right_reward_prob=right_reward_prob,
            left_reward_prob=left_reward_prob,
            min_length=next_block_len,
        )

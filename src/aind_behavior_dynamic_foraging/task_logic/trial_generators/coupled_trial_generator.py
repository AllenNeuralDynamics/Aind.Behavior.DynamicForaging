import logging
import random
from datetime import datetime, timedelta
from typing import Literal, Optional, Union

import numpy as np
from aind_behavior_services.task.distributions import (
    DistributionFamily,
    ExponentialDistribution,
    ExponentialDistributionParameters,
    TruncationParameters,
    UniformDistribution,
)
from pydantic import BaseModel, Field

from ..trial_models import Trial, TrialOutcome
from ._base import ITrialGenerator, _BaseTrialGeneratorSpecModel

BlockBehaviorEvaluationMode = Literal[
    "end",  # behavior stable at end of block to allow switching
    "anytime",  # behavior stable anytime in block to allow switching
]


class TrialGenerationEndConditions(BaseModel):
    ignore_win: int = Field(default=30, title="Window of trials to check ignored responses")
    ignore_ratio_threshold: float = Field(
        default=0.8, title="Threshold for acceptable ignored trials within window.", ge=0, le=1
    )
    max_trial: int = Field(default=1000, title="Maximal number of trials")
    max_time: timedelta = Field(timedelta(minutes=75), title="Maximal session time (min)")
    min_time: timedelta = Field(default=timedelta(minutes=30), title="Minimum session time (min)")


class BehaviorStabilityParameters(BaseModel):
    behavior_evaluation_mode: BlockBehaviorEvaluationMode = Field(
        default="end", title="Mode to evaluate behavior stability.", validate_default=True
    )
    behavior_stability_fraction: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Fraction scaling reward-probability difference for behavior.",
    )
    min_consecutive_stable_trials: int = Field(
        default=5,
        description="Minimum number of consecutive trials satisfying the behavioral stability fraction.",
    )


class RewardProbabilityParameters(BaseModel):
    base_reward_sum: float = Field(default=0.8, title="Sum of p_reward")
    family: int = Field(default=1, title="Reward family")
    pairs_n: int = Field(default=1, title="Number of pairs")


class Block(BaseModel):
    right_reward_prob: float
    left_reward_prob: float
    min_length: int


class CoupledTrialGeneratorSpec(_BaseTrialGeneratorSpecModel):
    type: Literal["CoupledTrialGenerator"] = "CoupledTrialGenerator"

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

    block_len_distribution: Union[UniformDistribution, ExponentialDistribution] = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 20),
            truncation_parameters=TruncationParameters(min=20, max=60),
        )
    )

    trial_generation_end_parameters: TrialGenerationEndConditions = Field(
        default=TrialGenerationEndConditions(), description="Conditions to end trial generation."
    )
    min_block_reward: int = Field(default=1, title="Minimal rewards in a block to switch")
    behavior_stability_parameters: Optional[BehaviorStabilityParameters] = Field(
        default=BehaviorStabilityParameters(),
        description="Parameters describing behavior stability required to switch blocks.",
    )
    extend_block_on_no_response: bool = Field(
        default=True,
        description="Add one trial to the min block length.",
    )
    kernel_size: int = Field(default=2, description="Kernel to evaluate choice fraction.")
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

    def create_generator(self) -> "CoupledTrialGenerator":
        return CoupledTrialGenerator(self)


class CoupledTrialGenerator(ITrialGenerator):
    def __init__(self, spec: CoupledTrialGeneratorSpec) -> None:
        """"""
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
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
        self.start_time = datetime.now()

    def next(self) -> Trial | None:
        """
        Generate next trial

        """
        self.logger.info("Generating next trial.")

        # check end conditions
        if not self.are_end_conditions_met(
            self.spec.trial_generation_end_parameters, self.is_right_choice_history, self.start_time
        ):
            self.logger.info("Trial generator end conditions met.")
            return

        # determine iti and quiescent period duration
        iti = self.evaluate_distribution(self.spec.inter_trial_interval_duration_distribution)
        quiescent = self.evaluate_distribution(self.spec.quiescent_duration_distribution)

        # iterate trials in block
        self.trials_in_block += 1

        return Trial(
            p_reward_left=self.block.left_reward_prob,
            p_reward_right=self.block.right_reward_prob,
            reward_consumption_duration=self.spec.reward_consumption_duration,
            response_deadline_duration=self.spec.response_duration,
            quiescence_period_duration=quiescent,
            inter_trial_interval_duration=iti,
        )

    @staticmethod
    def are_end_conditions_met(
        end_conditions: TrialGenerationEndConditions, choice_history: list[bool | None], start_time: datetime
    ) -> bool:
        """

        Check if end conditions are met to stop session

        :param end_conditons: conditions to be met for trial generation to stop

        """
        time_elapsed = datetime.now() - start_time
        if time_elapsed < end_conditions.min_time:
            return True

        if end_conditions.max_trial < len(choice_history):
            return False

        if end_conditions.max_time < time_elapsed:
            return False

        frac = end_conditions.ignore_ratio_threshold
        win = end_conditions.ignore_win
        if choice_history[-win:].count(None) > frac * win:
            return False

        return True

    @staticmethod
    def evaluate_distribution(
        distribution: Union[UniformDistribution, ExponentialDistribution],
    ) -> float:
        if distribution.family == DistributionFamily.EXPONENTIAL:
            return (
                np.random.exponential(1 / distribution.distribution_parameters.rate)
                + distribution.truncation_parameters.min
            )
        elif distribution.family == DistributionFamily.UNIFORM:
            return random.uniform(
                distribution.distribution_parameters.min,
                distribution.distribution_parameters.max,
            )

        else:
            raise ValueError(f"Distribution {distribution.family} not recognized.")

    def update(self, outcome: TrialOutcome) -> None:
        """
        Check if block should switch, generate next block if necessary, and  generate next trial

        :param outcome: trial outcome of previous trial
        """

        self.logger.info(f"Updating coupled trial generator with trial outcome of {outcome}")

        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)
        self.trials_in_block += 1

        if self.spec.extend_block_on_no_response and outcome.is_right_choice is None:
            self.logger.info("Extending minimum block length due to ignored trial.")
            self.block.min_length += 1

        switch_block = self.is_block_switch_allowed(
            trials_in_block=self.trials_in_block,
            min_block_reward=self.spec.min_block_reward,
            block_left_rewards=self.reward_history.count(False),
            block_right_rewards=self.reward_history.count(True),
            choice_history=self.is_right_choice_history,
            right_reward_prob=self.block.right_reward_prob,
            left_reward_prob=self.block.left_reward_prob,
            beh_stability_params=self.spec.behavior_stability_parameters,
            block_length=self.block.min_length,
            kernel_size=self.spec.kernel_size,
        )

        if switch_block:
            self.logger.info("Switching block.")
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

    def is_behavior_stable(
        self,
        choice_history: list,
        right_reward_prob: float,
        left_reward_prob: float,
        beh_stability_params: BehaviorStabilityParameters,
        trials_in_block: int,
        kernel_size: int = 2,
    ) -> Optional[bool]:
        """
        This function replaces _check_advanced_block_switch. Checks if behavior within block
        allows for switching

        choice_history: 1D array with 0: left, 1: right and None: ignored entries.
        right_reward_prob: reward probability for right side
        left_reward_prob: reward probability for left side
        beh_stability_params: Parameters to evaluate behavior
        trials_in_block: number of trials in current block. In couple trials, both sides have same block length so block length is int.
        kernel_size: kernel to evaluate choice fraction

        """

        self.logger.info("Evaluating block behavior.")

        # do not prohibit block transition if does not rely on behavior or not enough trials to evaluate or reward probs are the same.
        if not beh_stability_params or left_reward_prob == right_reward_prob or len(choice_history) < kernel_size:
            self.logger.debug(
                "Behavior stability evaluation skipped: "
                f"parameters_missing={not bool(beh_stability_params)}, "
                f"rewards_equal={left_reward_prob == right_reward_prob}, "
                f"trials_available={len(choice_history)} < kernel_size({kernel_size})"
            )
            return True

        # compute fraction of right choices with running average using a sliding window
        block_history = choice_history[-(trials_in_block + kernel_size - 1) :]
        block_choice_frac = self.compute_choice_fraction(kernel_size, block_history)
        self.logger.debug(f"Choice fraction of block is {block_choice_frac}.")

        # margin based on right and left probabilities and scaled by switch threshold. Window for evaluating behavior
        delta = abs((left_reward_prob - right_reward_prob) * float(beh_stability_params.behavior_stability_fraction))
        threshold = (
            [0, left_reward_prob - delta] if left_reward_prob > right_reward_prob else [left_reward_prob + delta, 1]
        )
        self.logger.debug(f"Behavior stability threshold applied: {threshold}")

        # block_choice_fractions above threshold
        points_above_threshold = np.logical_and(
            block_choice_frac >= threshold[0],
            block_choice_frac <= threshold[1],
        )

        # evaluate stability based on mode
        min_stable = beh_stability_params.min_consecutive_stable_trials
        mode = beh_stability_params.behavior_evaluation_mode
        if mode == "end":
            # requires consecutive trials at end of trial
            self.logger.info(f"Evaluating last {min_stable} trials for end-of-block stability.")
            if len(points_above_threshold) < min_stable:
                self.logger.info("Not enough trials to evaluate stability at block end.")
                return False
            stable = np.all(points_above_threshold[-min_stable:])
            self.logger.info(f"Behavior stable at block end: {stable}")
            return stable

        elif mode == "anytime":
            # allows consecutive trials any time in the behavior
            self.logger.info(f"Evaluating block for stability anytime over {min_stable} consecutive trials.")
            run_len = 0
            for i, v in enumerate(points_above_threshold):
                if v:
                    run_len += 1
                else:
                    run_len = 0
                if run_len >= min_stable:
                    self.logger.info(f"Behavior stable at trial index {i}.")
                    return True
            self.logger.info("Behavior not stable in block anytime evaluation.")
            return False

        else:
            raise ValueError(f"Behavior evaluation mode {mode} not recognized.")

    @staticmethod
    def compute_choice_fraction(kernel_size: int, choice_history: list[int | None]):
        """
        Compute fraction of right choices with running average using a sliding window

        :param kernel_size: kernel to evaluate choice fraction
        :param choice_history: 1D array with 0: left, 1: right and None: ignored entries.
        """

        n_windows = len(choice_history) - kernel_size + 1
        choice_fraction = np.empty(n_windows, dtype=float)  # create empty array to store running averages
        for i in range(n_windows):
            window = np.array(choice_history[i : i + kernel_size], dtype=float)
            choice_fraction[i] = np.nanmean(window)
        return choice_fraction

    def is_block_switch_allowed(
        self,
        trials_in_block: int,
        min_block_reward: int,
        block_left_rewards: int,
        block_right_rewards: int,
        choice_history: list,
        right_reward_prob: float,
        left_reward_prob: float,
        beh_stability_params: BehaviorStabilityParameters,
        block_length: int,
        kernel_size: int = 2,
    ) -> bool:
        """
        trials_in_block: number of trials in block
        min_block_reward: minimum reward to allow switching
        block_left_rewards: number of left rewarded trials in current block
        block_right_rewards: number of left rewarded trials in current block
        choice_history: 2D array (rows = sides, columns = trials) with 0: left, 1: right and 2: ignored entries.
        right_reward_prob: reward probability for right side
        left_reward_prob: reward probability for left side
        beh_stability_params: parameters to evaluate behavior
        block_length: planned number of trials in current block. In couple trials, both sides have same block length so block length is int.
        kernel_size: kernel to evaluate choice fraction
        """

        self.logger.info("Evaluating block switch.")

        # has planned block length been reached?
        block_length_ok = trials_in_block >= block_length
        self.logger.debug(f"Planned block length reached: {block_length_ok}")

        # is behavior qualified to switch?
        behavior_ok = self.is_behavior_stable(
            choice_history,
            right_reward_prob,
            left_reward_prob,
            beh_stability_params,
            trials_in_block,
            kernel_size,
        )
        self.logger.debug(f"Behavior meets stability criteria: {behavior_ok}")

        # has reward criteria been met?
        reward_ok = block_left_rewards + block_right_rewards >= min_block_reward
        self.logger.debug(f"Reward criterion satisfied: {reward_ok}")

        # conditions to switch:
        #   - planned block length reached
        #   - minimum reward requirement is reached
        #   - behavior is stable

        return block_length_ok and reward_ok and behavior_ok

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
        Generate the next block for a coupled task.

        :param reward_families: Description
        :param reward_family_index: Description
        :param reward_pairs_n: Description
        :param base_reward_sum: Description
        :param current_block: Description
        :param block_len_distribution: Description
        """

        self.logger.info("Generating next block.")

        # determine candidate reward pairs
        reward_pairs = reward_families[reward_family_index][:reward_pairs_n]
        reward_prob = np.array(reward_pairs, dtype=float)
        reward_prob /= reward_prob.sum(axis=1, keepdims=True)
        reward_prob *= float(base_reward_sum)
        self.logger.info(f"Candidate reward pairs normalized and scaled: {reward_prob.tolist()}")

        # create pool including all reward probabiliteis and mirrored pairs
        reward_prob_pool = np.vstack([reward_prob, np.fliplr(reward_prob)])

        if current_block:  # exclude previous block if history exists
            self.logger.info("Excluding previous block reward probability.")
            last_block_reward_prob = [current_block.right_reward_prob, current_block.left_reward_prob]

            # remove blocks identical to last block
            reward_prob_pool = reward_prob_pool[np.any(reward_prob_pool != last_block_reward_prob, axis=1)]
            self.logger.debug(f"Pool after removing identical to last block: {reward_prob_pool.tolist()}")

            # remove blocks with same high-reward side (if last block had a clear high side)
            if last_block_reward_prob[0] != last_block_reward_prob[1]:
                high_side_last = last_block_reward_prob[0] > last_block_reward_prob[1]
                high_side_pool = reward_prob_pool[:, 0] > reward_prob_pool[:, 1]
                reward_prob_pool = reward_prob_pool[high_side_pool != high_side_last]
                self.logger.debug(f"Pool after removing same high-reward side: {reward_prob_pool.tolist()}")

        # remove duplicates
        reward_prob_pool = np.unique(reward_prob_pool, axis=0)
        self.logger.debug(f"Final reward probability pool after removing duplicates: {reward_prob_pool.tolist()}")

        # randomly pick next block reward probability
        right_reward_prob, left_reward_prob = reward_prob_pool[random.choice(range(reward_prob_pool.shape[0]))]
        self.logger.info(
            f"Selected next block reward probabilities: right={right_reward_prob}, left={left_reward_prob}"
        )

        # randomly pick block length
        next_block_len = round(self.evaluate_distribution(block_len_distribution))
        self.logger.info(f"Selected next block length: {next_block_len}")

        return Block(
            right_reward_prob=right_reward_prob,
            left_reward_prob=left_reward_prob,
            min_length=next_block_len,
        )

import random
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

AutoWaterModes = Literal["Natural", "Both", "High pro"]
BlockBehaviorEvaluationMode = Literal[
    "ignore",  # do not take behavior into account when switching blocks
    "end",  # behavior must be stable at end of block to allow switching
    "anytime",
]  # behavior can be stable anytime in block to allow switching


class RewardProbability(BaseModel):
    base_reward_sum: float = Field(default=0.8, title="Sum of p_reward")
    family: int = Field(default=1, title="Reward family")
    pairs_n: int = Field(default=1, title="Number of pairs")


class AutoWater(BaseModel):
    auto_water_type: AutoWaterModes = Field(default="Natural", title="Auto water mode")
    multiplier: float = Field(default=0.8, title="Multiplier for auto reward")
    unrewarded: int = Field(default=200, title="Number of unrewarded trials before auto water")
    ignored: int = Field(default=100, title="Number of ignored trials before auto water")


class Warmup(BaseModel):
    min_trial: int = Field(default=50, title="Warmup finish criteria: minimal trials")
    max_choice_ratio_bias: float = Field(
        default=0.1, title="Warmup finish criteria: maximal choice ratio bias from 0.5"
    )
    min_finish_ratio: float = Field(default=0.8, title="Warmup finish criteria: minimal finish ratio")
    windowsize: int = Field(
        default=20,
        title="Warmup finish criteria: window size to compute the bias and ratio",
    )


class Block(BaseModel):
    right_reward_prob: float
    left_reward_prob: float
    min_length: int


class CoupledTrialGeneratorSpec(_BaseTrialGeneratorSpecModel):
    type: Literal["CoupledTrialGenerator"] = "CoupledTrialGenerator"

    iti: Union[UniformDistribution, ExponentialDistribution] = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 2),
            truncation_parameters=TruncationParameters(min=1, max=8),
        )
    )
    quiescent_period: Union[UniformDistribution, ExponentialDistribution] = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1),
            truncation_parameters=TruncationParameters(min=0, max=1),
        )
    )

    response_time: float = Field(default=1.0, title="Response time")
    reward_consume_time: float = Field(
        default=3.0,
        title="Reward consume time",
        description="Time of the no-lick period before trial end",
    )
    block_parameters: Union[UniformDistribution, ExponentialDistribution] = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 20),
            truncation_parameters=TruncationParameters(min=20, max=60),
        )
    )

    min_reward: int = Field(default=1, title="Minimal rewards in a block to switch")
    auto_water: Optional[AutoWater] = Field(default=None, description="Parameters describing auto water.")
    behavior_evaluation_mode: BlockBehaviorEvaluationMode = Field(default="now", title="Auto block mode")
    switch_thr: float = Field(default=0.5, title="Switch threshold for auto block")
    points_in_a_row: int = Field(default=5, title="Points in a row for auto block")
    warmup: Optional[Warmup] = Field(default=None, description="Parameters describing warmup.")
    no_response_trial_addition: bool = Field(
        default=True,
        description="Add one trial to the block length on both lickspouts.",
    )
    kernel_size: int
    reward_probability_specs: RewardProbability = Field(default=RewardProbability())
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

        self.spec = spec
        self.is_right_choice_history: list[bool | None] = []
        self.reward_history: list[bool] = []
        self.block_history: list[Block] = []
        self.block: Block = Block()
        self.trials_in_block = 0

    def next(self) -> Trial | None:
        """
        generate next trial

        :param self: Description
        :return: Description
        :rtype: Trial | None
        """

        iti = self.evaluate_distribution(self.spec.iti)
        quiescent = self.evaluate_distribution(self.spec.quiescent_period)

        self.trials_in_block += 1

        return Trial(
            p_reward_left=self.block.left_reward_prob,
            p_reward_right=self.block.right_reward_prob,
            reward_consumption_duration=self.spec.reward_consume_time,
            response_deadline_duration=self.spec.response_time,
            quiescence_period_duration=quiescent,
            inter_trial_interval_duration=iti,
        )

    @staticmethod
    def evaluate_distribution(
        distribution: Union[UniformDistribution, ExponentialDistribution],
    ) -> Union[UniformDistribution, ExponentialDistribution]:
        if distribution.family == DistributionFamily.EXPONENTIAL:
            return (
                np.random.exponential(1 / distribution.distribution_parameters.rate, 1)
                + distribution.truncation_parameters.min
            )
        elif distribution.family == DistributionFamily.UNIFORM:
            return random.uniform(
                distribution.distribution_parameters.min,
                distribution.distribution_parameters.max,
            )

        else:
            raise ValueError(f"Distibution {distribution.family} not recognized.")

    def update(self, outcome: TrialOutcome) -> None:
        """
        Check if block should switch, generate next block if necessary, and  generate next trial

        :param self: Description
        :param outcome: Description
        :type outcome: TrialOutcome
        """

        self.is_right_choice_history.append[outcome.is_right_choice]
        self.reward_history.append[outcome.is_rewarded]
        self.trials_in_block += 1

        switch_block = self.switch_block(
            trials_in_block=self.trials_in_block,
            min_block_reward=self.spec.min_reward,
            block_left_rewards=self.reward_history.count(False),
            block_right_rewards=self.reward_history.count(True),
            choice_history=self.is_right_choice_history,
            right_reward_prob=self.block.right_reward_prob,
            left_reward_prob=self.block.left_reward_prob,
            beh_eval_mode=self.spec.behavior_evaluation_mode,
            block_length=self.block.min_length,
            points_in_a_row=self.spec.points_in_a_row,
            switch_thr=self.spec.switch_thr,
            kernel_size=self.spec.kernel_size,
        )

        if switch_block:
            self.trials_in_block = 0
            self.block = self.generate_block(
                reward_families=self.spec.reward_family,
                reward_family_index=self.spec.reward_probability_specs.family,
                reward_pairs_n=self.spec.reward_probability_specs.pairs_n,
                base_reward_sum=self.spec.reward_probability_specs.base_reward_sum,
                block_history=self.block_history,
                block_distribution=self.spec.block_parameters,
            )
            self.block_history.append(self.block)

    def is_behavior_stable(
        self,
        choice_history: np.ndarray,
        right_reward_prob: float,
        left_reward_prob: float,
        beh_eval_mode: BlockBehaviorEvaluationMode,
        trials_in_block: int,
        points_in_a_row: int = 3,
        switch_thr: float = 0.8,
        kernel_size: int = 2,
    ) -> Optional[bool]:
        """
        This function replaces _check_advanced_block_switch. Checks if behavior within block
        allows for switching

        choice_history: 1D array with 0: left, 1: right and None: ignored entries.
        right_reward_prob: reward probability for right side
        left_reward_prob: reward probability for left side
        beh_eval_mode: mode to evaluate behavior
        trials_in_block: number of trials in current block. In couple trials, both sides have same block length so block length is int.
        points_in_a_row: number of consecutive trials above threshold required
        switch_thr: fraction threshold to define stable behavior
        kernel_size: kernal to evaluate choice fraction

        """

        # do not prohibit block transition if does not rely on behavior or not enough trials to evaluate or reward probs are the same.
        if beh_eval_mode == "ignore" or left_reward_prob == right_reward_prob or len(choice_history) < kernel_size:
            return True

        # compute fraction of right choices with running average using a sliding window
        block_history = choice_history[-(trials_in_block + kernel_size - 1) :]
        block_choice_frac = self.compute_choice_fraction(kernel_size, block_history)

        # margin based on right and left probabilities and scaled by switch threshold. Window for evaluating behavior
        delta = abs((left_reward_prob - right_reward_prob) * float(switch_thr))
        threshold = (
            [0, left_reward_prob - delta] if left_reward_prob > right_reward_prob else [left_reward_prob + delta, 1]
        )

        # block_choice_fractions above threshold
        points_above_threshold = np.logical_and(
            block_choice_frac >= threshold[0],
            block_choice_frac <= threshold[1],
        )

        # check consecutive pts above threshold
        if points_in_a_row <= 0:
            return True

        if beh_eval_mode == "end":
            # requires consecutive trials ending on the last trial
            # check if the current trial occurs at the end of a long enough consecutive run above threshold
            if len(points_above_threshold) < points_in_a_row:
                return False
            return np.all(points_above_threshold[-points_in_a_row:])

        elif beh_eval_mode == "anytime":
            # allows consecutive trials any time in the behavior
            run_len = 0
            for v in points_above_threshold:
                if v:
                    run_len += 1
                else:
                    if run_len >= points_in_a_row:
                        return True
                    else:
                        run_len = 0
            return run_len >= points_in_a_row

        else:
            raise ValueError(f"Behavior evaluation mode {beh_eval_mode} not recognized.")

    def compute_choice_fraction(self, kernel_size: int, choice_history: list[int | None]):
        """
        Compute fraction of right choices with running average using a sliding window

        :param kernel_size: kernal to evaluate choice fraction
        :param choice_history: 1D array with 0: left, 1: right and None: ignored entries.
        """

        n_windows = len(choice_history) - kernel_size + 1
        choice_fraction = np.empty(n_windows, dtype=float)  # create empty array to store running averages
        for i in range(n_windows):
            window = choice_history[i : i + kernel_size].astype(float)
            choice_fraction[i] = np.nanmean(window)
        return choice_fraction

    def switch_block(
        self,
        trials_in_block: int,
        min_block_reward: int,
        block_left_rewards: int,
        block_right_rewards: int,
        choice_history: np.ndarray,
        right_reward_prob: float,
        left_reward_prob: float,
        beh_eval_mode: BlockBehaviorEvaluationMode,
        block_length: int,
        points_in_a_row: int = 3,
        switch_thr: float = 0.8,
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
        beh_eval_mode: mode to evaluate behavior
        block_length: planned number of trials in current block. In couple trials, both sides have same block length so block length is int.
        points_in_a_row: number of consecutive trials above threshold required
        switch_thr: fraction threshold to define stable behavior
        kernel_size: kernal to evaluate choice fraction
        """

        # has planned block length been reached?
        block_length_ok = trials_in_block >= block_length

        # is behavior qualified to switch?
        behavior_ok = self.is_behavior_stable(
            choice_history,
            right_reward_prob,
            left_reward_prob,
            beh_eval_mode,
            trials_in_block,
            points_in_a_row,
            switch_thr,
            kernel_size,
        )

        # has reward criteria been met?
        reward_ok = block_left_rewards + block_right_rewards >= min_block_reward

        # conditions to switch:
        #   - planned block length reached
        #   - minimum reward requirement is reached
        #   - behavior is stable

        return block_length_ok and reward_ok and behavior_ok

    def generate_block(
        self,
        reward_families: list,
        reward_family_index: int,
        reward_pairs_n: int,
        base_reward_sum: float,
        block_history: list[Block],
        block_distribution: Union[UniformDistribution, ExponentialDistribution],
    ) -> Block:
        """
        Generate the next block for a coupled task.

        :param reward_families: Description
        :param reward_family_index: Description
        :param reward_pairs_n: Description
        :param base_reward_sum: Description
        :param reward_prob_history: Description
        :param block_distribution: Description
        """

        # determine candidate reward pairs
        reward_pairs = reward_families[reward_family_index][:reward_pairs_n]
        reward_prob = np.array(reward_pairs, dtype=float)
        reward_prob /= reward_prob.sum(axis=1, keepdims=True)
        reward_prob *= float(base_reward_sum)

        # create pool including all reward probabiliteis and mirrored pairs
        reward_prob_pool = np.vstack([reward_prob, np.fliplr(reward_prob)])

        if block_history:  # exclude previous block if history exists
            reward_prob_history = [[block.right_reward_prob, block.left_reward_prob] for block in block_history]
            last_block_reward_prob = reward_prob_history[:, -1]

            # remove blocks identical to last block
            reward_prob_pool = reward_prob_pool[np.any(reward_prob_pool != last_block_reward_prob, axis=1)]

            # remove blocks with same high-reward side (if last block had a clear high side)
            if last_block_reward_prob[0] != last_block_reward_prob[1]:
                high_side_last = last_block_reward_prob[0] > last_block_reward_prob[1]
                high_side_pool = reward_prob_pool[:, 0] > reward_prob_pool[:, 1]
                reward_prob_pool = reward_prob_pool[high_side_pool != high_side_last]

        # remove duplicates
        reward_prob_pool = np.unique(reward_prob_pool, axis=0)

        # randomly pick next block reward probability
        right_reward_prob, left_reward_prob = reward_prob_pool[random.choice(range(reward_prob_pool.shape[0]))]

        # randomly pick block length
        next_block_len = self.evaluate_distribution(block_distribution)

        return Block(
            right_reward_prob=right_reward_prob,
            left_reward_prob=left_reward_prob,
            min_length=next_block_len,
        )

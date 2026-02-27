import logging
from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from ..trial_models import TrialOutcome
from .block_based_trial_generator import (
    BlockBasedTrialGenerator,
    BlockBasedTrialGeneratorSpec,
)

logger = logging.getLogger(__name__)

BlockBehaviorEvaluationMode = Literal[
    "end",  # behavior stable at end of block to allow switching
    "anytime",  # behavior stable anytime in block to allow switching
]


class CoupledTrialGenerationEndConditions(BaseModel):
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


class CoupledTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
    type: Literal["CoupledTrialGenerator"] = "CoupledTrialGenerator"

    trial_generation_end_parameters: CoupledTrialGenerationEndConditions = Field(
        default=CoupledTrialGenerationEndConditions(), description="Conditions to end trial generation."
    )

    behavior_stability_parameters: Optional[BehaviorStabilityParameters] = Field(
        default=BehaviorStabilityParameters(),
        description="Parameters describing behavior stability required to switch blocks.",
    )
    extend_block_on_no_response: bool = Field(
        default=True,
        description="Add one trial to the min block length.",
    )

    def create_generator(self) -> "CoupledTrialGenerator":
        return CoupledTrialGenerator(self)


class CoupledTrialGenerator(BlockBasedTrialGenerator):
    spec: CoupledTrialGeneratorSpec

    def __init__(self, spec: CoupledTrialGeneratorSpec) -> None:
        """"""

        super().__init__(spec)
        self.start_time = datetime.now()

    def are_end_conditions_met(self) -> bool:
        """
        Check if end conditions are met to stop session
        """

        end_conditions = self.spec.trial_generation_end_parameters
        choice_history = self.is_right_choice_history

        time_elapsed = datetime.now() - self.start_time
        frac = end_conditions.ignore_ratio_threshold
        win = end_conditions.ignore_win

        if time_elapsed > end_conditions.min_time and choice_history[-win:].count(None) >= frac * win:
            logger.debug("Minimum time and ignored trial count exceeded.")
            return True

        if end_conditions.max_time < time_elapsed:
            logger.debug("Maximum session time exceeded.")
            return True

        if end_conditions.max_trial < len(choice_history):
            logger.debug("Maximum trial count exceeded.")
            return True

        return False

    def update(self, outcome: TrialOutcome) -> None:
        """
        Check if block should switch, generate next block if necessary, and  generate next trial

        :param outcome: trial outcome of previous trial
        """

        logger.info(f"Updating coupled trial generator with trial outcome of {outcome}")

        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)
        self.trials_in_block += 1

        if self.spec.baiting:
            if outcome.is_right_choice is True:
                logger.debug("Resesting right bait.")
                self.is_right_baited = False
            elif outcome.is_right_choice is False:
                logger.debug("Resesting left bait.")
                self.is_left_baited = False

        if self.spec.extend_block_on_no_response and outcome.is_right_choice is None:
            logger.info("Extending minimum block length due to ignored trial.")
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

        logger.info("Evaluating block behavior.")

        # do not prohibit block transition if does not rely on behavior or not enough trials to evaluate or reward probs are the same.
        if not beh_stability_params or left_reward_prob == right_reward_prob or len(choice_history) < kernel_size:
            logger.debug(
                "Behavior stability evaluation skipped: "
                f"parameters_missing={not bool(beh_stability_params)}, "
                f"rewards_equal={left_reward_prob == right_reward_prob}, "
                f"trials_available={len(choice_history)} < kernel_size({kernel_size})"
            )
            return True

        # compute fraction of right choices with running average using a sliding window
        block_history = choice_history[-(trials_in_block + kernel_size - 1) :]
        block_choice_frac = self.compute_choice_fraction(kernel_size, block_history)
        logger.debug(f"Choice fraction of block is {block_choice_frac}.")

        # margin based on right and left probabilities and scaled by switch threshold. Window for evaluating behavior
        delta = abs((left_reward_prob - right_reward_prob) * float(beh_stability_params.behavior_stability_fraction))
        threshold = (
            [0, left_reward_prob - delta] if left_reward_prob > right_reward_prob else [left_reward_prob + delta, 1]
        )
        logger.debug(f"Behavior stability threshold applied: {threshold}")

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
            logger.info(f"Evaluating last {min_stable} trials for end-of-block stability.")
            if len(points_above_threshold) < min_stable:
                logger.info("Not enough trials to evaluate stability at block end.")
                return False
            stable = np.all(points_above_threshold[-min_stable:])
            logger.info(f"Behavior stable at block end: {stable}")
            return stable

        elif mode == "anytime":
            # allows consecutive trials any time in the behavior
            logger.info(f"Evaluating block for stability anytime over {min_stable} consecutive trials.")
            run_len = 0
            for i, v in enumerate(points_above_threshold):
                if v:
                    run_len += 1
                else:
                    run_len = 0
                if run_len >= min_stable:
                    logger.info(f"Behavior stable at trial index {i}.")
                    return True
            logger.info("Behavior not stable in block anytime evaluation.")
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

        logger.info("Evaluating block switch.")

        # has planned block length been reached?
        block_length_ok = trials_in_block >= block_length
        logger.debug(f"Planned block length reached: {block_length_ok}")

        # is behavior qualified to switch?
        behavior_ok = self.is_behavior_stable(
            choice_history,
            right_reward_prob,
            left_reward_prob,
            beh_stability_params,
            trials_in_block,
            kernel_size,
        )
        logger.debug(f"Behavior meets stability criteria: {behavior_ok}")

        # has reward criteria been met?
        reward_ok = block_left_rewards + block_right_rewards >= min_block_reward
        logger.debug(f"Reward criterion satisfied: {reward_ok}")

        # conditions to switch:
        #   - planned block length reached
        #   - minimum reward requirement is reached
        #   - behavior is stable

        return block_length_ok and reward_ok and behavior_ok

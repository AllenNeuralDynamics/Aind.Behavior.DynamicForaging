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
    """Defines the conditions under which a foraging session should terminate."""

    ignore_win: int = Field(default=30, ge=0, description="Number of recent trials to check for ignored responses.")
    ignore_ratio_threshold: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Maximum fraction of ignored trials within the window before the session is ended.",
    )
    max_trial: int = Field(default=1000, ge=0, description="Maximum number of trials allowed in a session.")
    max_time: timedelta = Field(default=timedelta(minutes=75), description="Maximum session duration (min).")
    min_time: timedelta = Field(default=timedelta(minutes=30), description="Minimum session duration (min)")


class BehaviorStabilityParameters(BaseModel):
    """Parameters controlling when behavior is considered stable enough to switch blocks."""

    behavior_evaluation_mode: BlockBehaviorEvaluationMode = Field(
        default="end",
        description="When to evaluate stability — at the end of the block (end) or at any point during the block (anytime).",
        validate_default=True,
    )
    behavior_stability_fraction: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Fraction scaling reward-probability difference for behavior.",
    )
    min_consecutive_stable_trials: int = Field(
        default=5,
        ge=0,
        description="Minimum number of consecutive trials satisfying the behavioral stability fraction.",
    )


class CoupledTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
    type: Literal["CoupledTrialGenerator"] = "CoupledTrialGenerator"

    trial_generation_end_parameters: CoupledTrialGenerationEndConditions = Field(
        default=CoupledTrialGenerationEndConditions(), description="Conditions to end trial generation.", validate_default=True
    )

    behavior_stability_parameters: Optional[BehaviorStabilityParameters] = Field(
        default=BehaviorStabilityParameters(),
        description="Parameters controlling behavior-dependent block switching. If None, block switches rely only on length and reward criteria.",
    )
    extend_block_on_no_response: bool = Field(
        default=True,
        description="Whether to extend the minimum block length by one trial when the animal does not respond.",
    )

    def create_generator(self) -> "CoupledTrialGenerator":
        return CoupledTrialGenerator(self)


class CoupledTrialGenerator(BlockBasedTrialGenerator):
    """Trial generator for a coupled block-based dynamic foraging task.

    Extends BlockBasedTrialGenerator with session end conditions, baiting state
    management, and behavior-dependent block switching.

    Attributes:
        spec: The CoupledTrialGeneratorSpec defining task parameters.
        start_time: Timestamp recorded at initialization, used to track elapsed
            session time.
    """

    spec: CoupledTrialGeneratorSpec

    def __init__(self, spec: CoupledTrialGeneratorSpec) -> None:
        """Initializes the generator and records the session start time.

        Args:
            spec: The CoupledTrialGeneratorSpec defining task parameters.
        """

        super().__init__(spec)
        self.start_time = datetime.now()

    def _are_end_conditions_met(self) -> bool:
        """Checks whether the session should end.

        Evaluates three termination conditions: excessive ignored trials after
        minimum session time, maximum session time exceeded, and maximum trial
        count exceeded.

        Returns:
            True if any end condition is met, False otherwise.
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
        """Updates generator state from the previous trial outcome and switches block if criteria are met.

        Records choice and reward history, manages baiting state, optionally extends
        the block on no response, and triggers a block switch if all switch criteria
        are satisfied.

        Args:
            outcome: The TrialOutcome from the most recently completed trial.
        """

        logger.info(f"Updating coupled trial generator with trial outcome of {outcome}")

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

        if self.spec.extend_block_on_no_response and outcome.is_right_choice is None:
            logger.info("Extending minimum block length due to ignored trial.")
            self.block.min_length += 1

        switch_block = self._is_block_switch_allowed(
            trials_in_block=self.trials_in_block,
            min_block_reward=self.spec.min_block_reward,
            choice_history=self.is_right_choice_history,
            p_right_reward=self.block.p_right_reward,
            l_right_reward=self.block.l_right_reward,
            beh_stability_params=self.spec.behavior_stability_parameters,
            block_length=self.block.min_length,
            kernel_size=self.spec.kernel_size,
        )

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

    def _is_behavior_stable(
        self,
        choice_history: list,
        p_right_reward: float,
        l_right_reward: float,
        beh_stability_params: BehaviorStabilityParameters,
        trials_in_block: int,
        kernel_size: int = 2,
    ) -> Optional[bool]:
        """Evaluates whether the animal's choice behavior is stable enough to allow a block switch.

        Computes a sliding-window choice fraction and checks whether it falls within
        a threshold derived from the reward probability difference. Stability is
        assessed either at the end of the block or at any point, depending on
        the evaluation mode.

        Args:
            choice_history: Trial history with True for right, False for left,
                and None for ignored trials.
            p_right_reward: Reward probability for the right port in the current block.
            l_right_reward: Reward probability for the left port in the current block.
            beh_stability_params: Parameters defining the stability threshold and
                evaluation mode.
            trials_in_block: Number of trials elapsed in the current block.
            kernel_size: Sliding window size for computing choice fraction.

        Returns:
            True if behavior is stable or evaluation is skipped, False if behavior
            does not meet the stability criterion.

        Raises:
            ValueError: If the behavior evaluation mode is not recognized.
        """

        logger.info("Evaluating block behavior.")

        # do not prohibit block transition if does not rely on behavior or not enough trials to evaluate or reward probs are the same.
        if not beh_stability_params or l_right_reward == p_right_reward or len(choice_history) < kernel_size:
            logger.debug(
                "Behavior stability evaluation skipped: "
                f"parameters_missing={not bool(beh_stability_params)}, "
                f"rewards_equal={l_right_reward == p_right_reward}, "
                f"trials_available={len(choice_history)} < kernel_size({kernel_size})"
            )
            return True

        # compute fraction of right choices with running average using a sliding window
        block_history = choice_history[-(trials_in_block + kernel_size - 1) :]
        block_choice_frac = self.compute_choice_fraction(kernel_size, block_history)
        logger.debug(f"Choice fraction of block is {block_choice_frac}.")

        # margin based on right and left probabilities and scaled by switch threshold. Window for evaluating behavior
        delta = abs((l_right_reward - p_right_reward) * float(beh_stability_params.behavior_stability_fraction))
        threshold = [0, l_right_reward - delta] if l_right_reward > p_right_reward else [l_right_reward + delta, 1]
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
        """Computes a sliding-window fraction of right choices over the trial history.

        Ignores None (no-response) trials by treating them as NaN in the mean.

        Args:
            kernel_size: Number of trials in each sliding window.
            choice_history: Trial history with 1 for right, 0 for left, and
                None for ignored trials.

        Returns:
            Array of per-window right-choice fractions, of length
            len(choice_history) - kernel_size + 1.
        """

        n_windows = len(choice_history) - kernel_size + 1
        choice_fraction = np.empty(n_windows, dtype=float)  # create empty array to store running averages
        for i in range(n_windows):
            window = np.array(choice_history[i : i + kernel_size], dtype=float)
            choice_fraction[i] = np.nanmean(window)
        return choice_fraction

    def _is_block_switch_allowed(
        self,
        trials_in_block: int,
        min_block_reward: int,
        choice_history: list,
        p_right_reward: float,
        l_right_reward: float,
        beh_stability_params: BehaviorStabilityParameters,
        block_length: int,
        kernel_size: int = 2,
    ) -> bool:
        """Determines whether all criteria are met to switch to the next block.

        A block switch requires: the planned block length has been reached, the
        minimum reward count has been collected, and behavior meets the stability
        criterion.

        Args:
            trials_in_block: Number of trials elapsed in the current block.
            min_block_reward: Minimum total rewards required before switching.
            choice_history: Trial history with True for right, False for left,
                and None for ignored trials.
            p_right_reward: Reward probability for the right port in the current block.
            l_right_reward: Reward probability for the left port in the current block.
            beh_stability_params: Parameters defining the behavior stability criterion.
            block_length: Planned minimum number of trials in the current block.
            kernel_size: Sliding window size for computing choice fraction.

        Returns:
            True if all switch criteria are satisfied, False otherwise.
        """

        logger.info("Evaluating block switch.")

        # has planned block length been reached?
        block_length_ok = trials_in_block >= block_length
        logger.debug(f"Planned block length reached: {block_length_ok}")

        # is behavior qualified to switch?
        behavior_ok = self._is_behavior_stable(
            choice_history,
            p_right_reward,
            l_right_reward,
            beh_stability_params,
            trials_in_block,
            kernel_size,
        )
        logger.debug(f"Behavior meets stability criteria: {behavior_ok}")

        # has reward criteria been met?
        reward_ok = self.reward_history.count(False) + self.reward_history.count(True) >= min_block_reward
        logger.debug(f"Reward criterion satisfied: {reward_ok}")

        # conditions to switch:
        #   - planned block length reached
        #   - minimum reward requirement is reached
        #   - behavior is stable

        return block_length_ok and reward_ok and behavior_ok

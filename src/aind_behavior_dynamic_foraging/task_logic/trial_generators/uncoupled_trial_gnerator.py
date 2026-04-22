import logging
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from aind_behavior_services.task.distributions_utils import draw_sample
from pydantic import BaseModel, Field

from ..trial_models import TrialOutcome
from .block_based_trial_generator import Block, BlockBasedTrialGenerator, BlockBasedTrialGeneratorSpec

logger = logging.getLogger(__name__)


class UncoupledTrialGenerationEndConditions(BaseModel):
    """Defines the conditions under which a foraging session should terminate."""

    ignore_win: int = Field(default=30, ge=0, description="Number of recent trials to check for ignored responses.")
    ignore_ratio_threshold: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Maximum fraction of ignored trials within the window before the session is ended.",
    )
    max_trial: int = Field(default=1000, ge=0, description="Maximum number of trials allowed in a session.")
    max_time: float = Field(default=75 * 60, description="Maximum session duration (sec).")
    min_time: float = Field(default=30 * 60, description="Minimum session duration (sec)")


class UncoupledTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
    type: Literal["UncoupledTrialGenerator"] = "UncoupledTrialGenerator"

    trial_generation_end_parameters: UncoupledTrialGenerationEndConditions = Field(
        default=UncoupledTrialGenerationEndConditions(),
        description="Conditions to end trial generation.",
        validate_default=True,
    )

    reward_probabilities: list[float] = Field(
        default=[0.1, 0.5, 0.9], min_length=2, description="Reward probabilites to use during session."
    )
    maximum_dominance_streak: float = Field(
        default=3, description="Maximum number of consecutive blocks a side can have the higher probability."
    )

    def create_generator(self) -> "UncoupledTrialGenerator":
        return UncoupledTrialGenerator(self)


class UncoupledTrialGenerator(BlockBasedTrialGenerator):
    """Trial generator for a Uncoupled block-based dynamic foraging task.

    Attributes:
        spec: The UncoupledTrialGeneratorSpec defining task parameters.
        start_time: Timestamp recorded at initialization, used to track elapsed
            session time.
    """

    spec: UncoupledTrialGeneratorSpec

    def __init__(self, spec: UncoupledTrialGeneratorSpec) -> None:
        """Initializes the generator and records the session start time.

        Args:
            spec: The UncoupledTrialGeneratorSpec defining task parameters.
        """

        super().__init__(spec)
        self.start_time = datetime.now()

        block_len_min = spec.block_len.distribution_parameters.min
        block_len_max = spec.block_len.distribution_parameters.max
        self.block_length_stagger = (round(block_len_max - block_len_min - 0.5) / 2 + block_len_min) / 2

        self.block = self._generate_first_block()
        self.trials_in_right_block = 0
        self.right_dominance_streak = 0
        self.trials_in_left_block = 0
        self.left_dominance_streak = 0

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

        if (
            time_elapsed > timedelta(seconds=end_conditions.min_time)
            and choice_history[-win:].count(None) >= frac * win
        ):
            logger.debug("Minimum time and ignored trial count exceeded.")
            return True

        if timedelta(seconds=end_conditions.max_time) < time_elapsed:
            logger.debug("Maximum session time exceeded.")
            return True

        if end_conditions.max_trial < len(choice_history):
            logger.debug("Maximum trial count exceeded.")
            return True

        return False

    def update(self, outcome: TrialOutcome | str) -> None:
        """Updates generator state from the previous trial outcome and switches block if criteria are met.

        Increments the trial counters for both sides and checks whether either side
        has exceeded its minimum block length. If so, updates the dominance streaks
        based on the current block probabilities before generating a new block for
        the switching side.

        Args:
            outcome: The TrialOutcome from the most recently completed trial.
        """

        super().update(outcome)

        self.trials_in_left_block += 1
        self.trials_in_right_block += 1

        right_switching = self._is_block_switch_allowed(self.trials_in_right_block, self.block.right_length)
        left_switching = self._is_block_switch_allowed(self.trials_in_left_block, self.block.left_length)

        # update dominant block counts before switching
        self._update_dominance_streak(right_switching=right_switching, left_switching=left_switching)

        if right_switching:
            logger.info("Switching right block.")
            self.trials_in_right_block = 0

        if left_switching:
            logger.info("Switching left block.")
            self.trials_in_left_block = 0

        if right_switching or left_switching:
            self.block = self._generate_next_block(right_switching, self.block)

    def _update_dominance_streak(self, right_switching: bool, left_switching: bool) -> None:
        if not (right_switching or left_switching):
            return

        if self.block.p_right_reward > self.block.p_left_reward:
            self.right_dominance_streak += 1
            self.left_dominance_streak = 0
        elif self.block.p_left_reward > self.block.p_right_reward:
            self.left_dominance_streak += 1
            self.right_dominance_streak = 0
        else:
            self.right_dominance_streak += 1
            self.left_dominance_streak += 1

    def _is_block_switch_allowed(self, trials_in_block: int, block_len: int) -> bool:

        return trials_in_block > block_len

    def _generate_first_block(self) -> Block:
        """Generate the initial block for both sides, ensuring neither side starts at minimum
        reward probability and staggering the lower side's block length.

        Returns:
            A Block with valid initial reward probabilities and staggered block lengths.
        """

        p_left_reward = np.random.choice(self.spec.reward_probabilities)
        p_right_reward = np.random.choice(self.spec.reward_probabilities)
        right_length = draw_sample(self.spec.block_len)
        left_length = draw_sample(self.spec.block_len)

        while p_right_reward == p_left_reward == min(self.spec.reward_probabilities):
            if np.random.choice([True, False]):
                p_right_reward = np.random.choice(self.spec.reward_probabilities)
            else:
                p_left_reward = np.random.choice(self.spec.reward_probabilities)

        if p_right_reward < p_left_reward:
            right_length -= self.block_length_stagger
        elif p_left_reward < p_right_reward:
            left_length -= self.block_length_stagger
        else:
            if np.random.choice([True, False]):
                right_length -= self.block_length_stagger
            else:
                left_length -= self.block_length_stagger

        return Block(
            p_right_reward=p_right_reward,
            p_left_reward=p_left_reward,
            right_length=right_length,
            left_length=left_length,
        )

    def _generate_next_block(
        self,
        right_switching: bool,
        block: Block,
    ) -> Block:
        """Generate a new block for the switching side, with both-lowest correction if needed.

        Updates the reward probability and length for the switching side. If the switching
        side is right (right_switching=True), the right side is updated; otherwise the left
        side is updated. After updating, if both sides would be at the minimum reward
        probability, the switching side's block is staggered and the non-switching side is
        resampled to a non-minimum probability to ensure the task remains rewarding.

        Args:
            right_switching: If True, generate a new block for the right side.
                             If False, generate a new block for the left side.
            block: The current Block whose values are used as a base for the new block.

        Returns:
            A new Block with updated reward probability and length for the switching side,
            and both-lowest correction applied if necessary.
        """

        p_right_reward = block.p_right_reward
        p_left_reward = block.p_left_reward
        right_length = block.right_length
        left_length = block.left_length

        if right_switching:
            p_right_reward = self._determine_reward_probability(
                self.right_dominance_streak,
                self.spec.maximum_dominance_streak,
                [x for x in self.spec.reward_probabilities if x != p_right_reward],
            )
            right_length = draw_sample(self.spec.block_len)

        else:
            p_left_reward = self._determine_reward_probability(
                self.left_dominance_streak,
                self.spec.maximum_dominance_streak,
                [x for x in self.spec.reward_probabilities if x != p_left_reward],
            )
            left_length = draw_sample(self.spec.block_len)

        p_min = min(self.spec.reward_probabilities)
        if p_right_reward == p_left_reward == p_min:
            available = [x for x in self.spec.reward_probabilities if x != p_min]
            if right_switching:
                right_length -= self.block_length_stagger
                p_left_reward = np.random.choice(available)
                left_length = draw_sample(self.spec.block_len)
            else:
                left_length -= self.block_length_stagger
                p_right_reward = np.random.choice(available)
                right_length = draw_sample(self.spec.block_len)

        return Block(
            p_right_reward=p_right_reward,
            p_left_reward=p_left_reward,
            right_length=right_length,
            left_length=left_length,
        )

    @staticmethod
    def _determine_reward_probability(
        dominance_streak: int, maximum_dominance_streak: int, reward_probabilities: list[float]
    ) -> float:
        """Selects randomly from the provided reward probabilities unless the dominance streak
        has reached or exceeded the maximum, in which case the minimum available probability
        is returned to rebalance the task.

        Args:
            dominance_streak: Number of consecutive blocks the switching side has had
                a higher or equal reward probability than the other side.
            maximum_dominance_streak: Threshold at which the switching side is forced
                to the minimum reward probability.
            reward_probabilities: List of candidate probabilities to sample from.
                Should already exclude the previous block's probability to prevent repeats.

        Returns:
            A reward probability for the new block.
        """

        if dominance_streak >= maximum_dominance_streak:
            min_p_reward = min(reward_probabilities)
            logger.info(
                f"Side exceded maximum number of higher probability blocks. Forcing to lower side {min_p_reward}."
            )
            return min_p_reward

        else:
            return np.random.choice(reward_probabilities)

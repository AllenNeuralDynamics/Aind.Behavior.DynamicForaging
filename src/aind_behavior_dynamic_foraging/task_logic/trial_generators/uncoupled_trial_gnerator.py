import logging
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from aind_behavior_services.task.distributions import (
    UniformDistribution,
    UniformDistributionParameters,
)
from aind_behavior_services.task.distributions_utils import draw_sample
from pydantic import BaseModel, Field

from ..trial_models import TrialOutcome
from .block_based_trial_generator import (
    Block,
    BlockBasedTrialGenerator,
    BlockBasedTrialGeneratorSpec,
)

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
        default=[0.1, 0.5, 0.9],
        min_length=2,
        description="Reward probabilites to use during session.",
    )
    maximum_dominance_streak: float = Field(
        default=3,
        description="Maximum number of consecutive blocks a side can have the higher probability.",
    )

    block_len: UniformDistribution = Field(
        default=UniformDistribution(
            distribution_parameters=UniformDistributionParameters(min=20, max=60),
        ),
        description="Distribution describing block length.",
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
        """Records the session start time, calculates right and left block stagger, and generates first block.
        Code adapted from https://github.com/AllenNeuralDynamics/dynamic-foraging-task/blob/develop/src/foraging_gui/reward_schedules/uncoupled_block.py
        Right and left reward probabilities evolve independently in separate blocks.
        Block lengths are staggered. A dominance streak counter prevents one side from
        holding the higher reward probability for too many consecutive blocks.

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
            logger.info("Minimum time and ignored trial count exceeded.")
            return True

        if timedelta(seconds=end_conditions.max_time) < time_elapsed:
            logger.info("Maximum session time exceeded.")
            return True

        if end_conditions.max_trial < len(choice_history):
            logger.info("Maximum trial count exceeded.")
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

        switches = []
        if left_switching := self._is_block_switch_allowed(self.trials_in_left_block, self.block.left_length):
            switches.append(False)
        if right_switching := self._is_block_switch_allowed(self.trials_in_right_block, self.block.right_length):
            switches.append(True)

        if right_switching or left_switching:
            self._update_dominance_streak()  # update dominant block counts before switching
            for switch in switches:
                new_block = self._generate_next_block(
                    right_switching=switch,
                    right_dominance_streak=self.right_dominance_streak,
                    left_dominance_streak=self.left_dominance_streak,
                    max_dominance_streak=self.spec.maximum_dominance_streak,
                    reward_probabilities=self.spec.reward_probabilities,
                    block_len=self.spec.block_len,
                    block_stagger=self.block_length_stagger,
                    block=self.block,
                )
            # reset the counter for any side whose probability changed
            if new_block.p_right_reward != self.block.p_right_reward:
                self.trials_in_right_block = 0
            if new_block.p_left_reward != self.block.p_left_reward:
                self.trials_in_left_block = 0
            self.block = new_block
            logger.info(
                f"New block generated: p_right_reward={self.block.p_right_reward}, p_left_reward={self.block.p_left_reward}, right_length={self.block.right_length}, left_length={self.block.left_length}."
            )

    def _update_dominance_streak(self) -> None:
        """Update the per-side dominance streak counters based on the current block.

        The streak for the dominant side increments and the other side resets to zero.
        If probabilities are equal, both streaks increment. These counters are used by
        _generate_next_block to force a side to the minimum probability once it has
        dominated for too many consecutive blocks.
        """
        if self.block.p_right_reward > self.block.p_left_reward:
            logger.info("Increminting right dominance streak and reseting left.")
            self.right_dominance_streak += 1
            self.left_dominance_streak = 0
        elif self.block.p_left_reward > self.block.p_right_reward:
            logger.info("Increminting left dominance streak and reseting right.")
            self.left_dominance_streak += 1
            self.right_dominance_streak = 0
        else:
            logger.info("Increminting right and left dominance streak.")
            self.right_dominance_streak += 1
            self.left_dominance_streak += 1

    def _is_block_switch_allowed(self, trials_in_block: int, block_len: int) -> bool:
        """Return True if the trial counter has exceeded the current block length."""
        return trials_in_block > block_len

    def _generate_first_block(self) -> Block:
        """Generate the initial block for both sides, ensuring neither side starts at minimum
        reward probability and staggering the lower side's block length.

        Returns:
            A Block with valid initial reward probabilities and staggered block lengths.
        """

        logger.info("Generating first block.")
        p_left_reward = np.random.choice(self.spec.reward_probabilities)
        p_right_reward = np.random.choice(self.spec.reward_probabilities)
        right_length = round(draw_sample(self.spec.block_len))
        left_length = round(draw_sample(self.spec.block_len))
        while p_right_reward == p_left_reward == min(self.spec.reward_probabilities):
            if np.random.choice([True, False]):
                logger.debug("Right and left reward are both equal to min. Redrawing right probability.")
                p_right_reward = np.random.choice(self.spec.reward_probabilities)
            else:
                logger.debug("Right and left reward are both equal to min. Redrawing left probability.")
                p_left_reward = np.random.choice(self.spec.reward_probabilities)

        if p_right_reward < p_left_reward:
            logger.debug("Staggering right block.")
            right_length -= self.block_length_stagger
        elif p_left_reward < p_right_reward:
            logger.debug("Staggering left block.")
            left_length -= self.block_length_stagger
        else:
            if np.random.choice([True, False]):
                logger.debug("Staggering right block.")
                right_length -= self.block_length_stagger
            else:
                logger.debug("Staggering left block.")
                left_length -= self.block_length_stagger

        return Block(
            p_right_reward=p_right_reward,
            p_left_reward=p_left_reward,
            right_length=right_length,
            left_length=left_length,
        )

    @staticmethod
    def _generate_next_block(
        right_switching: bool,
        right_dominance_streak: int,
        left_dominance_streak: int,
        max_dominance_streak: int,
        reward_probabilities: list[float],
        block_len: UniformDistribution,
        block_stagger: int,
        block: Block,
    ) -> Block:
        """Generate a new block for the switching side, with both-lowest correction if needed.

        Updates the reward probability and length for the switching side. If the switching
        side is right (right_switching=True), the right side is updated; otherwise the left
        side is updated. After updating, if both sides are at the minimum reward
        probability, the switching side's block is staggered and the non-switching side is
        resampled to a non-minimum probability to ensure the task remains rewarding.

        Args:
            right_switching: If True, generate a new block for the right side.
                             If False, generate a new block for the left side. Since blocks are staggered, blocks will never switch at same trial.
            block: The current Block whose values are used as a base for the new block.
            right_dominance_streak: Number of consecutive blocks the right side has had
                a higher or equal reward probability than the left side.
            left_dominance_streak: Number of consecutive blocks the left side has had
                a higher or equal reward probability than the right side.
            max_dominance_streak: Threshold at which the switching side is forced
                to the minimum reward probability.
            reward_probabilities: List of candidate probabilities to sample from.
                Should already exclude the previous block's probability to prevent repeats.
            block_len: Disribution used to calculate trials in block.
            block_stagger: Number of trials to stagger right and left block length.

        Returns:
            A new Block with updated reward probability and length for the switching side,
            and both-lowest correction applied if necessary.
        """

        p_right_reward = block.p_right_reward
        p_left_reward = block.p_left_reward
        right_length = block.right_length
        left_length = block.left_length
        p_min = min(reward_probabilities)

        if right_switching:
            logger.info("Generating right block.")
            r_available = [x for x in reward_probabilities if x != p_right_reward]
            p_right_reward = np.random.choice(r_available) if right_dominance_streak < max_dominance_streak else p_min
            right_length = round(draw_sample(block_len))

            if p_right_reward == p_left_reward == p_min:
                logger.info(
                    "Right and left reward are both equal to min. Staggering right block length and generating new left block."
                )
                right_length -= block_stagger
                p_left_reward = np.random.choice([x for x in reward_probabilities if x != p_min])
                left_length = round(draw_sample(block_len))
        else:
            logger.info("Generating left block.")
            l_available = [x for x in reward_probabilities if x != p_left_reward]
            p_left_reward = np.random.choice(l_available) if left_dominance_streak < max_dominance_streak else p_min
            left_length = round(draw_sample(block_len))

            if p_right_reward == p_left_reward == p_min:
                logger.info(
                    "Right and left reward are both equal to min. Staggering left block length and generating new right block."
                )
                left_length -= block_stagger
                p_right_reward = np.random.choice([x for x in reward_probabilities if x != p_min])
                right_length = round(draw_sample(block_len))

        return Block(
            p_right_reward=p_right_reward,
            p_left_reward=p_left_reward,
            right_length=right_length,
            left_length=left_length,
        )

import logging
import unittest

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators.uncoupled_trial_gnerator import (
    Block,
    TrialOutcome,
    UncoupledTrialGenerator,
    UncoupledTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial

from .util import simulate_response

logging.basicConfig(level=logging.DEBUG)


class TestUncoupledTrialGenerator(unittest.TestCase):
    """Unit tests for UncoupledTrialGenerator."""

    def setUp(self):
        np.random.seed(42)
        self.spec = UncoupledTrialGeneratorSpec()
        self.generator = UncoupledTrialGenerator(self.spec)

    def test_session(self):
        """Simulates a full experimental session to verify generator stability."""

        trial = Trial()
        outcome = TrialOutcome(
            trial=trial,
            is_right_choice=np.random.choice([True, False, None]),
            is_rewarded=np.random.choice([True, False]),
        )
        for i in range(500):
            trial = self.generator.next()
            self.generator.update(outcome)
            outcome = simulate_response(
                previous_reward=outcome.is_rewarded,
                previous_choice=outcome.is_right_choice,
                previous_left_bait=False,
                previous_right_bait=False,
            )

        if not trial:
            return

    ### Test _generate_first_block ###

    def test_first_block_never_both_at_minimum(self):
        """Both sides should never simultaneously start at minimum probability."""
        for seed in range(500):
            self.generator._generate_first_block()
            self.assertFalse(
                self.generator.block.p_right_reward
                == self.generator.block.p_left_reward
                == min(self.spec.reward_probabilities),
            )

    def test_first_block_lengths_are_positive(self):
        """Block lengths should remain positive after stagger is applied."""
        for seed in range(500):
            self.generator._generate_first_block()
            self.assertGreater(self.generator.block.right_length, 0)
            self.assertGreater(self.generator.block.left_length, 0)

    ### Test _generate_next_block ###

    def test_switching_side_probability_does_not_repeat(self):
        """After a switch, the new probability should differ from the previous one."""

        prev_block = self.generator.block
        self.generator.trials_in_right_block = 1
        self.generator.trials_in_left_block = 1
        for seed in range(500):
            self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))

            if self.generator.trials_in_right_block == 0:
                # block switch
                self.assertNotEqual(
                    self.generator.block.p_right_reward,
                    prev_block.p_right_reward,
                )
            if self.generator.trials_in_left_block == 0:
                self.assertNotEqual(
                    self.generator.block.p_left_reward,
                    prev_block.p_left_reward,
                )
            prev_block = self.generator.block

    def test_both_sides_never_simultaneously_at_minimum(self):
        """Both sides should never simultaneously be at the minimum reward probability."""

        p_min = min(self.spec.reward_probabilities)
        for seed in range(500):
            self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))
            self.assertFalse(
                self.generator.block.p_right_reward == p_min and self.generator.block.p_left_reward == p_min
            )

    ### Test _update_dominance_streak ###

    def test_right_streak_increments_when_right_is_higher(self):
        """Right dominance streak should increment when right probability is higher."""

        self.generator.block = Block(p_right_reward=0.9, p_left_reward=0.1, right_length=0, left_length=1000)
        self.generator.right_dominance_streak = 0
        self.generator.left_dominance_streak = 0
        self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))
        self.assertEqual(self.generator.right_dominance_streak, 1)
        self.assertEqual(self.generator.left_dominance_streak, 0)

    def test_left_streak_increments_when_left_is_higher(self):
        """Left dominance streak should increment when left probability is higher."""

        self.generator.block = Block(p_right_reward=0.1, p_left_reward=0.9, right_length=0, left_length=1000)
        self.generator.right_dominance_streak = 0
        self.generator.left_dominance_streak = 0
        self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))
        self.assertEqual(self.generator.right_dominance_streak, 0)
        self.assertEqual(self.generator.left_dominance_streak, 1)

    def test_both_streaks_increment_when_equal(self):
        """Both streaks should increment when probabilities are equal."""

        self.generator.block = Block(p_right_reward=0.1, p_left_reward=0.9, right_length=0, left_length=1000)
        self.generator.right_dominance_streak = 0
        self.generator.left_dominance_streak = 0
        self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))
        self.assertEqual(self.generator.right_dominance_streak, 0)
        self.assertEqual(self.generator.left_dominance_streak, 1)

    def test_streak_does_not_update_when_no_switch(self):
        """Dominance streaks should not change when no block switch occurs."""

        self.generator.block = Block(p_right_reward=0.1, p_left_reward=0.9, right_length=1000, left_length=1000)
        self.generator.right_dominance_streak = 2
        self.generator.left_dominance_streak = 0
        self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))
        self.assertEqual(self.generator.right_dominance_streak, 2)
        self.assertEqual(self.generator.left_dominance_streak, 0)

    def test_trial_counters_increment_each_trial(self):
        """Trial counters should increment by one each trial when no switch occurs."""

        self.generator.block = Block(p_right_reward=0.1, p_left_reward=0.9, right_length=1000, left_length=1000)
        self.generator.trials_in_right_block = 0
        self.generator.trials_in_left_block = 0
        for i in range(1, 6):
            self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))
            self.assertEqual(self.generator.trials_in_right_block, i)
            self.assertEqual(self.generator.trials_in_left_block, i)

    def test_right_counter_resets_on_right_switch(self):
        """Right trial counter should reset to zero when right block switches."""

        self.generator.block = Block(p_right_reward=0.1, p_left_reward=0.9, right_length=0, left_length=1000)
        self.generator.trials_in_right_block = 100
        self.generator.trials_in_left_block = 100
        self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))
        self.assertEqual(self.generator.trials_in_right_block, 0)

    def test_left_counter_resets_on_left_switch(self):
        """Left trial counter should reset to zero when left block switches."""

        self.generator.block = Block(p_right_reward=0.1, p_left_reward=0.9, right_length=1000, left_length=0)
        self.generator.trials_in_right_block = 100
        self.generator.trials_in_left_block = 100
        self.generator.update(TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True))
        self.assertEqual(self.generator.trials_in_left_block, 0)


if __name__ == "__main__":
    unittest.main()

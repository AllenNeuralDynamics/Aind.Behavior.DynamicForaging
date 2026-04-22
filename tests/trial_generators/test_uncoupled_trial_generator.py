import unittest
import numpy as np
from unittest.mock import patch
from aind_behavior_dynamic_foraging.task_logic.trial_generators.uncoupled_trial_gnerator import (
    Block,
    UncoupledTrialGenerator,
    UncoupledTrialGeneratorSpec,
    TrialOutcome
)

class TestUncoupledTrialGenerator(unittest.TestCase):
    """Unit tests for UncoupledTrialGenerator."""

    def setUp(self):
        np.random.seed(42)
        self.spec = UncoupledTrialGeneratorSpec()
        self.generator = UncoupledTrialGenerator(self.spec)

    def _make_block(self, p_right=0.9, p_left=0.1, right_length=100, left_length=100):
        return Block(
            p_right_reward=p_right,
            p_left_reward=p_left,
            right_length=right_length,
            left_length=left_length,
        )

    ### Test _generate_first_block ###

    def test_first_block_never_both_at_minimum(self):
        """Both sides should never simultaneously start at minimum probability."""
        for seed in range(100):
            self.generator._generate_first_block()
            self.assertFalse(
                self.generator.block.p_right_reward == self.generator.block.p_left_reward == self.p_min,
                f"Seed {seed}: both sides started at minimum probability.",
            )

    def test_first_block_lower_side_is_staggered(self):
        """The side with the lower starting probability should have a shorter block length."""
        for seed in range(100):
            self.generator._generate_first_block()
            if self.generator.block.p_right_reward < self.generator.block.p_left_reward:
                self.assertLess(self.generator.block.right_length, self.generator.block.left_length,
                                f"Seed {seed}: right is lower but not staggered.")
            elif self.generator.block.p_left_reward < self.generator.block.p_right_reward:
                self.assertLess(self.generator.block.left_length, self.generator.block.right_length,
                                f"Seed {seed}: left is lower but not staggered.")

    def test_first_block_lengths_are_positive(self):
        """Block lengths should remain positive after stagger is applied."""
        for seed in range(100):
            self.generator._generate_first_block()
            self.assertGreater(self.generator.block.right_length, 0,
                               f"Seed {seed}: right block length is non-positive.")
            self.assertGreater(self.generator.block.left_length, 0,
                               f"Seed {seed}: left block length is non-positive.")

    ### Test _generate_next_block ###

    def test_switching_side_probability_does_not_repeat(self):
        """After a switch, the new probability should differ from the previous one."""
        for seed in range(100):
            prev_right = self.generator.block.p_right_reward
            prev_left = self.generator.block.p_left_reward

            self.generator.block = self._make_block(right_length=1, left_length=1000)
            self.generator.update(TrialOutcome())
            self.generator.update(TrialOutcome())  # triggers right switch

            self.assertNotEqual(self.generator.block.p_right_reward, prev_right,
                                f"Seed {seed}: right probability repeated after switch.")

    def test_both_sides_never_simultaneously_at_minimum(self):
        """Both sides should never simultaneously be at the minimum reward probability."""
        for seed in range(50):
            for trial in range(500):
                self.generator.update(TrialOutcome())
                self.assertFalse(
                    self.generator.block.p_right_reward == self.generator.block.p_left_reward == min(self.spec.reward_probabilities),
                    f"Seed {seed}, trial {trial}: both sides at minimum.",
                )


    ### Test _determine_reward_probability ###

    def test_force_lowest_when_streak_exceeded(self):
        """Side should be forced to minimum probability when dominance streak is exceeded."""
        self.generator.right_dominance_streak = self.generator.spec.maximum_dominance_streak
        self.generator.block = self._make_block(p_right=0.9, p_left=0.1, right_length=1, left_length=1000)
        self.generator.update(TrialOutcome())
        self.generator.update(TrialOutcome())  # triggers right switch with streak at max
        self.assertEqual(self.generator.block.p_right_reward, self.p_min)

    def test_no_force_below_maximum_streak(self):
        """Side should not be forced to minimum when streak is below maximum."""
        
        self.generator.right_dominance_streak = self.generator.spec.maximum_dominance_streak - 1
        self.generator.block = self._make_block(p_right=0.9, p_left=0.1, right_length=1, left_length=1000)
        prev_left = self.generator.block.p_left_reward

        # Run many seeds to verify minimum is not always returned
        results = set()
        for seed in range(20):
            np.random.seed(seed)
            self.generator.block = self._make_block(p_right=0.9, p_left=0.5, right_length=1, left_length=1000)
            self.generator.right_dominance_streak = self.generator.spec.maximum_dominance_streak - 1
            self.generator.update(TrialOutcome())
            self.generator.update(TrialOutcome())
            results.add(self.generator.block.p_right_reward)

        self.assertGreater(len(results), 1, "Expected varied probabilities below max streak.")

    # ------------------------------------------------------------------
    # _update_dominance_streak
    # ------------------------------------------------------------------

    def test_right_streak_increments_when_right_is_higher(self):
        """Right dominance streak should increment when right probability is higher."""
        
        self.generator.block = self._make_block(p_right=0.9, p_left=0.1, right_length=1, left_length=1000)
        self.generator.right_dominance_streak = 0
        self.generatorleft_dominance_streak = 0
        self.generator.update(TrialOutcome())
        self.generator.update(TrialOutcome())
        self.assertEqual(self.generator.right_dominance_streak, 1)
        self.assertEqual(self.generatorleft_dominance_streak, 0)

    def test_left_streak_increments_when_left_is_higher(self):
        """Left dominance streak should increment when left probability is higher."""
        
        self.generator.block = self._make_block(p_right=0.1, p_left=0.9, right_length=1, left_length=1000)
        self.generator.right_dominance_streak = 0
        self.generatorleft_dominance_streak = 0
        self.generator.update(TrialOutcome())
        self.generator.update(TrialOutcome())
        self.assertEqual(self.generatorleft_dominance_streak, 1)
        self.assertEqual(self.generator.right_dominance_streak, 0)

    def test_both_streaks_increment_when_equal(self):
        """Both streaks should increment when probabilities are equal."""
        
        self.generator.block = self._make_block(p_right=0.5, p_left=0.5, right_length=1, left_length=1000)
        self.generator.right_dominance_streak = 0
        self.generatorleft_dominance_streak = 0
        self.generator.update(TrialOutcome())
        self.generator.update(TrialOutcome())
        self.assertEqual(self.generator.right_dominance_streak, 1)
        self.assertEqual(self.generatorleft_dominance_streak, 1)

    def test_streak_does_not_update_when_no_switch(self):
        """Dominance streaks should not change when no block switch occurs."""
        
        self.generator.block = self._make_block(right_length=1000, left_length=1000)
        self.generator.right_dominance_streak = 2
        self.generatorleft_dominance_streak = 0
        self.generator.update(TrialOutcome())
        self.assertEqual(self.generator.right_dominance_streak, 2)
        self.assertEqual(self.generatorleft_dominance_streak, 0)

    def test_trial_counters_increment_each_trial(self):
        """Trial counters should increment by one each trial when no switch occurs."""
        
        self.generator.block = self._make_block(right_length=1000, left_length=1000)
        self.generatortrials_in_right_block = 0
        self.generatortrials_in_left_block = 0
        for i in range(1, 6):
            self.generator.update(TrialOutcome())
            self.assertEqual(self.generatortrials_in_right_block, i)
            self.assertEqual(self.generatortrials_in_left_block, i)

    def test_right_counter_resets_on_right_switch(self):
        """Right trial counter should reset to zero when right block switches."""
        
        self.generator.block = self._make_block(right_length=1, left_length=1000)
        self.generator.update(TrialOutcome())
        self.generator.update(TrialOutcome())
        self.assertEqual(self.generatortrials_in_right_block, 0)

    def test_left_counter_resets_on_left_switch(self):
        """Left trial counter should reset to zero when left block switches."""
        
        self.generator.block = self._make_block(right_length=1000, left_length=1)
        self.generator.update(TrialOutcome())
        self.generator.update(TrialOutcome())
        self.assertEqual(self.generatortrials_in_left_block, 0)

    def test_right_counter_does_not_reset_on_left_switch(self):
        """Right trial counter should not reset when only the left block switches."""
        
        self.generator.block = self._make_block(right_length=1000, left_length=1)
        self.generatortrials_in_right_block = 0
        self.generator.update(TrialOutcome())
        self.generator.update(TrialOutcome())
        self.assertEqual(self.generatortrials_in_right_block, 2)


if __name__ == "__main__":
    unittest.main()
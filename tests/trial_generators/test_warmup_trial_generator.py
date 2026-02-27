import unittest

from aind_behavior_dynamic_foraging.task_logic.trial_generators.warmup_trial_generator import (
    WarmupTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome


def make_outcome(is_right_choice: bool | None, is_rewarded: bool) -> TrialOutcome:
    return TrialOutcome(trial=Trial(), is_right_choice=is_right_choice, is_rewarded=is_rewarded)


class TestWarmupEndConditions(unittest.TestCase):
    def setUp(self):
        self.spec = WarmupTrialGeneratorSpec()
        self.generator = self.spec.create_generator()

    def test_end_conditions_not_met_too_few_trials(self):
        self.generator.is_right_choice_history.append([True] * 5)
        self.assertFalse(self.generator.are_end_conditions_met())

    def test_end_conditions_not_met_high_bias(self):
        # all right choices = biased
        self.generator.is_right_choice_history.append([True] * 10)
        self.assertFalse(self.generator.are_end_conditions_met())

    def test_end_conditions_not_met_low_response_rate(self):
        # ignored
        self.generator.is_right_choice_history.append([None] * 10)
        self.assertFalse(self.generator.are_end_conditions_met())

    def test_end_conditions_met(self):
        # balanced choices, high response rate
        for i in range(50):
            self.generator.is_right_choice_history.append(i % 2 == 0)
        self.assertTrue(self.generator.are_end_conditions_met())

    ### block switches ###

    def test_block_switches_every_update(self):
        initial_block = self.generator.block
        self.generator.update(make_outcome(is_right_choice=True, is_rewarded=True))
        self.assertIsNot(self.generator.block, initial_block)

    def test_block_switches_on_ignored_trial(self):
        initial_block = self.generator.block
        self.generator.update(make_outcome(is_right_choice=None, is_rewarded=False))
        self.assertIsNot(self.generator.block, initial_block)

    def test_trials_in_block_resets_on_update(self):
        self.generator.trials_in_block = 5
        self.generator.update(make_outcome(is_right_choice=True, is_rewarded=True))
        self.assertEqual(self.generator.trials_in_block, 0)

    def test_block_added_to_history(self):
        initial_history_len = len(self.generator.block_history)
        self.generator.update(make_outcome(is_right_choice=True, is_rewarded=True))
        self.assertEqual(len(self.generator.block_history), initial_history_len + 1)

    def test_bait_resets_on_right_choice(self):
        self.generator.is_right_baited = True
        self.generator.update(make_outcome(is_right_choice=True, is_rewarded=True))
        self.assertFalse(self.generator.is_right_baited)

    def test_bait_resets_on_left_choice(self):
        self.generator.is_left_baited = True
        self.generator.update(make_outcome(is_right_choice=False, is_rewarded=True))
        self.assertFalse(self.generator.is_left_baited)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_warmup_trial_generator import (
    CoupledWarmupTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome

from .util import simulate_response


def make_outcome(is_right_choice: bool | None, is_rewarded: bool) -> TrialOutcome:
    return TrialOutcome(trial=Trial(), is_right_choice=is_right_choice, is_rewarded=is_rewarded)


class TestWarmup(unittest.TestCase):
    def setUp(self):
        self.spec = CoupledWarmupTrialGeneratorSpec()
        self.generator = self.spec.create_generator()

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

    def test_end_conditions_not_met_too_few_trials(self):
        self.generator.is_right_choice_history.append([True] * 5)
        self.assertFalse(self.generator._are_end_conditions_met())

    def test_end_conditions_not_met_high_bias(self):
        # all right choices = biased
        self.generator.is_right_choice_history.append([True] * 10)
        self.assertFalse(self.generator._are_end_conditions_met())

    def test_end_conditions_not_met_low_response_rate(self):
        # ignored
        self.generator.is_right_choice_history.append([None] * 10)
        self.assertFalse(self.generator._are_end_conditions_met())

    def test_end_conditions_met(self):
        # balanced choices, high response rate
        for i in range(50):
            self.generator.is_right_choice_history.append(i % 2 == 0)
        self.assertTrue(self.generator._are_end_conditions_met())

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

    def test_bait_on_no_choice(self):
        self.generator.is_left_baited = True
        self.generator.is_right_baited = True
        self.generator.update(make_outcome(is_right_choice=None, is_rewarded=True))
        self.assertTrue(self.generator.is_left_baited)
        self.assertTrue(self.generator.is_right_baited)

    #### test update ####
    def test_string_trial_outcome(self):
        string_outcome = TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True).model_dump_json()
        self.generator.update(string_outcome)


if __name__ == "__main__":
    unittest.main()

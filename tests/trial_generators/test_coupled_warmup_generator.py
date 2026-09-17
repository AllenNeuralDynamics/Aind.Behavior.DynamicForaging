import unittest

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_warmup_trial_generator import (
    CoupledTrialGenerator,
    CoupledWarmupTrialGeneratorSpec,
    WarmupTrialGenerator,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome
from tests.trial_generators.util import simulate_response


def make_outcome(is_right_choice: bool | None, is_rewarded: bool) -> TrialOutcome:
    return TrialOutcome(trial=Trial(), is_right_choice=is_right_choice, is_rewarded=is_rewarded)


class TestCoupledWarmupGenerator(unittest.TestCase):
    def setUp(self):
        self.spec = CoupledWarmupTrialGeneratorSpec()
        self.generator = self.spec.create_generator()

    def test_session(self):
        """Simulates a full experimental session to verify generator stability."""

        trial = self.generator.next()
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
                trial=trial,
            )

        if not trial:
            return

    def test_warmup_generator_active_until_end_conditions_met(self):
        warmup = self.generator.warmup_generator
        for i in range(50):
            warmup.is_right_choice_history.append(i % 2 == 0)
            warmup.outcome_history.append(make_outcome(i % 2 == 0, i % 3 == 0))
            warmup.reward_history.append(i % 3 == 0)

        self.assertTrue(warmup._are_end_conditions_met())
        self.assertIs(self.generator._active_generator, self.generator.warmup_generator)

    def test_switches_to_coupled_generator_after_warmup_completion(self):

        trial = self.generator.next()
        outcome = TrialOutcome(
            trial=trial,
            is_right_choice=np.random.choice([True, False, None]),
            is_rewarded=np.random.choice([True, False]),
        )
        for i in range(50):
            trial = self.generator.next()
            self.generator.update(outcome)
            outcome = simulate_response(
                previous_reward=outcome.is_rewarded,
                previous_choice=outcome.is_right_choice,
                previous_left_bait=False,
                previous_right_bait=False,
                trial=trial,
            )
        self.assertTrue(isinstance(self.generator._active_generator, WarmupTrialGenerator))
        self.assertTrue(self.generator._active_generator._are_end_conditions_met())
        self.generator.next()
        self.assertTrue(isinstance(self.generator._active_generator, CoupledTrialGenerator))

    def test_state_transfer_from_warmup_to_coupled_generator(self):
        trial = self.generator.next()
        outcome = TrialOutcome(
            trial=trial,
            is_right_choice=np.random.choice([True, False, None]),
            is_rewarded=np.random.choice([True, False]),
        )
        for i in range(50):
            trial = self.generator.next()
            self.generator.update(outcome)
            outcome = simulate_response(
                previous_reward=outcome.is_rewarded,
                previous_choice=outcome.is_right_choice,
                previous_left_bait=False,
                previous_right_bait=False,
                trial=trial,
            )
        self.generator.next()

        warmup = self.generator.warmup_generator
        coupled = self.generator.coupled_generator

        self.assertEqual(coupled.outcome_history, warmup.outcome_history)
        self.assertEqual(coupled.is_right_choice_history, warmup.is_right_choice_history)
        self.assertEqual(coupled.reward_history, warmup.reward_history)
        self.assertEqual(coupled.start_time, warmup.start_time)


if __name__ == "__main__":
    unittest.main()

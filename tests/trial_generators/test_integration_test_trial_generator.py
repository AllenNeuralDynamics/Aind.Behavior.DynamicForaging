import unittest

from aind_behavior_dynamic_foraging.task_logic.trial_generators import IntegrationTestTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialMetrics, TrialOutcome


class TestIntegrationTestTrialGenerator(unittest.TestCase):
    def test_returns_none_after_exhaustion(self):
        spec = IntegrationTestTrialGeneratorSpec()
        generator = spec.create_generator()
        num_trials = len(generator.trial_opts)

        for _ in range(num_trials):
            trial = generator.next()
            self.assertIsNotNone(trial)
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)

        trial = generator.next()
        self.assertIsNone(trial)

    def test_first_trial_has_both_rewards(self):
        spec = IntegrationTestTrialGeneratorSpec()
        generator = spec.create_generator()
        trial = generator.next()
        self.assertEqual(trial.p_reward_left, 1.0)
        self.assertEqual(trial.p_reward_right, 1.0)

    def test_second_trial_is_left_only(self):
        spec = IntegrationTestTrialGeneratorSpec()
        generator = spec.create_generator()

        generator.next()
        outcome = TrialOutcome(trial=Trial(), is_right_choice=True, is_rewarded=True)
        generator.update(outcome)

        trial = generator.next()
        self.assertEqual(trial.p_reward_left, 1.0)
        self.assertEqual(trial.p_reward_right, 0.0)

    def test_sequential_access(self):
        spec = IntegrationTestTrialGeneratorSpec()
        generator = spec.create_generator()
        expected_count = len(generator.trial_opts)

        trials_count = 0
        trial = generator.next()

        while trial is not None:
            trials_count += 1
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)
            trial = generator.next()

        self.assertEqual(trials_count, expected_count)

    def test_get_metrics_returns_empty_metrics(self):
        spec = IntegrationTestTrialGeneratorSpec()
        generator = spec.create_generator()

        metrics = generator.get_metrics()
        self.assertIsInstance(metrics, TrialMetrics)
        self.assertIsNone(metrics.bias)

        generator.update(TrialOutcome(trial=generator.next(), is_right_choice=True, is_rewarded=True))
        self.assertIsInstance(generator.get_metrics(), TrialMetrics)


if __name__ == "__main__":
    unittest.main()

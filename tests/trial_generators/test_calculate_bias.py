import math
import unittest

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome
from aind_behavior_dynamic_foraging.task_logic.utils.calculate_bias import calculate_bias


def make_outcomes(n, right_prob=0.5, reward_prob=0.5, **trial_kwargs) -> list[TrialOutcome]:
    outcomes = []
    for _ in range(n):
        is_right = bool(np.random.rand() < right_prob)
        is_rewarded = bool(np.random.rand() < reward_prob)
        outcomes.append(TrialOutcome(is_right_choice=is_right, is_rewarded=is_rewarded, trial=Trial(**trial_kwargs)))
    return outcomes


class TestCalculateBias(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_right_bias_is_positive(self):
        """Always choosing right with alternating rewards should produce positive bias."""
        outcomes = make_outcomes(200, 0.9, 0.5)
        bias = calculate_bias(outcomes)
        self.assertGreater(bias, 0)

    def test_left_bias_is_negative(self):
        """Always choosing left with alternating rewards should produce negative bias."""
        outcomes = make_outcomes(100, 0.1, 0.5)
        bias = calculate_bias(outcomes)
        print(bias)
        self.assertLess(bias, 0)

    def test_unbiased_near_zero(self):
        """Alternating choices should produce bias near zero."""
        outcomes = make_outcomes(100)
        bias = calculate_bias(outcomes)
        self.assertAlmostEqual(bias, 0, delta=1.0)

    def test_uniform_choices_returns_nan(self):
        """All same choice and reward should return nan since bias is undefined."""
        outcomes = make_outcomes(100, 1, 1)
        bias = calculate_bias(outcomes)
        self.assertTrue(math.isnan(bias))

    def test_ignored_trials_excluded(self):
        """Adding ignored trials should not change the result."""
        valid = make_outcomes(80)
        ignored = [TrialOutcome(is_right_choice=None, is_rewarded=False, trial=Trial()) for _ in range(20)]
        self.assertEqual(calculate_bias(valid + ignored), calculate_bias(valid))

    def test_auto_response_trials_excluded(self):
        """Adding auto response trials should not change the result."""
        valid = make_outcomes(80)
        auto = make_outcomes(20, 1, 1, is_auto_response_right=True)
        self.assertEqual(calculate_bias(valid + auto), calculate_bias(valid))

    def test_only_uses_last_200_trials(self):
        """Trials beyond the last 200 should not affect the result."""
        old = make_outcomes(100, 0)
        recent = make_outcomes(200, 0.9)
        self.assertEqual(calculate_bias(old + recent), calculate_bias(recent))

    def test_too_few_trials_returns_nan(self):
        """Fewer trials than lag should return nan since regression cannot be fit."""
        outcomes = make_outcomes(5)
        bias = calculate_bias(outcomes)
        self.assertTrue(math.isnan(bias))


if __name__ == "__main__":
    unittest.main()

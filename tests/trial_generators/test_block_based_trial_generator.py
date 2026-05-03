import logging
import unittest
from unittest.mock import patch

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    AntiBiasParameters,
    AutoWaterParameters,
    BiasThreshold,
    Block,
    BlockBasedTrialGenerator,
    BlockBasedTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial

logging.basicConfig(level=logging.DEBUG)


class ConcreteBlockBasedTrialGenerator(BlockBasedTrialGenerator):
    def _are_end_conditions_met(self) -> bool:
        return False

    def _is_block_switch_allowed(self) -> bool:
        return True

    def _generate_next_block(self) -> Block:
        return Block(p_left_reward=0, p_right_reward=0, left_length=0, right_length=0)


class ConcreteBlockBasedTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
    def create_generator(self) -> "ConcreteBlockBasedTrialGenerator":
        return ConcreteBlockBasedTrialGenerator(self)


class TestBlockBasedTrialGenerator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.spec = ConcreteBlockBasedTrialGeneratorSpec()
        self.generator = self.spec.create_generator()
        self.generator.block = Block(p_left_reward=0, p_right_reward=0, left_length=0, right_length=0)

    #### Test next ####

    def test_next_returns_trial(self):
        trial = self.generator.next()
        self.assertIsInstance(trial, Trial)

    def test_next_returns_correct_reward_probs(self):
        trial = self.generator.next()
        self.assertEqual(trial.p_reward_left, self.generator.block.p_left_reward)
        self.assertEqual(trial.p_reward_right, self.generator.block.p_right_reward)

    #### Test unbaited ####

    def test_baiting_disabled_reward_prob_unchanged(self):
        """Without baiting, reward probs should equal block probs exactly."""
        self.generator.block = Block(p_right_reward=0.8, p_left_reward=0.2, right_length=10, left_length=10)
        self.generator.is_left_baited = True
        self.generator.is_right_baited = True
        trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 0.8)
        self.assertEqual(trial.p_reward_left, 0.2)


class TestAntiBiasBlockBasedTrialGenerator(unittest.TestCase):
    def _patch_bias(self, bias_value: float) -> dict:

        return patch(
            "aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator.calculate_bias",
            return_value=bias_value,
        )

    def _make_generator(
        self,
        bias: float,
        trials_in_bias_intervention: int = 15,
        water_corrections: int = 0,
        maximum_water_corrections: int = 5,
        bias_window_length: int = 5,
        intervention_interval: int = 10,
        total_offset: float = 0.0,
        threshold: BiasThreshold = BiasThreshold(upper=0.7, lower=0.3),
    ) -> ConcreteBlockBasedTrialGenerator:
        ab = AntiBiasParameters(
            maximum_water_corrections=maximum_water_corrections,
            bias_window_length=bias_window_length,
            intervention_interval=intervention_interval,
            threshold=threshold,
        )
        spec = ConcreteBlockBasedTrialGeneratorSpec(antibias_parameters=ab)
        gen = spec.create_generator()
        gen.block = Block(p_left_reward=0.2, p_right_reward=0.8, left_length=10, right_length=10)
        gen.total_lickspout_offset = total_offset
        gen.bias = bias
        gen.trials_in_bias_intervention = trials_in_bias_intervention
        gen.water_corrections = water_corrections
        gen.is_right_choice_history = [True] * 100
        gen.reward_history = [True] * 100
        return gen

    def test_returns_false_when_antibias_disabled(self):
        """Antibias should never trigger when antibias_parameters is None."""
        spec = ConcreteBlockBasedTrialGeneratorSpec(antibias_parameters=None)
        gen = spec.create_generator()
        self.assertFalse(gen._are_antibias_conditions_met())

    def test_returns_false_before_intervention_interval(self):
        """Condition should not trigger before the intervention interval is exceeded."""

        gen = self._make_generator(bias=0.5, intervention_interval=10)
        gen.trials_in_bias_intervention = 5
        self.assertFalse(gen._are_antibias_conditions_met())

    def test_returns_false_when_bias_within_thresholds(self):
        """No intervention when bias sits between lower and upper thresholds."""
        gen = self._make_generator(
            bias=0.5,
            intervention_interval=10,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
            bias_window_length=5,
        )
        gen.trials_in_bias_intervention = 15
        gen.is_right_choice_history = [True, False] * 50
        gen.reward_history = [True] * 100

        with self._patch_bias(0.5):
            result = gen._are_antibias_conditions_met()

        self.assertFalse(result)

    def test_returns_true_when_bias_above_upper_threshold(self):
        """Intervention when bias is above threshold"""
        gen = self._make_generator(
            bias=0.9,
            intervention_interval=10,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
            bias_window_length=5,
        )
        gen.trials_in_bias_intervention = 15
        gen.is_right_choice_history = [True] * 100
        gen.reward_history = [True] * 100

        with self._patch_bias(0.9):
            result = gen._are_antibias_conditions_met()

        self.assertTrue(result)

    def test_returns_true_when_bias_below_lower_threshold(self):
        """Intervention when bias is below threshold"""
        gen = self._make_generator(
            bias=0.2,
            intervention_interval=10,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
            bias_window_length=5,
        )
        gen.trials_in_bias_intervention = 15
        gen.is_right_choice_history = [False] * 100
        gen.reward_history = [False] * 100

        with self._patch_bias(0.9):
            result = gen._are_antibias_conditions_met()

        self.assertTrue(result)

    def test_bias_stored_on_generator_after_check(self):
        """The computed bias value should be saved on the generator."""
        gen = self._make_generator(
            bias=0,
            intervention_interval=10,
            bias_window_length=5,
        )

        with self._patch_bias(0.42):
            gen._are_antibias_conditions_met()

        self.assertAlmostEqual(gen.bias, 0.42)

    def test_gives_right_water_on_left_bias(self):
        """Negative bias (left bias) → give right water."""
        gen = self._make_generator(bias=-0.9, maximum_water_corrections=5)
        is_right, delta = gen._determine_antibias_intervention()
        self.assertTrue(is_right)
        self.assertEqual(delta, 0.0)

    def test_gives_left_water_on_right_bias(self):
        """Positive bias (right bias) → give left water."""
        gen = self._make_generator(bias=0.9, maximum_water_corrections=5)
        is_right, delta = gen._determine_antibias_intervention()
        self.assertFalse(is_right)
        self.assertEqual(delta, 0.0)

    def test_water_corrections_counter_increments(self):
        gen = self._make_generator(bias=-0.9, water_corrections=2, maximum_water_corrections=5)
        gen._determine_antibias_intervention()
        self.assertEqual(gen.water_corrections, 3)

    def test_switches_to_lickspout_after_max_corrections_left_bias(self):
        """After exhausting water corrections, move lickspout right (combat left bias)."""
        gen = self._make_generator(bias=-0.9, water_corrections=5, maximum_water_corrections=5)
        is_right, delta = gen._determine_antibias_intervention()
        self.assertIsNone(is_right)
        self.assertGreater(delta, 0)

    def test_switches_to_lickspout_after_max_corrections_right_bias(self):
        """After exhausting water corrections, move lickspout left (combat right bias)."""
        gen = self._make_generator(bias=0.9, water_corrections=5, maximum_water_corrections=5)
        is_right, delta = gen._determine_antibias_intervention()
        self.assertIsNone(is_right)
        self.assertLess(delta, 0)

    def test_water_corrections_reset_after_lickspout_move(self):
        gen = self._make_generator(bias=-0.9, water_corrections=5, maximum_water_corrections=5)
        gen._determine_antibias_intervention()
        self.assertEqual(gen.water_corrections, 0)

    # #### Test lickspout centering ####

    def test_no_centering_when_offset_is_zero(self):
        """No correction when already centered, even if bias drops below lower threshold."""
        gen = self._make_generator(
            bias=0.1,
            total_offset=0.0,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
        )
        _, delta = gen._determine_antibias_intervention()
        self.assertEqual(delta, 0.0)

    def test_centering_moves_toward_zero_from_positive_offset(self):
        """Positive offset + low bias → negative delta (move back left)."""
        gen = self._make_generator(
            bias=0.1,
            total_offset=1.0,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
        )
        _, delta = gen._determine_antibias_intervention()
        self.assertLess(delta, 0)

    def test_centering_moves_toward_zero_from_negative_offset(self):
        """Negative offset + low bias → positive delta (move back right)."""
        gen = self._make_generator(
            bias=0.1,
            total_offset=-1.0,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
        )
        _, delta = gen._determine_antibias_intervention()
        self.assertGreater(delta, 0)

    def test_centering_step_capped_at_offset_magnitude(self):
        """Centering delta should not overshoot: capped at min(0.5, |offset|)."""
        gen = self._make_generator(
            bias=0.1,
            total_offset=0.2,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
        )
        _, delta = gen._determine_antibias_intervention()
        self.assertLessEqual(abs(delta), 0.2)

    def test_total_lickspout_offset_updated_after_move(self):
        """total_lickspout_offset should accumulate the delta applied."""
        gen = self._make_generator(bias=-0.9, water_corrections=5, maximum_water_corrections=5, total_offset=0.0)
        _, delta = gen._determine_antibias_intervention()
        self.assertAlmostEqual(gen.total_lickspout_offset, delta)

    #### Test next ####

    def test_next_gives_right_autowater_on_left_bias(self):
        gen = self._make_generator(bias=-0.9)
        with self._patch_bias(-0.9):
            trial = gen.next()
        self.assertIsNotNone(trial)
        self.assertTrue(trial.is_auto_response_right)

    def test_next_gives_left_autowater_on_right_bias(self):
        gen = self._make_generator(bias=0.9)
        with self._patch_bias(0.9):
            trial = gen.next()
        self.assertIsNotNone(trial)
        self.assertFalse(trial.is_auto_response_right)

    def test_next_no_antibias_when_below_interval(self):
        """No antibias effect when trials_in_bias_intervention has not exceeded interval."""
        gen = self._make_generator(bias=-0.9, trials_in_bias_intervention=5)
        trial = gen.next()
        self.assertIsNone(trial.is_auto_response_right)

    def test_next_antibias_overrides_autowater(self):
        """When both autowater and antibias conditions are met, antibias takes precedence."""
        ab = AntiBiasParameters(
            intervention_interval=10,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
            maximum_water_corrections=5,
            bias_window_length=5,
        )
        aw = AutoWaterParameters(min_ignored_trials=1, min_unrewarded_trials=1, reward_fraction=0.8)
        spec = ConcreteBlockBasedTrialGeneratorSpec(antibias_parameters=ab, autowater_parameters=aw)
        gen = spec.create_generator()
        gen.block = Block(p_left_reward=0.2, p_right_reward=0.8, left_length=10, right_length=10)
        gen.bias = -0.9
        gen.trials_in_bias_intervention = 15
        gen.is_right_choice_history = [None]  # ignored trial → autowater would also fire
        gen.reward_history = [False]

        with self._patch_bias(-0.9):
            trial = gen.next()

        # Antibias (left bias → give right water) should win
        self.assertTrue(trial.is_auto_response_right)

    def test_next_lickspout_delta_nonzero_after_corrections_exhausted(self):
        """After max water corrections, next() should produce a nonzero lickspout delta."""
        gen = self._make_generator(bias=-0.9, water_corrections=5)
        with self._patch_bias(-0.9):
            trial = gen.next()
        self.assertNotEqual(trial.lickspout_offset_delta, 0)

    def test_next_no_lickspout_delta_when_antibias_not_triggered(self):
        gen = self._make_generator(bias=-0.9, trials_in_bias_intervention=5)
        trial = gen.next()
        self.assertEqual(trial.lickspout_offset_delta, 0)


class TestBlockBaseBaitingTrialGenerator(unittest.TestCase):
    def setUp(self):
        self.spec = ConcreteBlockBasedTrialGeneratorSpec(is_baiting=True)
        self.generator = self.spec.create_generator()

    def test_baiting_sets_prob_to_1_when_baited(self):
        """If bait is held, reward prob should be 1.0 on that side."""
        self.generator.block = Block(p_right_reward=0.5, p_left_reward=0.5, right_length=10, left_length=10)
        self.generator.is_right_baited = True
        self.generator.is_left_baited = True

        trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)

    def test_baiting_accumulates_when_random_exceeds_prob(self):
        """Bait should carry over when random number exceeds reward prob."""
        self.generator.block = Block(p_right_reward=0.5, p_left_reward=0.5, right_length=10, left_length=10)
        self.generator.is_right_baited = False
        self.generator.is_left_baited = False

        with patch("numpy.random.random", return_value=np.array([0.9, 0.9])):
            trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 0.5)
        self.assertEqual(trial.p_reward_left, 0.5)

    def test_baiting_triggers_when_random_below_prob(self):
        """Bait should trigger reward prob of 1.0 when random number is below reward prob."""
        self.generator.block = Block(p_right_reward=0.5, p_left_reward=0.5, right_length=10, left_length=10)
        self.generator.is_right_baited = False
        self.generator.is_left_baited = False

        with patch("numpy.random.random", return_value=np.array([0.1, 0.1])):
            trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)


if __name__ == "__main__":
    unittest.main()

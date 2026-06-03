import logging
import unittest

from aind_behavior_dynamic_foraging.task_logic.interventions.bias_intervention import (
    BiasIntervention,
    BiasInterventionParameters,
)

logging.basicConfig(level=logging.DEBUG)


class TestBiasIntervention(unittest.TestCase):
    def test_returns_false_when_antibias_disabled(self):
        """Antibias should never trigger when bias_intervention_parameters is None."""
        bias_intervention = BiasIntervention(bias_intervention_parameters=None)
        self.assertFalse(bias_intervention.are_intervention_conditions_met(0.9))

    def test_returns_false_before_intervention_interval(self):
        """Condition should not trigger before the intervention interval is exceeded."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.trials_in_bias_intervention = 5
        self.assertFalse(bias_intervention.are_intervention_conditions_met(0.5))

    def test_returns_false_when_bias_within_thresholds(self):
        """No intervention when bias sits between lower and upper thresholds."""

        bias_intervention = BiasIntervention(BiasInterventionParameters(bias_window_length=5))
        bias_intervention.trials_in_bias_intervention = 15
        result = bias_intervention.are_intervention_conditions_met(0.4)

        self.assertFalse(result)

    def test_returns_true_when_bias_above_upper_threshold(self):
        """Intervention when bias is above threshold"""

        bias_intervention = BiasIntervention(BiasInterventionParameters(bias_window_length=5))
        bias_intervention.trials_in_bias_intervention = 15
        result = bias_intervention.are_intervention_conditions_met(0.9)

        self.assertTrue(result)

    def test_returns_true_when_bias_below_lower_threshold(self):
        """Intervention when bias is below threshold"""
        bias_intervention = BiasIntervention(BiasInterventionParameters(bias_window_length=5))
        bias_intervention.trials_in_bias_intervention = 15
        result = bias_intervention.are_intervention_conditions_met(0.2)

        self.assertTrue(result)

    def test_gives_right_water_on_left_bias(self):
        """Negative bias (left bias) → give right water."""

        bias_intervention = BiasIntervention(BiasInterventionParameters())
        is_right, delta = bias_intervention.determine_intervention(-0.9)
        self.assertTrue(is_right)
        self.assertEqual(delta, 0.0)

    def test_gives_left_water_on_right_bias(self):
        """Positive bias (right bias) → give left water."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        is_right, delta = bias_intervention.determine_intervention(0.9)
        self.assertFalse(is_right)
        self.assertEqual(delta, 0.0)

    def test_water_corrections_counter_increments(self):
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.water_corrections = 2
        bias_intervention.determine_intervention(-0.9)
        self.assertEqual(bias_intervention.water_corrections, 3)

    def test_switches_to_lickspout_after_max_corrections_left_bias(self):
        """After exhausting water corrections, move lickspout right (combat left bias)."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.water_corrections = 5
        is_right, delta = bias_intervention.determine_intervention(-0.9)
        self.assertIsNone(is_right)
        self.assertGreater(delta, 0)

    def test_switches_to_lickspout_after_max_corrections_right_bias(self):
        """After exhausting water corrections, move lickspout left (combat right bias)."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.water_corrections = 5
        is_right, delta = bias_intervention.determine_intervention(0.9)
        self.assertIsNone(is_right)
        self.assertLess(delta, 0)

    def test_water_corrections_reset_after_lickspout_move(self):
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.water_corrections = 5
        bias_intervention.determine_intervention(0.9)
        self.assertEqual(bias_intervention.water_corrections, 0)

    # #### Test lickspout centering ####

    def test_no_centering_when_offset_is_zero(self):
        """No correction when already centered, even if bias drops below lower threshold."""

        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.total_lickspout_offset = 0
        bias_intervention.water_corrections = 5
        _, delta = bias_intervention.determine_intervention(0.2)
        self.assertEqual(delta, 0.0)

    def test_centering_moves_toward_zero_from_positive_offset(self):
        """Positive offset + low bias → negative delta (move back left)."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.total_lickspout_offset = 1
        bias_intervention.water_corrections = 5
        _, delta = bias_intervention.determine_intervention(0.2)
        self.assertLess(delta, 0)

    def test_centering_moves_toward_zero_from_negative_offset(self):
        """Negative offset + low bias → positive delta (move back right)."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.total_lickspout_offset = -1
        bias_intervention.water_corrections = 5
        _, delta = bias_intervention.determine_intervention(0.2)
        self.assertGreater(delta, 0)

    def test_centering_step_capped_at_offset_magnitude(self):
        """Centering delta should not overshoot: capped at min(0.05, |offset|)."""

        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.total_lickspout_offset = 0.01
        bias_intervention.water_corrections = 5
        _, delta = bias_intervention.determine_intervention(0.2)
        self.assertLessEqual(abs(delta), 0.01)

    def test_total_lickspout_offset_updated_after_move(self):
        """total_lickspout_offset should accumulate the delta applied."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.total_lickspout_offset = 0
        bias_intervention.water_corrections = 5
        _, delta = bias_intervention.determine_intervention(0.9)
        self.assertAlmostEqual(bias_intervention.total_lickspout_offset, delta)

    def test_trials_in_bias_intervention_increments_when_no_intervention(self):
        """Counter should increment each time conditions are checked but not met."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.trials_in_bias_intervention = 0
        bias_intervention.are_intervention_conditions_met(0.5)
        self.assertEqual(bias_intervention.trials_in_bias_intervention, 1)

    def test_trials_in_bias_intervention_does_not_increment_when_triggered(self):
        """Counter should not increment when intervention conditions are met."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.trials_in_bias_intervention = 15
        bias_intervention.are_intervention_conditions_met(0.9)
        self.assertNotEqual(bias_intervention.trials_in_bias_intervention, 16)

    def test_trials_in_bias_intervention_resets_after_determine_intervention(self):
        """Counter should reset to 0 after determine_intervention is called."""
        bias_intervention = BiasIntervention(BiasInterventionParameters())
        bias_intervention.trials_in_bias_intervention = 15
        bias_intervention.determine_intervention(0.9)
        self.assertEqual(bias_intervention.trials_in_bias_intervention, 0)

    def test_trials_in_bias_intervention_does_not_increment_when_disabled(self):
        """Counter should not change when bias intervention is not configured."""
        bias_intervention = BiasIntervention(bias_intervention_parameters=None)
        bias_intervention.trials_in_bias_intervention = 0
        bias_intervention.are_intervention_conditions_met(0.9)
        self.assertEqual(bias_intervention.trials_in_bias_intervention, 0)

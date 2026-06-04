import logging
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.interventions.auto_water_intervention import (
    AutoWaterInterventionParameters,
)
from aind_behavior_dynamic_foraging.task_logic.interventions.bias_intervention import (
    BiasInterventionParameters,
    BiasThreshold,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    Block,
    BlockBasedTrialGenerator,
    BlockBasedTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome

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
        assert trial is not None
        self.assertEqual(trial.p_reward_left, self.generator.block.p_left_reward)
        self.assertEqual(trial.p_reward_right, self.generator.block.p_right_reward)


class TestBiasInterventionBlockBasedTrialGenerator(unittest.TestCase):
    def _patch_bias(self, bias_value: float) -> Any:

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
        ab = BiasInterventionParameters(
            maximum_water_corrections=maximum_water_corrections,
            bias_window_length=bias_window_length,
            intervention_interval=intervention_interval,
            threshold=threshold,
        )
        spec = ConcreteBlockBasedTrialGeneratorSpec(bias_intervention_parameters=ab)
        gen = spec.create_generator()
        gen.block = Block(p_left_reward=0.2, p_right_reward=0.8, left_length=10, right_length=10)
        gen.bias_intervention.total_lickspout_offset = total_offset
        gen.bias = bias
        gen.bias_intervention.trials_in_bias_intervention = trials_in_bias_intervention
        gen.bias_intervention.water_corrections = water_corrections
        gen.is_right_choice_history = [True] * 100
        gen.reward_history = [True] * 100
        return gen

    def test_bias_stored_on_generator_after_check(self):
        """The computed bias value should be saved on the generator."""
        gen = self._make_generator(
            bias=0,
            intervention_interval=10,
            bias_window_length=5,
        )

        with self._patch_bias(0.42):
            gen.update(TrialOutcome(is_rewarded=True, is_right_choice=True, trial=Trial()))

        self.assertAlmostEqual(gen.bias, 0.42)

    #### Test next ####

    def test_next_gives_right_auto_water_on_left_bias(self):
        gen = self._make_generator(bias=-0.9)
        trial = gen.next()
        assert trial is not None
        self.assertTrue(trial.is_auto_response_right)

    def test_next_gives_left_auto_water_on_right_bias(self):
        gen = self._make_generator(bias=0.9)
        trial = gen.next()
        assert trial is not None
        self.assertFalse(trial.is_auto_response_right)

    def test_next_no_bias_intervention_when_below_interval(self):
        """No bias intervention when trials_in_bias_intervention has not exceeded interval."""
        gen = self._make_generator(bias=-0.9, trials_in_bias_intervention=5)
        trial = gen.next()
        assert trial is not None
        self.assertIsNone(trial.is_auto_response_right)

    def test_next_bias_intervention_overrides_auto_water(self):
        """When both auto-water and bias intervention conditions are met, bias intervention takes precedence."""
        bip = BiasInterventionParameters(
            intervention_interval=10,
            threshold=BiasThreshold(upper=0.7, lower=0.3),
            maximum_water_corrections=5,
            bias_window_length=5,
        )
        aw = AutoWaterInterventionParameters(min_ignored_trials=1, min_unrewarded_trials=1, reward_fraction=0.8)
        spec = ConcreteBlockBasedTrialGeneratorSpec(
            bias_intervention_parameters=bip, auto_water_intervention_parameters=aw
        )
        gen = spec.create_generator()
        gen.block = Block(p_left_reward=0.2, p_right_reward=0.8, left_length=10, right_length=10)
        gen.bias = -0.9
        gen.bias_intervention.trials_in_bias_intervention = 15
        gen.is_right_choice_history = [None]  # ignored trial → auto_water would also fire
        gen.reward_history = [False]
        trial = gen.next()

        # bias intervention (left bias → give right water) should win
        assert trial is not None
        self.assertTrue(trial.is_auto_response_right)

    def test_next_lickspout_delta_nonzero_after_corrections_exhausted(self):
        """After max water corrections, next() should produce a nonzero lickspout delta."""
        gen = self._make_generator(bias=-0.9, water_corrections=5)
        trial = gen.next()
        assert trial is not None
        self.assertEqual(trial.lickspout_offset_delta, 0.05)

    def test_next_no_lickspout_delta_when_bias_intervention_not_triggered(self):
        gen = self._make_generator(bias=-0.9, trials_in_bias_intervention=5)
        trial = gen.next()
        assert trial is not None
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

        assert trial is not None
        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)

    def test_baiting_accumulates_when_random_exceeds_prob(self):
        """Bait should carry over when random number exceeds reward prob."""
        self.generator.block = Block(p_right_reward=0.5, p_left_reward=0.5, right_length=10, left_length=10)
        self.generator.is_right_baited = True
        self.generator.is_left_baited = True

        with patch("numpy.random.random", return_value=np.array([0.9, 0.9])):
            trial = self.generator.next()
        assert trial is not None
        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)

    def test_baiting_triggers_when_random_below_prob(self):
        """Bait should trigger reward prob of 1.0 when random number is below reward prob."""
        self.generator.block = Block(p_right_reward=0.5, p_left_reward=0.5, right_length=10, left_length=10)
        self.generator.is_right_baited = False
        self.generator.is_left_baited = False

        with patch("numpy.random.random", return_value=np.array([0.1, 0.1])):
            trial = self.generator.next()
        assert trial is not None
        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)


if __name__ == "__main__":
    unittest.main()

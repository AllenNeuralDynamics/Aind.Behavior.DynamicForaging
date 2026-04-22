import logging
import unittest
from unittest.mock import patch

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
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

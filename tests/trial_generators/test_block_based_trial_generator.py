import logging
import unittest
from unittest.mock import patch

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    Block,
    BlockBasedTrialGenerator,
    BlockBasedTrialGeneratorSpec,
    RewardProbabilityParameters,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial

logging.basicConfig(level=logging.DEBUG)


class ConcreteBlockBasedTrialGenerator(BlockBasedTrialGenerator):
    def _are_end_conditions_met(self) -> bool:
        return False


class ConcreteBlockBasedTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
    def create_generator(self) -> "ConcreteBlockBasedTrialGenerator":
        return ConcreteBlockBasedTrialGenerator(self)


class TestBlockBasedTrialGenerator(unittest.TestCase):
    def setUp(self):
        self.spec = ConcreteBlockBasedTrialGeneratorSpec()
        self.generator = self.spec.create_generator()

    #### Test generate_next_block ####

    def test_next_block_differs_from_current(self):
        current = self.generator.block
        next_block = self.generator._generate_next_block(
            reward_pairs=self.spec.reward_probability_parameters.reward_pairs,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_len=self.spec.block_len,
            current_block=current,
        )
        self.assertNotEqual(
            (next_block.p_right_reward, next_block.p_left_reward),
            (current.p_right_reward, current.p_left_reward),
        )

    def test_next_block_switches_high_reward_side(self):
        current = self.generator.block
        next_block = self.generator._generate_next_block(
            reward_pairs=self.spec.reward_probability_parameters.reward_pairs,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_len=self.spec.block_len,
            current_block=current,
        )
        current_high_is_right = current.p_right_reward > current.p_left_reward
        next_high_is_right = next_block.p_right_reward > next_block.p_left_reward
        self.assertNotEqual(current_high_is_right, next_high_is_right)

    def test_next_block_switches_high_reward_side_multiple_pairs(self):
        spec = ConcreteBlockBasedTrialGeneratorSpec(
            reward_probability_parameters=RewardProbabilityParameters(
                reward_pairs=[[8, 1], [6, 1], [3, 1]],
            )
        )
        generator = spec.create_generator()

        current = generator.block
        next_block = generator._generate_next_block(
            reward_pairs=spec.reward_probability_parameters.reward_pairs,
            base_reward_sum=spec.reward_probability_parameters.base_reward_sum,
            block_len=spec.block_len,
            current_block=current,
        )

        current_high_is_right = current.p_right_reward > current.p_left_reward
        next_high_is_right = next_block.p_right_reward > next_block.p_left_reward
        self.assertNotEqual(current_high_is_right, next_high_is_right)

    def test_next_block_never_repeats_current_multiple_pairs(self):
        spec = ConcreteBlockBasedTrialGeneratorSpec(
            reward_probability_parameters=RewardProbabilityParameters(
                reward_pairs=[[8, 1], [6, 1], [3, 1]],
            )
        )
        generator = spec.create_generator()

        current = generator.block
        for _ in range(50):
            next_block = generator._generate_next_block(
                reward_pairs=spec.reward_probability_parameters.reward_pairs,
                base_reward_sum=spec.reward_probability_parameters.base_reward_sum,
                block_len=spec.block_len,
                current_block=current,
            )
            self.assertNotEqual(
                (next_block.p_right_reward, next_block.p_left_reward),
                (current.p_right_reward, current.p_left_reward),
            )
            self.assertNotEqual(
                next_block.p_right_reward > next_block.p_left_reward,
                current.p_right_reward > current.p_left_reward,
            )
            current = next_block

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
        self.generator.block = Block(p_right_reward=0.8, p_left_reward=0.2, min_length=10)
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
        self.generator.block = Block(p_right_reward=0.5, p_left_reward=0.5, min_length=10)
        self.generator.is_right_baited = True
        self.generator.is_left_baited = True

        trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)

    def test_baiting_accumulates_when_random_exceeds_prob(self):
        """Bait should carry over when random number exceeds reward prob."""
        self.generator.block = Block(p_right_reward=0.5, p_left_reward=0.5, min_length=10)
        self.generator.is_right_baited = False
        self.generator.is_left_baited = False

        with patch("numpy.random.random", return_value=np.array([0.9, 0.9])):
            trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 0.5)
        self.assertEqual(trial.p_reward_left, 0.5)

    def test_baiting_triggers_when_random_below_prob(self):
        """Bait should trigger reward prob of 1.0 when random number is below reward prob."""
        self.generator.block = Block(p_right_reward=0.5, p_left_reward=0.5, min_length=10)
        self.generator.is_right_baited = False
        self.generator.is_left_baited = False

        with patch("numpy.random.random", return_value=np.array([0.1, 0.1])):
            trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)


if __name__ == "__main__":
    unittest.main()

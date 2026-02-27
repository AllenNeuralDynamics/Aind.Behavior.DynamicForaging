import logging
import unittest
from unittest.mock import patch

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    Block,
    BlockBasedTrialGeneratorSpec,
    RewardProbabilityParameters,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial

logging.basicConfig(level=logging.DEBUG)


class TestBlockBasedTrialGenerator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.spec = BlockBasedTrialGeneratorSpec()
        self.generator = self.spec.create_generator()

    #### Test generate_next_block ####

    def test_next_block_differs_from_current(self):
        current = self.generator.block
        next_block = self.generator.generate_next_block(
            reward_families=self.spec.reward_family,
            reward_family_index=self.spec.reward_probability_parameters.family,
            reward_pairs_n=self.spec.reward_probability_parameters.pairs_n,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_len=self.spec.block_len,
            current_block=current,
        )
        self.assertNotEqual(
            (next_block.right_reward_prob, next_block.left_reward_prob),
            (current.right_reward_prob, current.left_reward_prob),
        )

    def test_next_block_switches_high_reward_side(self):
        current = self.generator.block
        next_block = self.generator.generate_next_block(
            reward_families=self.spec.reward_family,
            reward_family_index=self.spec.reward_probability_parameters.family,
            reward_pairs_n=self.spec.reward_probability_parameters.pairs_n,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_len=self.spec.block_len,
            current_block=current,
        )
        current_high_is_right = current.right_reward_prob > current.left_reward_prob
        next_high_is_right = next_block.right_reward_prob > next_block.left_reward_prob
        self.assertNotEqual(current_high_is_right, next_high_is_right)

    def test_next_block_switches_high_reward_side_multiple_pairs(self):
        spec = BlockBasedTrialGeneratorSpec(
            reward_probability_parameters=RewardProbabilityParameters(
                family=0,  # [[8,1],[6,1],[3,1],[1,1]] - 4 pairs
                pairs_n=3,
            )
        )
        generator = self.spec.create_generator()

        current = generator.block
        next_block = generator.generate_next_block(
            reward_families=spec.reward_family,
            reward_family_index=spec.reward_probability_parameters.family,
            reward_pairs_n=spec.reward_probability_parameters.pairs_n,
            base_reward_sum=spec.reward_probability_parameters.base_reward_sum,
            block_len=spec.block_len,
            current_block=current,
        )

        current_high_is_right = current.right_reward_prob > current.left_reward_prob
        next_high_is_right = next_block.right_reward_prob > next_block.left_reward_prob
        self.assertNotEqual(current_high_is_right, next_high_is_right)

    def test_next_block_never_repeats_current_multiple_pairs(self):
        spec = BlockBasedTrialGeneratorSpec(
            reward_probability_parameters=RewardProbabilityParameters(
                family=0,
                pairs_n=3,
            )
        )
        generator = spec.create_generator()

        current = generator.block
        for _ in range(50):
            next_block = generator.generate_next_block(
                reward_families=spec.reward_family,
                reward_family_index=spec.reward_probability_parameters.family,
                reward_pairs_n=spec.reward_probability_parameters.pairs_n,
                base_reward_sum=spec.reward_probability_parameters.base_reward_sum,
                block_len=spec.block_len,
                current_block=current,
            )
            self.assertNotEqual(
                (next_block.right_reward_prob, next_block.left_reward_prob),
                (current.right_reward_prob, current.left_reward_prob),
            )
            self.assertNotEqual(
                next_block.right_reward_prob > next_block.left_reward_prob,
                current.right_reward_prob > current.left_reward_prob,
            )
            current = next_block

    #### Test next ####

    def test_next_returns_trial(self):
        trial = self.generator.next()
        self.assertIsInstance(trial, Trial)

    def test_next_returns_correct_reward_probs(self):
        trial = self.generator.next()
        self.assertEqual(trial.p_reward_left, self.generator.block.left_reward_prob)
        self.assertEqual(trial.p_reward_right, self.generator.block.right_reward_prob)

    ### test unbaited ###

    def test_baiting_disabled_reward_prob_unchanged(self):
        """Without baiting, reward probs should equal block probs exactly."""
        self.generator.block = Block(right_reward_prob=0.8, left_reward_prob=0.2, min_length=10)
        self.generator.is_left_baited = True
        self.generator.is_right_baited = True
        trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 0.8)
        self.assertEqual(trial.p_reward_left, 0.2)


class TestBlockBaseBaitingTrialGenerator(unittest.TestCase):
    ### test baiting ###

    def setUp(self):
        self.spec = BlockBasedTrialGeneratorSpec(is_baiting=True)
        self.generator = self.spec.create_generator()

    def test_baiting_sets_prob_to_1_when_baited(self):
        """If bait is held, reward prob should be 1.0 on that side."""
        self.generator.block = Block(right_reward_prob=0.5, left_reward_prob=0.5, min_length=10)
        self.generator.is_right_baited = True
        self.generator.is_left_baited = True

        trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)

    def test_baiting_accumulates_when_random_exceeds_prob(self):
        """Bait should carry over when random number exceeds reward prob."""
        self.generator.block = Block(right_reward_prob=0.5, left_reward_prob=0.5, min_length=10)
        self.generator.is_right_baited = False
        self.generator.is_left_baited = False

        # force random numbers above reward prob so bait does not trigger from RNG
        with patch("numpy.random.random", return_value=np.array([0.9, 0.9])):
            trial = self.generator.next()

        # reward prob should remain unchanged since bait was not set and RNG didn't trigger
        self.assertEqual(trial.p_reward_right, 0.5)
        self.assertEqual(trial.p_reward_left, 0.5)

    def test_baiting_triggers_when_random_below_prob(self):
        """Bait should trigger reward prob of 1.0 when random number is below reward prob."""
        self.generator.block = Block(right_reward_prob=0.5, left_reward_prob=0.5, min_length=10)
        self.generator.is_right_baited = False
        self.generator.is_left_baited = False

        # force random numbers below reward prob so bait triggers from RNG
        with patch("numpy.random.random", return_value=np.array([0.1, 0.1])):
            trial = self.generator.next()

        self.assertEqual(trial.p_reward_right, 1.0)
        self.assertEqual(trial.p_reward_left, 1.0)


if __name__ == "__main__":
    unittest.main()

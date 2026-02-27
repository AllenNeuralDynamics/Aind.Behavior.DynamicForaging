import logging
import unittest

from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import BlockBasedTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    RewardProbabilityParameters,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome

logging.basicConfig(level=logging.DEBUG)


class TestCoupledTrialGenerator(unittest.TestCase):
    
    def setUp(self):
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
            block_len_distribution=self.spec.block_len_distribution,
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
            block_len_distribution=self.spec.block_len_distribution,
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
            block_len_distribution=spec.block_len_distribution,
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
                block_len_distribution=spec.block_len_distribution,
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


if __name__ == "__main__":
    unittest.main()

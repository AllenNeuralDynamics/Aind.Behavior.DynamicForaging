import logging
import unittest

from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.base_coupled_trial_generator import (
    BaseCoupledTrialGenerator,
    BaseCoupledTrialGeneratorSpec,
    RewardProbabilityParameters,
)

logging.basicConfig(level=logging.DEBUG)


class ConcreteBlockBasedTrialGenerator(BaseCoupledTrialGenerator):
    def _are_end_conditions_met(self) -> bool:
        return False

    def _is_block_switch_allowed(self) -> bool:
        return True


class ConcreteBlockBasedTrialGeneratorSpec(BaseCoupledTrialGeneratorSpec):
    def create_generator(self) -> "ConcreteBlockBasedTrialGenerator":
        return ConcreteBlockBasedTrialGenerator(self)


class TestBaseCoupledTrialGenerator(unittest.TestCase):
    def setUp(self):
        self.spec = ConcreteBlockBasedTrialGeneratorSpec()
        self.generator = self.spec.create_generator()

    #### Test generate_next_block ####

    def test_next_block_differs_from_current(self):
        current = self.generator.block
        next_block = self.generator._generate_next_block(
            reward_pairs=self.spec.reward_probability_parameters.reward_pairs,
            base_reward_sum=self.spec.reward_probability_parameters.base_reward_sum,
            block_length=self.spec.block_length,
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
            block_length=self.spec.block_length,
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
            block_length=spec.block_length,
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
                block_length=spec.block_length,
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

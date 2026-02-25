import logging
import unittest

from aind_behavior_dynamic_foraging.task_logic.trial_generators import CoupledTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generator import (
    RewardProbabilityParameters,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome

logging.basicConfig(level=logging.DEBUG)


class TestCoupledTrialGenerator(unittest.TestCase):
    ##### Tests is_behavior_stable #####

    def test_behavior_stable_end(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        beh_params = generator.spec.behavior_stability_parameters
        right_prob = generator.block.right_reward_prob
        left_prob = generator.block.left_reward_prob
        kernel_size = generator.spec.kernel_size
        min_stable = beh_params.min_consecutive_stable_trials

        high_reward_is_right = right_prob > left_prob

        beh_params.behavior_evaluation_mode = "end"

        # stable at end: wrong side early, correct side at end
        choices = [not high_reward_is_right] * 10 + [high_reward_is_right] * (min_stable + kernel_size - 1)
        self.assertTrue(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_not_stable_end(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        beh_params = generator.spec.behavior_stability_parameters
        right_prob = generator.block.right_reward_prob
        left_prob = generator.block.left_reward_prob
        kernel_size = generator.spec.kernel_size
        min_stable = beh_params.min_consecutive_stable_trials

        high_reward_is_right = right_prob > left_prob

        beh_params.behavior_evaluation_mode = "end"

        # unstable at end: correct side early, wrong side at end
        choices = [high_reward_is_right] * 10 + [not high_reward_is_right] * (min_stable + kernel_size - 1)
        self.assertFalse(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_stable_anytime(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        beh_params = generator.spec.behavior_stability_parameters
        right_prob = generator.block.right_reward_prob
        left_prob = generator.block.left_reward_prob
        kernel_size = generator.spec.kernel_size
        min_stable = beh_params.min_consecutive_stable_trials

        high_reward_is_right = right_prob > left_prob

        beh_params.behavior_evaluation_mode = "anytime"

        # stable run early, then drifts off — should still pass
        choices = [high_reward_is_right] * (min_stable + kernel_size - 1) + [not high_reward_is_right] * 10
        self.assertTrue(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

        # stable at end: wrong side early, correct side at end
        choices = [not high_reward_is_right] * 10 + [high_reward_is_right] * (min_stable + kernel_size - 1)
        self.assertTrue(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_alteranating_choices_behavior_not_stable(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        # IMPORTANT: Need to force right probability lower than left since
        # the stability threshold lower bound is anchored to left_prob + delta.
        # With very asymmetric probabilities (e.g. left=0.08) the threshold can be
        # permissive enough that an alternating animal (choice fraction ~0.5) is considered
        # stable. This is inherited behavior from the original implementation https://github.com/AllenNeuralDynamics/dynamic-foraging-task/blob/653293091179fa284c22c6dccff4f0bd49848b1e/src/foraging_gui/MyFunctions.py#L639
        # is this right?
        beh_params = generator.spec.behavior_stability_parameters
        left_prob = 0.7111111111111111
        right_prob = 0.08888888888888889
        kernel_size = generator.spec.kernel_size

        # never stable: alternating throughout
        choices = [True, False] * 15

        beh_params.behavior_evaluation_mode = "anytime"
        self.assertFalse(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

        beh_params.behavior_evaluation_mode = "end"
        self.assertFalse(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_alteranating_choices_behavior_stable(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        # IMPORTANT: Force right probability higher than left since
        # the stability threshold lower bound is anchored to left_prob + delta.
        # With very asymmetric probabilities (e.g. left=0.08) the threshold can be
        # permissive enough that an alternating animal (choice fraction ~0.5) is considered
        # stable. This is inherited behavior from the original implementation https://github.com/AllenNeuralDynamics/dynamic-foraging-task/blob/653293091179fa284c22c6dccff4f0bd49848b1e/src/foraging_gui/MyFunctions.py#L639
        # is this right?
        beh_params = generator.spec.behavior_stability_parameters
        right_prob = 0.7111111111111111
        left_prob = 0.08888888888888889
        kernel_size = generator.spec.kernel_size

        # never stable: alternating throughout
        choices = [True, False] * 15

        beh_params.behavior_evaluation_mode = "anytime"
        self.assertTrue(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

        beh_params.behavior_evaluation_mode = "end"
        self.assertTrue(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_stable_equal_reward_prob(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        beh_params = generator.spec.behavior_stability_parameters
        right_prob = 0.5
        left_prob = 0.5
        kernel_size = generator.spec.kernel_size

        choices = [True] * 15
        self.assertTrue(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_stable_choice_len_less_than_kernel(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        beh_params = generator.spec.behavior_stability_parameters
        right_prob = generator.block.right_reward_prob
        left_prob = generator.block.left_reward_prob
        kernel_size = generator.spec.kernel_size

        choices = [True]
        self.assertTrue(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_stable_no_beh_stability_params(self):
        spec = CoupledTrialGeneratorSpec(behavior_stability_parameters=None)
        generator = spec.create_generator()

        beh_params = generator.spec.behavior_stability_parameters
        right_prob = generator.block.right_reward_prob
        left_prob = generator.block.left_reward_prob
        kernel_size = generator.spec.kernel_size

        choices = [True]
        self.assertTrue(
            generator.is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    #### Test is_block_switch_allowed ####

    def test_block_switch_all_conditions_met_switches(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        generator.block.right_reward_prob = 0.8
        generator.block.left_reward_prob = 0.2
        generator.block.min_length = 20
        generator.trials_in_block = 20

        result = generator.is_block_switch_allowed(
            trials_in_block=generator.trials_in_block,
            min_block_reward=1,
            block_left_rewards=0,
            block_right_rewards=5,
            choice_history=[True] * 20,
            right_reward_prob=generator.block.right_reward_prob,
            left_reward_prob=generator.block.left_reward_prob,
            beh_stability_params=generator.spec.behavior_stability_parameters,
            block_length=generator.block.min_length,
            kernel_size=generator.spec.kernel_size,
        )
        self.assertTrue(result)

    def test_block_switch_block_length_not_reached(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        generator.block.right_reward_prob = 0.8
        generator.block.left_reward_prob = 0.2
        generator.block.min_length = 20

        result = generator.is_block_switch_allowed(
            trials_in_block=10,  # below min_length
            min_block_reward=1,
            block_left_rewards=0,
            block_right_rewards=5,
            choice_history=[True] * 10,
            right_reward_prob=generator.block.right_reward_prob,
            left_reward_prob=generator.block.left_reward_prob,
            beh_stability_params=generator.spec.behavior_stability_parameters,
            block_length=generator.block.min_length,
            kernel_size=generator.spec.kernel_size,
        )
        self.assertFalse(result)

    def test_block_switch_reward_not_met(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        generator.block.right_reward_prob = 0.8
        generator.block.left_reward_prob = 0.2
        generator.block.min_length = 20

        result = generator.is_block_switch_allowed(
            trials_in_block=20,
            min_block_reward=5,
            block_left_rewards=0,
            block_right_rewards=0,  # no rewards
            choice_history=[True] * 20,
            right_reward_prob=generator.block.right_reward_prob,
            left_reward_prob=generator.block.left_reward_prob,
            beh_stability_params=generator.spec.behavior_stability_parameters,
            block_length=generator.block.min_length,
            kernel_size=generator.spec.kernel_size,
        )
        self.assertFalse(result)

    def test_block_switch_behavior_not_stable(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        generator.block.right_reward_prob = 0.8
        generator.block.left_reward_prob = 0.2
        generator.block.min_length = 20

        result = generator.is_block_switch_allowed(
            trials_in_block=20,
            min_block_reward=1,
            block_left_rewards=5,
            block_right_rewards=0,
            choice_history=[False] * 20,  # always choosing low-reward side
            right_reward_prob=generator.block.right_reward_prob,
            left_reward_prob=generator.block.left_reward_prob,
            beh_stability_params=generator.spec.behavior_stability_parameters,
            block_length=generator.block.min_length,
            kernel_size=generator.spec.kernel_size,
        )
        self.assertFalse(result)

    #### Test generate_next_block ####

    def test_next_block_differs_from_current(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        current = generator.block
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

    def test_next_block_switches_high_reward_side(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

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

    def test_next_block_switches_high_reward_side_multiple_pairs(self):
        spec = CoupledTrialGeneratorSpec(
            reward_probability_parameters=RewardProbabilityParameters(
                family=0,  # [[8,1],[6,1],[3,1],[1,1]] - 4 pairs
                pairs_n=3,
            )
        )
        generator = spec.create_generator()

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
        spec = CoupledTrialGeneratorSpec(
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

    #### Test update ####

    def _make_outcome(self, is_right_choice, is_rewarded):
        return TrialOutcome(trial=Trial(), is_right_choice=is_right_choice, is_rewarded=is_rewarded)

    def test_update_appends_to_history(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        generator.update(self._make_outcome(True, True))
        self.assertEqual(len(generator.is_right_choice_history), 1)
        self.assertEqual(len(generator.reward_history), 1)

    def test_update_ignored_trial_extends_block_length(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        original_length = generator.block.min_length
        generator.update(self._make_outcome(None, False))
        self.assertEqual(generator.block.min_length, original_length + 1)

    def test_update_non_ignored_trial_does_not_extend_block(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        original_length = generator.block.min_length
        generator.update(self._make_outcome(True, True))
        self.assertEqual(generator.block.min_length, original_length)

    def test_update_block_switches_after_conditions_met(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        generator.block.right_reward_prob = 0.8
        generator.block.left_reward_prob = 0.2
        generator.block.min_length = 5
        generator.trials_in_block = 0

        initial_block = generator.block

        min_stable = generator.spec.behavior_stability_parameters.min_consecutive_stable_trials
        kernel_size = generator.spec.kernel_size
        for _ in range(min_stable + kernel_size - 1):
            generator.update(self._make_outcome(True, True))

        self.assertIsNot(generator.block, initial_block)

    def test_update_block_does_not_switch_before_min_length(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        generator.block.right_reward_prob = 0.8
        generator.block.left_reward_prob = 0.2
        generator.block.min_length = 100
        generator.trials_in_block = 0

        initial_block = generator.block

        for _ in range(5):
            generator.update(self._make_outcome(True, True))

        self.assertIs(generator.block, initial_block)

    #### Test next ####

    def test_next_returns_trial(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        trial = generator.next()
        self.assertIsInstance(trial, Trial)

    def test_next_returns_correct_reward_probs(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        trial = generator.next()
        self.assertEqual(trial.p_reward_left, generator.block.left_reward_prob)
        self.assertEqual(trial.p_reward_right, generator.block.right_reward_prob)

    def test_next_returns_none_after_max_trials(self):
        spec = CoupledTrialGeneratorSpec()
        generator = spec.create_generator()

        # exhaust the trial limit
        generator.is_right_choice_history = [True] * (spec.trial_generation_end_parameters.max_trial + 1)
        # bypass min_time
        generator.start_time = generator.start_time - spec.trial_generation_end_parameters.min_time

        trial = generator.next()
        self.assertIsNone(trial)


if __name__ == "__main__":
    unittest.main()

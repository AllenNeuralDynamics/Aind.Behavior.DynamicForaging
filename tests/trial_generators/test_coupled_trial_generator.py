import logging
import unittest
from datetime import timedelta

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators import CoupledTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome

from .util import simulate_response

logging.basicConfig(level=logging.DEBUG)


class TestCoupledTrialGenerator(unittest.TestCase):
    def setUp(self):
        self.spec = CoupledTrialGeneratorSpec()
        self.generator = self.spec.create_generator()

    def test_session(self):
        """Simulates a full experimental session to verify generator stability."""

        trial = Trial()
        outcome = TrialOutcome(
            trial=trial,
            is_right_choice=np.random.choice([True, False, None]),
            is_rewarded=np.random.choice([True, False]),
        )
        for i in range(500):
            trial = self.generator.next()
            self.generator.update(outcome)
            outcome = simulate_response(
                previous_reward=outcome.is_rewarded,
                previous_choice=outcome.is_right_choice,
                previous_left_bait=False,
                previous_right_bait=False,
            )

        if not trial:
            return

    ##### Tests _is_behavior_stable #####

    def test_behavior_stable_end(self):
        beh_params = self.generator.spec.behavior_stability_parameters
        right_prob = self.generator.block.p_right_reward
        left_prob = self.generator.block.p_left_reward
        kernel_size = self.generator.spec.kernel_size
        min_stable = beh_params.min_consecutive_stable_trials

        high_reward_is_right = right_prob > left_prob
        beh_params.behavior_evaluation_mode = "end"

        choices = [not high_reward_is_right] * 10 + [high_reward_is_right] * (min_stable + kernel_size - 1)
        self.assertTrue(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_not_stable_end(self):
        beh_params = self.generator.spec.behavior_stability_parameters
        right_prob = self.generator.block.p_right_reward
        left_prob = self.generator.block.p_left_reward
        kernel_size = self.generator.spec.kernel_size
        min_stable = beh_params.min_consecutive_stable_trials

        high_reward_is_right = right_prob > left_prob
        beh_params.behavior_evaluation_mode = "end"

        choices = [high_reward_is_right] * 10 + [not high_reward_is_right] * (min_stable + kernel_size - 1)
        self.assertFalse(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_stable_anytime(self):
        beh_params = self.generator.spec.behavior_stability_parameters
        right_prob = self.generator.block.p_right_reward
        left_prob = self.generator.block.p_left_reward
        kernel_size = self.generator.spec.kernel_size
        min_stable = beh_params.min_consecutive_stable_trials

        high_reward_is_right = right_prob > left_prob
        beh_params.behavior_evaluation_mode = "anytime"

        # stable run early, then drifts off — should still pass
        choices = [high_reward_is_right] * (min_stable + kernel_size - 1) + [not high_reward_is_right] * 10
        self.assertTrue(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

        # stable at end: wrong side early, correct side at end
        choices = [not high_reward_is_right] * 10 + [high_reward_is_right] * (min_stable + kernel_size - 1)
        self.assertTrue(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_alternating_choices_behavior_not_stable(self):
        beh_params = self.generator.spec.behavior_stability_parameters
        left_prob = 0.7111111111111111
        right_prob = 0.08888888888888889
        kernel_size = self.generator.spec.kernel_size

        choices = [True, False] * 15

        beh_params.behavior_evaluation_mode = "anytime"
        self.assertFalse(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

        beh_params.behavior_evaluation_mode = "end"
        self.assertFalse(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_alternating_choices_behavior_stable(self):
        beh_params = self.generator.spec.behavior_stability_parameters
        right_prob = 0.7111111111111111
        left_prob = 0.08888888888888889
        kernel_size = self.generator.spec.kernel_size

        choices = [True, False] * 15

        beh_params.behavior_evaluation_mode = "anytime"
        self.assertTrue(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

        beh_params.behavior_evaluation_mode = "end"
        self.assertTrue(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_stable_equal_reward_prob(self):
        beh_params = self.generator.spec.behavior_stability_parameters
        right_prob = 0.5
        left_prob = 0.5
        kernel_size = self.generator.spec.kernel_size

        choices = [True] * 15
        self.assertTrue(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_stable_choice_len_less_than_kernel(self):
        beh_params = self.generator.spec.behavior_stability_parameters
        right_prob = self.generator.block.p_right_reward
        left_prob = self.generator.block.p_left_reward
        kernel_size = self.generator.spec.kernel_size

        choices = [True]
        self.assertTrue(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    def test_behavior_stable_no_beh_stability_params(self):
        spec = CoupledTrialGeneratorSpec(behavior_stability_parameters=None)
        generator = spec.create_generator()

        beh_params = generator.spec.behavior_stability_parameters
        right_prob = generator.block.p_right_reward
        left_prob = generator.block.p_left_reward
        kernel_size = generator.spec.kernel_size

        choices = [True]
        self.assertTrue(
            self.generator._is_behavior_stable(choices, right_prob, left_prob, beh_params, len(choices), kernel_size)
        )

    #### Test _is_block_switch_allowed ####

    def test_block_switch_all_conditions_met_switches(self):
        self.generator.block.p_right_reward = 0.8
        self.generator.block.p_left_reward = 0.2
        self.generator.block.right_length = 20
        self.generator.trials_in_block = 20
        self.generator.reward_history = [True] * 5
        self.generator.is_right_choice_history = [True] * 20
        self.generator.spec.min_block_reward = 1

        result = self.generator._is_block_switch_allowed()
        self.assertTrue(result)

    def test_block_switch_block_length_not_reached(self):
        self.generator.block.p_right_reward = 0.8
        self.generator.block.p_left_reward = 0.2
        self.generator.block.right_length = 20
        self.generator.reward_history = [True] * 5
        self.generator.is_right_choice_history = [True] * 10
        self.generator.trials_in_block = 10
        self.generator.spec.min_block_reward = 1

        result = self.generator._is_block_switch_allowed()
        self.assertFalse(result)

    def test_block_switch_reward_not_met(self):
        self.generator.block.p_right_reward = 0.8
        self.generator.block.p_left_reward = 0.2
        self.generator.block.right_length = 20
        self.generator.reward_history = []  # no rewards
        self.generator.is_right_choice_history = [True] * 20
        self.generator.trials_in_block = 20
        self.generator.spec.min_block_reward = 5

        result = self.generator._is_block_switch_allowed()
        self.assertFalse(result)

    def test_block_switch_behavior_not_stable(self):
        self.generator.block.p_right_reward = 0.8
        self.generator.block.p_left_reward = 0.2
        self.generator.block.right_length = 20
        self.generator.reward_history = [True] * 5
        self.generator.is_right_choice_history = [False] * 20
        self.generator.trials_in_block = 20
        self.generator.spec.min_block_reward = 1

        result = self.generator._is_block_switch_allowed()
        self.assertFalse(result)

    #### Test update ####

    def _make_outcome(self, is_right_choice, is_rewarded):
        return TrialOutcome(trial=Trial(), is_right_choice=is_right_choice, is_rewarded=is_rewarded)

    def test_string_trial_outcome(self):
        string_outcome = self._make_outcome(True, True).model_dump_json()
        self.generator.update(string_outcome)

    def test_update_appends_to_history(self):
        self.generator.update(self._make_outcome(True, True))
        self.assertEqual(len(self.generator.is_right_choice_history), 1)
        self.assertEqual(len(self.generator.reward_history), 1)

    def test_update_ignored_trial_extends_block_length(self):
        original_length = self.generator.block.right_length
        self.generator.update(self._make_outcome(None, False))
        self.assertEqual(self.generator.block.right_length, original_length + 1)

    def test_update_non_ignored_trial_does_not_extend_block(self):
        original_length = self.generator.block.right_length
        self.generator.update(self._make_outcome(True, True))
        self.assertEqual(self.generator.block.right_length, original_length)

    def test_update_block_switches_after_conditions_met(self):
        self.generator.block.p_right_reward = 0.8
        self.generator.block.p_left_reward = 0.2
        self.generator.block.right_length = 5
        self.generator.trials_in_block = 0

        initial_block = self.generator.block

        min_stable = self.generator.spec.behavior_stability_parameters.min_consecutive_stable_trials
        kernel_size = self.generator.spec.kernel_size
        self.generator.reward_history = [True] * 10

        for _ in range(min_stable + kernel_size - 1):
            self.generator.update(self._make_outcome(True, True))

        self.assertIsNot(self.generator.block, initial_block)

    def test_update_block_does_not_switch_before_right_length(self):
        self.generator.block.p_right_reward = 0.8
        self.generator.block.p_left_reward = 0.2
        self.generator.block.right_length = 100
        self.generator.trials_in_block = 0

        initial_block = self.generator.block

        for _ in range(5):
            self.generator.update(self._make_outcome(True, True))

        self.assertIs(self.generator.block, initial_block)

    #### Test next ####

    def test_next_returns_none_after_max_trials(self):
        self.generator.is_right_choice_history = [True] * (self.spec.trial_generation_end_parameters.max_trial + 1)
        self.generator.start_time = self.generator.start_time - timedelta(
            self.spec.trial_generation_end_parameters.min_time
        )

        trial = self.generator.next()
        self.assertIsNone(trial)

    def test_baiting_disabled_bait_state_never_changes(self):
        self.generator.is_right_baited = True
        self.generator.is_left_baited = True
        self.generator.update(self._make_outcome(is_right_choice=True, is_rewarded=True))
        self.assertTrue(self.generator.is_right_baited)
        self.assertTrue(self.generator.is_left_baited)


class TestCoupledBaitingTrialGenerator(unittest.TestCase):
    def setUp(self):
        self.spec = CoupledTrialGeneratorSpec(is_baiting=True)
        self.generator = self.spec.create_generator()

    def _make_outcome(self, is_right_choice, is_rewarded):
        return TrialOutcome(trial=Trial(), is_right_choice=is_right_choice, is_rewarded=is_rewarded)

    def test_right_bait_resets_on_right_choice(self):
        self.generator.is_right_baited = True
        self.generator.update(self._make_outcome(is_right_choice=True, is_rewarded=True))
        self.assertFalse(self.generator.is_right_baited)

    def test_left_bait_resets_on_left_choice(self):
        self.generator.is_left_baited = True
        self.generator.update(self._make_outcome(is_right_choice=False, is_rewarded=True))
        self.assertFalse(self.generator.is_left_baited)

    def test_right_bait_preserved_on_left_choice(self):
        self.generator.is_right_baited = True
        self.generator.update(self._make_outcome(is_right_choice=False, is_rewarded=False))
        self.assertTrue(self.generator.is_right_baited)

    def test_left_bait_preserved_on_right_choice(self):
        self.generator.is_left_baited = True
        self.generator.update(self._make_outcome(is_right_choice=True, is_rewarded=True))
        self.assertTrue(self.generator.is_left_baited)

    def test_bait_not_reset_on_ignored_trial(self):
        self.generator.is_right_baited = True
        self.generator.is_left_baited = True
        self.generator.update(self._make_outcome(is_right_choice=None, is_rewarded=False))
        self.assertTrue(self.generator.is_right_baited)
        self.assertTrue(self.generator.is_left_baited)


if __name__ == "__main__":
    unittest.main()

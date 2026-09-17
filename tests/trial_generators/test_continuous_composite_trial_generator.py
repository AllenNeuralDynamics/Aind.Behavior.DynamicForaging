import unittest
from typing import Literal

from pydantic import Field

from aind_behavior_dynamic_foraging.task_logic.trial_generators import ContinuousCompositeTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_generators._base import (
    BaseTrialGeneratorSpecModel,
    ITrialGenerator,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.continuous_composite_trial_generator import (
    ContinuousCompositeTrialGenerator,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_trial_generator import (
    CoupledTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_warmup_trial_generator import (
    CoupledWarmupTrialGenerationEndConditions,
    CoupledWarmupTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome


class MockTrialGeneratorSpec(BaseTrialGeneratorSpecModel):
    type: Literal["MockTrialGenerator"] = "MockTrialGenerator"
    num_trials: int = Field(default=10, description="Number of trials to generate")

    def create_generator(self) -> "MockTrialGenerator":
        return MockTrialGenerator(self)


class MockTrialGenerator(ITrialGenerator):
    def __init__(self, spec: MockTrialGeneratorSpec) -> None:
        self.spec = spec
        self.trial_count = 0
        self.outcome_history = []
        self.is_right_choice_history = []
        self.reward_history = []

    def next(self) -> Trial | None:
        if self.trial_count >= self.spec.num_trials:
            return None
        return Trial(p_reward_left=1.0, p_reward_right=1.0)

    def update(self, outcome: TrialOutcome | str) -> None:
        if isinstance(outcome, str):
            outcome = TrialOutcome.model_validate_json(outcome)
        self.trial_count += 1
        self.outcome_history.append(outcome)
        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)


class TestContinuousCompositeTrialGenerator(unittest.TestCase):
    def test_accepts_multiple_generators(self):
        """The generic composite should accept any number of generator stages."""
        spec = ContinuousCompositeTrialGeneratorSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=5),
                MockTrialGeneratorSpec(num_trials=5),
                MockTrialGeneratorSpec(num_trials=5),
            ]
        )
        generator = spec.create_generator()
        self.assertIsInstance(generator, ContinuousCompositeTrialGenerator)
        self.assertEqual(len(generator._generators), 3)

    def test_concatenate_multiple_stages(self):
        """Trials should concatenate across all stages in order."""
        stage_trials = 5
        spec = ContinuousCompositeTrialGeneratorSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=stage_trials),
                MockTrialGeneratorSpec(num_trials=stage_trials),
            ]
        )
        generator = spec.create_generator()

        trials_count = 0
        trial = generator.next()
        while trial is not None:
            trials_count += 1
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)
            trial = generator.next()

        self.assertEqual(trials_count, 2 * stage_trials)

    def test_state_transfer_with_block_based_generators(self):
        """Session-level state should carry over between block-based stages."""
        warmup_spec = CoupledWarmupTrialGeneratorSpec(
            trial_generation_end_parameters=CoupledWarmupTrialGenerationEndConditions(
                min_trial=5,
                max_choice_bias=0.5,
                min_response_rate=0.5,
            ),
        )
        main_spec = CoupledTrialGeneratorSpec()

        composite_spec = ContinuousCompositeTrialGeneratorSpec(generators=[warmup_spec, main_spec])
        composite = composite_spec.create_generator()

        total_trial_count = 0
        trial = composite.next()
        while trial is not None and total_trial_count < 20:
            total_trial_count += 1
            outcome = TrialOutcome(
                trial=trial,
                is_right_choice=total_trial_count % 2 == 0,
                is_rewarded=total_trial_count % 3 == 0,
            )
            composite.update(outcome)
            trial = composite.next()

        warmup_generator = composite._generators[0]
        main_generator = composite._generators[1]

        self.assertEqual(
            main_generator.outcome_history[: len(warmup_generator.outcome_history)],
            warmup_generator.outcome_history,
        )
        self.assertEqual(
            main_generator.is_right_choice_history[: len(warmup_generator.is_right_choice_history)],
            warmup_generator.is_right_choice_history,
        )
        self.assertEqual(
            main_generator.reward_history[: len(warmup_generator.reward_history)],
            warmup_generator.reward_history,
        )
        self.assertEqual(main_generator.start_time, warmup_generator.start_time)
        self.assertFalse(main_generator.is_left_baited)
        self.assertFalse(main_generator.is_right_baited)

    def test_main_stage_works_after_transition(self):
        """After the stage boundary, the next generator should continue producing trials."""
        warmup_spec = CoupledWarmupTrialGeneratorSpec(
            trial_generation_end_parameters=CoupledWarmupTrialGenerationEndConditions(
                min_trial=5,
                max_choice_bias=0.5,
                min_response_rate=0.5,
            ),
        )
        main_spec = CoupledTrialGeneratorSpec()

        composite_spec = ContinuousCompositeTrialGeneratorSpec(generators=[warmup_spec, main_spec])
        composite = composite_spec.create_generator()

        trial_count = 0
        trial = composite.next()
        while trial is not None and trial_count < 20:
            trial_count += 1
            outcome = TrialOutcome(
                trial=trial,
                is_right_choice=True,
                is_rewarded=False,
            )
            composite.update(outcome)
            trial = composite.next()

        warmup_generator = composite._generators[0]
        main_generator = composite._generators[1]

        self.assertGreater(len(warmup_generator.outcome_history), 0)
        self.assertGreater(len(main_generator.outcome_history), len(warmup_generator.outcome_history))

    def test_updates_transition_correctly(self):
        """Updates should continue to target the active stage after each transition."""
        spec = ContinuousCompositeTrialGeneratorSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=3),
                MockTrialGeneratorSpec(num_trials=3),
                MockTrialGeneratorSpec(num_trials=3),
            ]
        )
        generator = spec.create_generator()

        for _ in range(3):
            trial = generator.next()
            self.assertIsNotNone(trial)
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)

        first_gen = generator._generators[0]
        self.assertEqual(first_gen.trial_count, 3)

        trial = generator.next()
        self.assertIsNotNone(trial)
        outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
        generator.update(outcome)

        second_gen = generator._generators[1]
        self.assertEqual(second_gen.trial_count, 1)


if __name__ == "__main__":
    unittest.main()

import unittest
from typing import Literal

from pydantic import Field

from aind_behavior_dynamic_foraging.task_logic.trial_generators import CompositeWarmupTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_generators._base import (
    BaseTrialGeneratorSpecModel,
    ITrialGenerator,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.composite_warmup_trial_generator import (
    CompositeWarmupTrialGenerator,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_warmup_trial_generator import (
    CoupledWarmupTrialGeneratorSpec,
    CoupledWarmupTrialGenerationEndConditions,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_trial_generator import (
    CoupledTrialGeneratorSpec,
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
        # Add attributes to simulate BlockBasedTrialGenerator interface
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


class TestCompositeWarmupTrialGenerator(unittest.TestCase):
    def test_requires_exactly_two_generators(self):
        """Test that CompositeWarmupTrialGenerator validates exactly 2 generators."""
        # Valid with 2 generators
        spec = CompositeWarmupTrialGeneratorSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=5),
                MockTrialGeneratorSpec(num_trials=5),
            ]
        )
        generator = spec.create_generator()
        self.assertIsInstance(generator, CompositeWarmupTrialGenerator)

    def test_concatenate_warmup_and_main(self):
        """Test that warmup and main stage trials concatenate properly."""
        warmup_trials = 5
        main_trials = 5
        spec = CompositeWarmupTrialGeneratorSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=warmup_trials),
                MockTrialGeneratorSpec(num_trials=main_trials),
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

        self.assertEqual(trials_count, warmup_trials + main_trials)

    def test_state_transfer_with_block_based_generators(self):
        """Test that session state is transferred between block-based generators.

        This test uses real CoupledWarmupTrialGenerator and CoupledTrialGenerator
        to verify that state transfer occurs at the warmup to main boundary.
        """
        # Create warmup and main stage generators with short durations for testing
        warmup_spec = CoupledWarmupTrialGeneratorSpec(
            trial_generation_end_parameters=CoupledWarmupTrialGenerationEndConditions(
                min_trial=5,
                max_choice_bias=0.5,  # Be lenient with bias threshold
                min_response_rate=0.5,  # Be lenient with response rate
            ),
        )
        main_spec = CoupledTrialGeneratorSpec()

        composite_spec = CompositeWarmupTrialGeneratorSpec(generators=[warmup_spec, main_spec])
        composite = composite_spec.create_generator()

        # Run through warmup stage and into main stage
        total_trial_count = 0
        trial = composite.next()
        while trial is not None and total_trial_count < 20:  # Safety limit
            total_trial_count += 1
            outcome = TrialOutcome(
                trial=trial,
                is_right_choice=total_trial_count % 2 == 0,
                is_rewarded=total_trial_count % 3 == 0,
            )
            composite.update(outcome)
            trial = composite.next()

        # Get the generators
        warmup_generator = composite._generators[0]
        main_generator = composite._generators[1]

        # Verify warmup generator has outcomes
        warmup_outcomes_count = len(warmup_generator.outcome_history)
        self.assertGreater(warmup_outcomes_count, 0, "Warmup generator should have outcomes")

        # Verify main generator has transferred history
        main_outcomes_count = len(main_generator.outcome_history)
        self.assertGreaterEqual(
            main_outcomes_count,
            warmup_outcomes_count,
            "Main generator should have at least the warmup outcomes transferred",
        )

    def test_main_stage_works_after_warmup_transition(self):
        """Test that main stage can continue generating trials after warmup transition.

        Verifies that after transferring state from warmup to main, the composite
        generator can continue to generate additional trials from the main stage.
        """
        warmup_spec = CoupledWarmupTrialGeneratorSpec(
            trial_generation_end_parameters=CoupledWarmupTrialGenerationEndConditions(
                min_trial=5,
                max_choice_bias=0.5,  # Be lenient with bias threshold
                min_response_rate=0.5,  # Be lenient with response rate
            ),
        )
        main_spec = CoupledTrialGeneratorSpec()

        composite_spec = CompositeWarmupTrialGeneratorSpec(generators=[warmup_spec, main_spec])
        composite = composite_spec.create_generator()

        # Run through all trials: warmup stage and into main stage
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

        # Get the generators
        warmup_generator = composite._generators[0]
        main_generator = composite._generators[1]

        warmup_outcomes_count = len(warmup_generator.outcome_history)
        self.assertGreater(warmup_outcomes_count, 0, "Warmup should have generated trials")

        # Verify main generator has transferred warmup outcomes plus additional main stage outcomes
        final_main_outcomes = len(main_generator.outcome_history)
        self.assertGreater(
            final_main_outcomes,
            warmup_outcomes_count,
            "Main stage should have transferred warmup outcomes plus new outcomes",
        )

    def test_updates_transition_correctly(self):
        """Test that updates go to the correct generator during transition."""
        spec = CompositeWarmupTrialGeneratorSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=3),
                MockTrialGeneratorSpec(num_trials=3),
            ]
        )
        generator = spec.create_generator()

        # Run through warmup
        for i in range(3):
            trial = generator.next()
            self.assertIsNotNone(trial)
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)

        # Verify warmup generator received 3 updates
        warmup_gen = generator._generators[0]
        self.assertEqual(warmup_gen.trial_count, 3)

        # Next trial should be from main generator
        trial = generator.next()
        self.assertIsNotNone(trial)
        outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
        generator.update(outcome)

        # Verify main generator received the update
        main_gen = generator._generators[1]
        self.assertEqual(main_gen.trial_count, 1)


if __name__ == "__main__":
    unittest.main()

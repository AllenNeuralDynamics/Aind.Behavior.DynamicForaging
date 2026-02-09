import unittest
from typing import Literal

from pydantic import Field, ValidationError

from aind_behavior_dynamic_foraging.task_logic.trial_generators import TrialGeneratorCompositeSpec
from aind_behavior_dynamic_foraging.task_logic.trial_generators._base import (
    BaseTrialGeneratorSpecModel,
    ITrialGenerator,
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

    def next(self) -> Trial | None:
        if self.trial_count >= self.spec.num_trials:
            return None
        return Trial(p_reward_left=1.0, p_reward_right=1.0)

    def update(self, outcome: TrialOutcome) -> None:
        self.trial_count += 1


class TestTrialGeneratorComposite(unittest.TestCase):
    def test_empty_generators_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            TrialGeneratorCompositeSpec(generators=[])

    def test_single_generator(self):
        num_trials = 10
        spec = TrialGeneratorCompositeSpec(generators=[MockTrialGeneratorSpec(num_trials=num_trials)])
        generator = spec.create_generator()

        trials_count = 0
        trial = generator.next()
        while trial is not None:
            trials_count += 1
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)
            trial = generator.next()

        self.assertEqual(trials_count, num_trials)

    def test_concatenate_n_generators(self):
        n = 5
        num_trials = 10
        spec = TrialGeneratorCompositeSpec(generators=[MockTrialGeneratorSpec(num_trials=num_trials) for _ in range(n)])
        generator = spec.create_generator()

        trials_count = 0
        trial = generator.next()
        while trial is not None:
            trials_count += 1
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)
            trial = generator.next()

        self.assertEqual(trials_count, n * num_trials)

    def test_first_trial_from_first_generator(self):
        spec = TrialGeneratorCompositeSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=5),
                MockTrialGeneratorSpec(num_trials=5),
            ]
        )
        generator = spec.create_generator()

        trial = generator.next()
        self.assertEqual(trial.p_reward_left, 1.0)
        self.assertEqual(trial.p_reward_right, 1.0)

    def test_transition_between_generators(self):
        num_trials = 5
        spec = TrialGeneratorCompositeSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=num_trials),
                MockTrialGeneratorSpec(num_trials=num_trials),
            ]
        )
        generator = spec.create_generator()

        for i in range(num_trials):
            trial = generator.next()
            self.assertIsNotNone(trial)
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)

        trial = generator.next()
        self.assertIsNotNone(trial)
        self.assertEqual(trial.p_reward_left, 1.0)
        self.assertEqual(trial.p_reward_right, 1.0)

    def test_updates_current_generator_only(self):
        spec = TrialGeneratorCompositeSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=5),
                MockTrialGeneratorSpec(num_trials=5),
            ]
        )
        generator = spec.create_generator()

        trial = generator.next()
        outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
        generator.update(outcome)

        self.assertIsInstance(generator._generators[0], MockTrialGenerator)
        self.assertIsInstance(generator._generators[1], MockTrialGenerator)
        self.assertEqual(generator._generators[0].trial_count, 1)
        self.assertEqual(generator._generators[1].trial_count, 0)

    def test_no_update_after_exhaustion(self):
        num_trials = 5
        spec = TrialGeneratorCompositeSpec(generators=[MockTrialGeneratorSpec(num_trials=num_trials)])
        generator = spec.create_generator()

        for _ in range(num_trials):
            trial = generator.next()
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)

        trial = generator.next()
        if trial is not None:
            outcome = TrialOutcome(trial=trial, is_right_choice=None, is_rewarded=False)
            generator.update(outcome)
        self.assertIsInstance(generator._generators[0], MockTrialGenerator)
        self.assertEqual(generator._generators[0].trial_count, num_trials)

    def test_nested_composite_generators(self):
        num_trials = 3
        inner_composite_1 = TrialGeneratorCompositeSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=num_trials),
                MockTrialGeneratorSpec(num_trials=num_trials),
            ]
        )
        inner_composite_2 = TrialGeneratorCompositeSpec(
            generators=[
                MockTrialGeneratorSpec(num_trials=num_trials),
                MockTrialGeneratorSpec(num_trials=num_trials),
            ]
        )

        outer_composite = TrialGeneratorCompositeSpec(generators=[inner_composite_1, inner_composite_2])
        self.assertIsInstance(outer_composite.generators[0], TrialGeneratorCompositeSpec)
        self.assertIsInstance(outer_composite.generators[1], TrialGeneratorCompositeSpec)

        generator = outer_composite.create_generator()

        trials_count = 0
        trial = generator.next()
        while trial is not None:
            trials_count += 1
            outcome = TrialOutcome(trial=trial, is_right_choice=True, is_rewarded=True)
            generator.update(outcome)
            trial = generator.next()

        self.assertEqual(trials_count, 4 * num_trials)


if __name__ == "__main__":
    unittest.main()

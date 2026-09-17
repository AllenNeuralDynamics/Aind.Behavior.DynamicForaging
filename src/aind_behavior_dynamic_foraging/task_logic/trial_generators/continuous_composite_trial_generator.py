from typing import Literal, TypeVar

from pydantic import Field, SerializeAsAny

from aind_behavior_dynamic_foraging.task_logic.trial_generators._base import (
    BaseTrialGeneratorSpecModel,
    ITrialGenerator,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    BlockBasedTrialGenerator,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.composite_trial_generator import (
    TrialGeneratorComposite,
    TrialGeneratorCompositeSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial
from aind_behavior_dynamic_foraging.task_logic.utils import calculate_bias

_TSpec = TypeVar("_TSpec", bound=BaseTrialGeneratorSpecModel, covariant=True)


class ContinuousCompositeTrialGeneratorSpec(TrialGeneratorCompositeSpec[_TSpec]):
    """Specification for a composite generator that preserves session state between stages."""

    type: Literal["ContinuousCompositeTrialGenerator"] = "ContinuousCompositeTrialGenerator"

    generators: list[SerializeAsAny[_TSpec]] = Field(
        description="List of block-based generator specifications to concatenate. "
        "When one generator returns None, the next one is activated while preserving "
        "session-state history across the stage boundary.",
        min_length=1,
    )

    def create_generator(self) -> "ContinuousCompositeTrialGenerator":
        return ContinuousCompositeTrialGenerator(self)


class ContinuousCompositeTrialGenerator(TrialGeneratorComposite):
    """Composite generator for chaining block-based generators while preserving session history."""

    def __init__(self, spec: ContinuousCompositeTrialGeneratorSpec[BaseTrialGeneratorSpecModel]) -> None:
        super().__init__(spec)

    def next(self) -> Trial | None:
        while self._current_index < len(self._generators):
            trial = self._generators[self._current_index].next()
            if trial is not None:
                return trial

            if self._current_index < len(self._generators) - 1:
                previous_generator = self._generators[self._current_index]
                self._current_index += 1
                next_generator = self._generators[self._current_index]
                self._transfer_session_state(previous_generator, next_generator)
            else:
                self._current_index += 1

        return None

    @staticmethod
    def _transfer_session_state(previous_generator: ITrialGenerator, next_generator: ITrialGenerator) -> None:
        if not isinstance(previous_generator, BlockBasedTrialGenerator) or not isinstance(
            next_generator, BlockBasedTrialGenerator
        ):
            return

        next_generator.outcome_history = previous_generator.outcome_history.copy()
        next_generator.is_right_choice_history = previous_generator.is_right_choice_history.copy()
        next_generator.reward_history = previous_generator.reward_history.copy()
        next_generator.start_time = previous_generator.start_time

        next_generator.bias = calculate_bias(
            outcomes=next_generator.outcome_history,
            outcome_window_length=(
                200
                if not next_generator.spec.bias_intervention_parameters
                else next_generator.spec.bias_intervention_parameters.bias_window_length
            ),
        )

        next_generator.is_left_baited = False
        next_generator.is_right_baited = False

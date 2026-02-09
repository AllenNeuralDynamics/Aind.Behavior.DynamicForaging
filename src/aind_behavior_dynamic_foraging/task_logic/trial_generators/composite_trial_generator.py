from typing import Generic, Literal, TypeVar

from pydantic import Field, SerializeAsAny

from ..trial_models import Trial, TrialOutcome
from ._base import BaseTrialGeneratorSpecModel, ITrialGenerator

_TSpec = TypeVar("_TSpec", bound=BaseTrialGeneratorSpecModel)


class TrialGeneratorCompositeSpec(BaseTrialGeneratorSpecModel, Generic[_TSpec]):
    """Specification for a composite trial generator that concatenates multiple generators."""

    type: Literal["TrialGeneratorComposite"] = "TrialGeneratorComposite"

    generators: list[SerializeAsAny[_TSpec]] = Field(
        description="List of trial generator specifications to concatenate. "
        "When one generator returns None, the next one is concatenated.",
        min_length=1,
    )

    def create_generator(self) -> "TrialGeneratorComposite":
        return TrialGeneratorComposite(self)


class TrialGeneratorComposite(ITrialGenerator):
    """
    A composite trial generator that concatenates multiple trial generators.

    When the current generator's next() method returns None, the composite
    automatically moves to the next generator in the list.
    """

    def __init__(self, spec: TrialGeneratorCompositeSpec[BaseTrialGeneratorSpecModel]) -> None:
        """
        Initialize the composite trial generator.

        :param spec: The specification containing the list of generator specs
        """
        self._spec = spec
        self._generators: list[ITrialGenerator] = [gen_spec.create_generator() for gen_spec in spec.generators]
        self._current_index = 0

    def next(self) -> Trial | None:
        """
        Get the next trial from the current generator.

        If the current generator returns None, automatically advance to the next
        generator in the list. Returns None only when all generators are exhausted.

        :return: The next Trial, or None if all generators are exhausted
        """
        while self._current_index < len(self._generators):
            trial = self._generators[self._current_index].next()

            if trial is not None:
                return trial

            # Current generator returned None, move to next
            self._current_index += 1

        # Finally, return None if all generators got consumed
        return None

    def update(self, outcome: TrialOutcome) -> None:
        """
        Update the current active generator with the trial outcome.

        :param outcome: The outcome of the last trial
        """
        if self._current_index < len(self._generators):
            self._generators[self._current_index].update(outcome)

import abc
from typing import Protocol

from pydantic import BaseModel

from ..trial_models import Trial, TrialOutcome


class BaseTrialGeneratorSpecModel(BaseModel, abc.ABC):
    """Base model for trial generator specifications."""

    type: str

    @abc.abstractmethod
    def create_generator(self) -> "ITrialGenerator":
        """Create a trial generator instance from the specification."""


class ITrialGenerator(Protocol):
    """Interface for trial generators."""

    def next(self) -> Trial | None:
        """Return the next trial to run. Return None if there are no more trials to run."""

    def update(self, outcome: TrialOutcome) -> None:
        """Update the trial generator with the outcome of the previous trial."""

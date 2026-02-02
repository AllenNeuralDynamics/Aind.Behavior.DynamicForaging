import abc
from typing import Protocol

from pydantic import BaseModel

from ..trial_models import Trial, TrialOutcome


class _BaseTrialGeneratorSpecModel(BaseModel, abc.ABC):
    type: str

    @abc.abstractmethod
    def create_generator(self) -> "_ITrialGenerator":
        pass


class _ITrialGenerator(Protocol):
    def next(self) -> Trial | None: ...

    def update(self, outcome: TrialOutcome) -> None: ...

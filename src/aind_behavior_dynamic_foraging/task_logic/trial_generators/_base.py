import abc
from typing import Protocol

from pydantic import BaseModel

from ..trial_models import Trial, TrialOutcome


class BaseTrialGeneratorSpecModel(BaseModel, abc.ABC):
    type: str

    @abc.abstractmethod
    def create_generator(self) -> "ITrialGenerator":
        pass


class ITrialGenerator(Protocol):
    def next(self) -> Trial | None: ...

    def update(self, outcome: TrialOutcome) -> None: ...

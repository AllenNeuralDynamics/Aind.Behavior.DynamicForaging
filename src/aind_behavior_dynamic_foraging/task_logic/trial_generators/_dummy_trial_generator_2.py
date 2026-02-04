from typing import Literal

from ..trial_models import Trial, TrialOutcome
from ._base import ITrialGenerator, _BaseTrialGeneratorSpecModel


class DummyTrialGeneratorModel2(_BaseTrialGeneratorSpecModel):
    type: Literal["DummyTrialGenerator2"] = "DummyTrialGenerator2"

    def create_generator(self) -> "DummyTrialGenerator2":
        return DummyTrialGenerator2(self)


class DummyTrialGenerator2(ITrialGenerator):
    def __init__(self, spec: DummyTrialGeneratorModel2) -> None:
        self._spec = spec
        self._idx = 0

    def next(self) -> Trial | None:
        if self._idx >= 10:
            return None
        else:
            return Trial(p_reward_left=self._idx % 2 == 0, has_reward_right=self._idx % 2 == 1)

    def update(self, outcome: TrialOutcome) -> None:
        self._idx += 1

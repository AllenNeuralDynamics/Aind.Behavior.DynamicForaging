import logging
from typing import Literal

from pydantic import Field

from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_trial_generator import (
    CoupledTrialGenerator,
    CoupledTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.warmup_trial_generator import (
    WarmupTrialGenerator,
    WarmupTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialMetrics, TrialOutcome

from .._base import BaseTrialGeneratorSpecModel, ITrialGenerator

logger = logging.getLogger(__name__)


class CoupledWarmupTrialGeneratorSpec(BaseTrialGeneratorSpecModel):
    type: Literal["CoupledWarmupTrialGenerator"] = "CoupledWarmupTrialGenerator"

    warmup_generator_spec: WarmupTrialGeneratorSpec = Field(
        default=WarmupTrialGeneratorSpec(), description="Specification for the warmup trial generator."
    )
    coupled_generator_spec: CoupledTrialGeneratorSpec = Field(
        default=CoupledTrialGeneratorSpec(), description="Specification for the coupled trial generator."
    )

    def create_generator(self) -> "CoupledWarmupTrialGenerator":
        return CoupledWarmupTrialGenerator(self)


class CoupledWarmupTrialGenerator(ITrialGenerator):
    spec: CoupledWarmupTrialGeneratorSpec

    def __init__(self, spec: CoupledWarmupTrialGeneratorSpec) -> None:
        self.spec = spec
        self.warmup_generator = WarmupTrialGenerator(spec.warmup_generator_spec)
        self.coupled_generator = CoupledTrialGenerator(spec.coupled_generator_spec)
        self._active_generator = self.warmup_generator

    def next(self):

        trial = self._active_generator.next()
        if trial is None and isinstance(self._active_generator, WarmupTrialGenerator):
            # copy session state from warmup generator to coupled generator
            self.coupled_generator.outcome_history = self.warmup_generator.outcome_history.copy()
            self.coupled_generator.is_right_choice_history = self.warmup_generator.is_right_choice_history.copy()
            self.coupled_generator.reward_history = self.warmup_generator.reward_history.copy()
            self.coupled_generator.start_time = self.warmup_generator.start_time

            self._active_generator = self.coupled_generator
            trial = self._active_generator.next()
        return trial

    def update(self, outcome: TrialOutcome | str) -> None:
        """
        Update the current active generator with the trial outcome.

        :param outcome: The outcome of the last trial
        """

        self._active_generator.update(outcome)

    def get_metrics(self) -> TrialMetrics:
        """Return metrics at current state of the trial generator."""

        return self._active_generator.get_metrics()

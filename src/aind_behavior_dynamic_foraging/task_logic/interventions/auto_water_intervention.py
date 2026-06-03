import logging
from typing import Optional

from pydantic import BaseModel, Field

from aind_behavior_dynamic_foraging.task_logic.interventions.base_intervention import BaseIntervention

logger = logging.getLogger(__name__)


class AutoWaterInterventionParameters(BaseModel):
    min_ignored_trials: int = Field(
        default=3, ge=0, description="Minimum consecutive ignored trials before auto water is triggered."
    )
    min_unrewarded_trials: int = Field(
        default=3, ge=0, description="Minimum consecutive unrewarded trials before auto water is triggered."
    )
    reward_fraction: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Fraction of full reward volume delivered during auto water (0=none, 1=full).",
    )  # TODO: Not implemented yet


class AutoWaterIntervention(BaseIntervention):
    """Manages auto water interventions during a task."""

    def __init__(
        self,
        auto_water_intervention_parameters: Optional[AutoWaterInterventionParameters] = None,
    ):

        self.parameters = auto_water_intervention_parameters

    def are_intervention_conditions_met(
        self, is_right_choice_history: list[bool | None], reward_history: list[bool]
    ) -> bool:
        """Checks whether autowater should be given.

        Returns:
            True if autowater conditions are met, False otherwise.
        """

        if self.parameters is None:
            logger.debug("Auto-water not configured.")
            return False

        min_ignore = self.parameters.min_ignored_trials
        min_unreward = self.parameters.min_unrewarded_trials

        is_ignored = [choice is None for choice in is_right_choice_history]
        if len(is_ignored) > min_ignore and all(is_ignored[-min_ignore:]):
            logger.debug("Past %s trials ignored." % min_ignore)
            return True

        is_unrewarded = [not reward for reward in reward_history]
        if len(is_unrewarded) > min_unreward and all(is_unrewarded[-min_unreward:]):
            logger.debug("Past %s trials unrewarded." % min_unreward)
            return True

        return False

    def determine_intervention(self, p_reward_right: float, p_reward_left: float) -> bool:
        """Determine auto-water interventions to perform: give water on higher probability side

        Returns:
           boolean indicating is_auto_response_right. True indicates auto-water given to right; False, left.
        """

        return True if p_reward_right > p_reward_left else False

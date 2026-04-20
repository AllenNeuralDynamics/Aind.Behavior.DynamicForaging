import logging
from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from ..trial_models import TrialOutcome
from .block_based_trial_generator import (
    BlockBasedTrialGenerator,
    BlockBasedTrialGeneratorSpec,
)

logger = logging.getLogger(__name__)

BlockBehaviorEvaluationMode = Literal[
    "end",  # behavior stable at end of block to allow switching
    "anytime",  # behavior stable anytime in block to allow switching
]


class UncoupledTrialGenerationEndConditions(BaseModel):
    """Defines the conditions under which a foraging session should terminate."""

    ignore_win: int = Field(default=30, ge=0, description="Number of recent trials to check for ignored responses.")
    ignore_ratio_threshold: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Maximum fraction of ignored trials within the window before the session is ended.",
    )
    max_trial: int = Field(default=1000, ge=0, description="Maximum number of trials allowed in a session.")
    max_time: float = Field(default=75 * 60, description="Maximum session duration (sec).")
    min_time: float = Field(default=30 * 60, description="Minimum session duration (sec)")

class UncoupledTrialGeneratorSpec(BlockBasedTrialGeneratorSpec):
    type: Literal["UncoupledTrialGenerator"] = "UncoupledTrialGenerator"

    trial_generation_end_parameters: UncoupledTrialGenerationEndConditions = Field(
        default=UncoupledTrialGenerationEndConditions(),
        description="Conditions to end trial generation.",
        validate_default=True,
    )

    def create_generator(self) -> "UncoupledTrialGenerator":
        return UncoupledTrialGenerator(self)


class UncoupledTrialGenerator(BlockBasedTrialGenerator):
    """Trial generator for a Uncoupled block-based dynamic foraging task.

    Attributes:
        spec: The UncoupledTrialGeneratorSpec defining task parameters.
        start_time: Timestamp recorded at initialization, used to track elapsed
            session time.
    """

    spec: UncoupledTrialGeneratorSpec

    def __init__(self, spec: UncoupledTrialGeneratorSpec) -> None:
        """Initializes the generator and records the session start time.

        Args:
            spec: The UncoupledTrialGeneratorSpec defining task parameters.
        """

        super().__init__(spec)
        self.start_time = datetime.now()


    def _are_end_conditions_met(self) -> bool:
        """Checks whether the session should end.

        Evaluates three termination conditions: excessive ignored trials after
        minimum session time, maximum session time exceeded, and maximum trial
        count exceeded.

        Returns:
            True if any end condition is met, False otherwise.
        """

        end_conditions = self.spec.trial_generation_end_parameters
        choice_history = self.is_right_choice_history

        time_elapsed = datetime.now() - self.start_time
        frac = end_conditions.ignore_ratio_threshold
        win = end_conditions.ignore_win

        if (
            time_elapsed > timedelta(seconds=end_conditions.min_time)
            and choice_history[-win:].count(None) >= frac * win
        ):
            logger.debug("Minimum time and ignored trial count exceeded.")
            return True

        if timedelta(seconds=end_conditions.max_time) < time_elapsed:
            logger.debug("Maximum session time exceeded.")
            return True

        if end_conditions.max_trial < len(choice_history):
            logger.debug("Maximum trial count exceeded.")
            return True

        return False

    def update(self, outcome: TrialOutcome | str) -> None:
        """Updates generator state from the previous trial outcome and switches block if criteria are met.

        Records choice and reward history, manages baiting state, optionally extends
        the block on no response, and triggers a block switch if all switch criteria
        are satisfied.

        Args:
            outcome: The TrialOutcome from the most recently completed trial.
        """

        logger.info(f"Updating usncoupled trial generator with trial outcome of {outcome}")

        if isinstance(outcome, str):
            outcome = TrialOutcome.model_validate_json(outcome)

        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)
        self.trials_in_block += 1

        if self.spec.is_baiting:
            if outcome.is_right_choice:
                logger.debug("Resesting right bait.")
                self.is_right_baited = False
            elif outcome.is_right_choice is False:
                logger.debug("Resesting left bait.")
                self.is_left_baited = False

        switch_block = self._is_block_switch_allowed()

        if switch_block:
            logger.info("Switching block.")
            self.trials_in_block = 0
            self.block = self._generate_next_block()
            self.block_history.append(self.block)

   
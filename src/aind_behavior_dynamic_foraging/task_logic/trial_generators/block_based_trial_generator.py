import datetime
import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
from aind_behavior_services.task.distributions import (
    Distribution,
    ExponentialDistribution,
    ExponentialDistributionParameters,
    ScalingParameters,
    TruncationParameters,
)
from aind_behavior_services.task.distributions_utils import draw_sample
from pydantic import BaseModel, Field

from aind_behavior_dynamic_foraging.task_logic.interventions.bias_intervention import (
    BiasIntervention,
    BiasInterventionParameters,
)
from aind_behavior_dynamic_foraging.task_logic.utils import calculate_bias, calculate_foraging_efficiency

from ..trial_models import Metadata, RewardSize, Trial, TrialMetrics
from ._base import BaseTrialGeneratorSpecModel, ITrialGenerator, TrialOutcome

logger = logging.getLogger(__name__)


class BlockBasedTrialMetadata(BaseModel):
    """Metadata for block based trial. These fields will NOT be used by the task engine."""

    time_elapsed: Optional[float] = Field(
        default=None, description="Time elapsed in session at start of trial (in minutes)."
    )
    time_remaining: Optional[float] = Field(
        default=None, description="Time remaining in session at start of trial (in minutes)."
    )
    current_trial: Optional[int] = Field(default=None, description="Current trial number in session.")
    responses: Optional[int] = Field(default=None, description="Number of responses made in session.")
    ignored: Optional[int] = Field(default=None, description="Number of ignored trials in session.")
    earned_water: Optional[float] = Field(default=None, description="Total water earned in session (in mL).")
    total_water: Optional[float] = Field(default=None, description="Total water delivered in session (in mL).")
    foraging_efficiency: Optional[float] = Field(
        default=None, description="Foraging efficiency in session (earned water / total water)."
    )
    is_autowater: bool = Field(default=False, description="Flag indicating if autowater is given for trial.")
    is_bias_water_intervention: bool = Field(
        default=False, description="Flag indicating if bias water intervention is given for trial."
    )
    is_bias_stage_intervention: bool = Field(
        default=False, description="Flag indicating if bias stage intervention is given for trial."
    )
    is_right_baited: bool = Field(default=False, description="Flag indicating if right side is baited.")
    is_left_baited: bool = Field(default=False, description="Flag indicating if left side is baited.")


class AutoWaterParameters(BaseModel):
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
    )


class Block(BaseModel):
    p_right_reward: float = Field(ge=0, le=1, description="Reward probability for right side during block.")
    p_left_reward: float = Field(ge=0, le=1, description="Reward probability for left side during block.")
    right_length: int = Field(ge=0, description="Minimum number of trials in block.")
    left_length: int = Field(ge=0, description="Minimum number of trials in block.")


class BlockBasedTrialGeneratorSpec(BaseTrialGeneratorSpecModel):
    type: Literal["BlockBasedTrialGenerator"] = "BlockBasedTrialGenerator"

    reward_size: RewardSize = Field(
        default=RewardSize(left=2, right=2), description="Parameters describing reward size."
    )

    quiescent_duration: Distribution = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1),
            truncation_parameters=TruncationParameters(min=0, max=1),
        ),
        description="Distribution describing the quiescence period before trial starts (in seconds). Each lick resets the timer.",
    )

    response_duration: float = Field(default=1.0, ge=0, description="Duration after go cue for animal response.")

    reward_consumption_duration: float = Field(
        default=3.0,
        ge=0,
        description="Duration of reward consumption before transition to ITI (in seconds).",
    )

    inter_trial_interval_duration: Distribution = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 2),
            truncation_parameters=TruncationParameters(max=8),
            scaling_parameters=ScalingParameters(offset=1),
        ),
        description="Distribution describing the inter-trial interval (in seconds).",
    )

    block_length: Distribution = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 20),
            truncation_parameters=TruncationParameters(min=20, max=60),
        ),
        description="Distribution describing block length. Block length is floored making the upper bound exclusive.",
    )

    autowater_parameters: Optional[AutoWaterParameters] = Field(
        default=AutoWaterParameters(),
        validate_default=True,
        description="Autowater settings. If set, free water is delivered when the animal exceeds the ignored or unrewarded trial thresholds.",
    )

    bias_intervention_parameters: Optional[BiasInterventionParameters] = Field(
        default=BiasInterventionParameters(),
        validate_default=True,
        description="Antibias settings. If set, trial generator will give water and move lickspouts to combat bias.",
    )

    is_baiting: bool = Field(default=False, description="Whether uncollected rewards carry over to the next trial.")


class BlockBasedTrialGenerator(ITrialGenerator, ABC):
    """Abstract trial generator for block-based dynamic foraging tasks.

    Manages block transitions, baiting logic, and trial generation. Subclasses
    must implement `_are_end_conditions_met` to define session termination logic.

    Attributes:
        spec: The specification used to configure this generator.
        is_right_choice_history: Record of whether each trial was a right choice.
            None indicates no choice was made (e.g. missed trial).
        reward_history: Record of whether each trial resulted in a reward.
        is_left_baited: Whether the left port currently has a baited reward.
        is_right_baited: Whether the right port currently has a baited reward.
        trials_in_bias_intervention: trials elapsed since last bias intervention
        water_corrections: number of water corrections applied to combat bias
        bias: bias of session. Negative values correspond to left bias, positive right.
    """

    def __init__(self, spec: BlockBasedTrialGeneratorSpec) -> None:
        """Initializes the generator and generates the first block.

        Args:
            spec: The BlockBasedTrialGenerator defining task parameters.
        """

        self.spec = spec
        self.start_time = datetime.datetime.now()
        self.outcome_history: list[TrialOutcome] = []
        self.is_right_choice_history: list[bool | None] = []
        self.reward_history: list[bool] = []
        self.is_left_baited: bool = False
        self.is_right_baited: bool = False
        self.block: Block

        self.bias: float = np.nan
        self.bias_intervention = BiasIntervention(self.spec.bias_intervention_parameters)

    def update(self, outcome: TrialOutcome | str):
        """Updates generator state from the previous trial outcome. Records choice and reward history and manages baiting state.
        Args:
            outcome: The TrialOutcome from the most recently completed trial.
        """
        logger.debug("Updating trial generator.")
        if isinstance(outcome, str):
            outcome = TrialOutcome.model_validate_json(outcome)

        self.outcome_history.append(outcome)
        self.is_right_choice_history.append(outcome.is_right_choice)
        self.reward_history.append(outcome.is_rewarded)

        if self.spec.is_baiting:
            if outcome.is_right_choice:
                logger.debug("Resesting right bait.")
                self.is_right_baited = False
            elif outcome.is_right_choice is False:
                logger.debug("Resesting left bait.")
                self.is_left_baited = False
            else:
                # trial ignored so current baiting state retained
                pass

        self.bias = calculate_bias(outcomes=self.outcome_history)

    def next(self) -> Trial | None:
        """Generates the next trial in the session.

        Checks end conditions, samples timing parameters, and applies baiting
        logic if enabled. Returns None if the session should end.

        Returns:
            The next Trial, or None if end conditions are met.
        """
        logger.info("Generating next trial.")

        # check end conditions
        if self._are_end_conditions_met():
            logger.info("Trial generator end conditions met.")
            return

        # determine iti and quiescent period duration
        iti = draw_sample(self.spec.inter_trial_interval_duration)
        quiescent = draw_sample(self.spec.quiescent_duration)

        # determine baiting
        if self.spec.is_baiting:
            random_numbers = np.random.random(2)

            self.is_left_baited = self.block.p_left_reward > random_numbers[0] or self.is_left_baited
            logger.debug("Left baited: %s" % self.is_left_baited)

            self.is_right_baited = self.block.p_right_reward > random_numbers[1] or self.is_right_baited
            logger.debug("Right baited: %s" % self.is_right_baited)

        is_auto_reward_right = None
        reward_fraction = 1

        # determine autowater
        if is_autowater := self._are_autowater_conditions_met():
            is_auto_reward_right = True if self.block.p_right_reward > self.block.p_left_reward else False
            reward_fraction = self.spec.autowater_parameters.reward_fraction
            logger.debug("Delivering autowater: is_auto_reward_right = %s" % is_auto_reward_right)

        # determine bias correction. Overrides autowater
        lickspout_offset_delta = 0
        if is_bias_intervention := self.bias_intervention.are_antibias_conditions_met(
            self.bias, len(self.outcome_history)
        ):
            is_auto_reward_right, lickspout_offset_delta = self.bias_intervention.determine_antibias_intervention(
                self.bias
            )

            reward_fraction = (
                1 if is_auto_reward_right is None else self.spec.bias_intervention_parameters.reward_fraction
            )
            logger.debug(
                "Performing bias intervention: is_auto_reward_right = %s, lickspout_offset_delta = %s."
                % (is_auto_reward_right, lickspout_offset_delta)
            )

        trial = Trial(
            p_reward_left=1 if (self.is_left_baited or is_auto_reward_right is False) else self.block.p_left_reward,
            p_reward_right=1 if (self.is_right_baited or is_auto_reward_right) else self.block.p_right_reward,
            reward_consumption_duration=self.spec.reward_consumption_duration,
            response_deadline_duration=self.spec.response_duration,
            quiescence_period_duration=quiescent,
            quiescence_period_refractory_duration=self.spec.inter_trial_interval_duration,
            inter_trial_interval_duration=iti,
            lickspout_offset_delta=lickspout_offset_delta,
            is_auto_reward_right=is_auto_reward_right,
            reward_size=RewardSize(
                left=self.spec.reward_size.left * reward_fraction, right=self.spec.reward_size.right * reward_fraction
            ),
            metadata=Metadata(
                p_reward_left=self.block.p_left_reward,
                p_reward_right=self.block.p_right_reward,
            ),
        )
        extra_metadata = BlockBasedTrialMetadata(
            is_autowater=is_autowater and not is_bias_intervention,
            is_bias_water_intervention=is_bias_intervention and is_auto_reward_right is not None,
            is_bias_stage_intervention=is_bias_intervention and lickspout_offset_delta != 0,
            is_right_baited=self.is_right_baited,
            is_left_baited=self.is_left_baited,
        )
        trial.metadata.extra = self._add_extra_metadata(extra_metadata)
        return trial

    def _add_extra_metadata(self, extra_metadata: BlockBasedTrialMetadata) -> BlockBasedTrialMetadata:
        """Adds extra metadata.

        Args:
            extra_metadata: The extra metadata to add to.

        Returns:
            Extra metadata added.
        """

        extra_metadata.time_elapsed = (datetime.datetime.now() - self.start_time).total_seconds() / 60
        extra_metadata.current_trial = len(self.outcome_history)
        extra_metadata.responses = sum([1 for choice in self.is_right_choice_history if choice is not None])
        extra_metadata.ignored = sum([1 for choice in self.is_right_choice_history if choice is None])
        extra_metadata.earned_water = sum(
            [
                oc.trial.reward_size.left
                for oc in self.outcome_history
                if oc.is_rewarded and not oc.is_right_choice and oc.trial.is_auto_reward_right is None
            ]
            + [
                oc.trial.reward_size.right
                for oc in self.outcome_history
                if oc.is_rewarded and oc.is_right_choice and oc.trial.is_auto_reward_right is None
            ]
        )
        extra_metadata.total_water = sum(
            [oc.trial.reward_size.left for oc in self.outcome_history if oc.is_rewarded and not oc.is_right_choice]
            + [oc.trial.reward_size.right for oc in self.outcome_history if oc.is_rewarded and oc.is_right_choice]
        )
        extra_metadata.foraging_efficiency = calculate_foraging_efficiency(
            is_baiting=self.spec.is_baiting,
            is_rewarded=self.reward_history,
            p_left_reward=[oc.trial.metadata.p_reward_left for oc in self.outcome_history],
            p_right_reward=[oc.trial.metadata.p_reward_right for oc in self.outcome_history],
        )
        return extra_metadata

    def get_metrics(self) -> TrialMetrics:
        """Return metrics at current state of the trial generator."""

        return TrialMetrics(bias=self.bias)

    def _are_autowater_conditions_met(self) -> bool:
        """Checks whether autowater should be given.

        Returns:
            True if autowater conditions are met, False otherwise.
        """

        if self.spec.autowater_parameters is None:
            logger.debug("Autowater not configured.")
            return False

        min_ignore = self.spec.autowater_parameters.min_ignored_trials
        min_unreward = self.spec.autowater_parameters.min_unrewarded_trials

        if min_ignore == 0 or min_unreward == 0:
            logger.debug(
                "Autowater enabled every trial (min_ignored_trials=%s, min_unrewarded_trials=%s).",
                min_ignore,
                min_unreward,
            )
            return True

        is_ignored = [choice is None for choice in self.is_right_choice_history]
        if len(is_ignored) >= min_ignore and all(is_ignored[-min_ignore:]):
            logger.debug("Past %s trials ignored." % min_ignore)
            return True

        is_unrewarded = [not reward for reward in self.reward_history]
        if len(is_unrewarded) >= min_unreward and all(is_unrewarded[-min_unreward:]):
            logger.debug("Past %s trials unrewarded." % min_unreward)
            return True

        return False

    @abstractmethod
    def _are_end_conditions_met(self) -> bool:
        """Checks whether the session should end.

        Returns:
            True if end conditions are met and no further trials should be
            generated, False otherwise.
        """
        pass

    @abstractmethod
    def _generate_next_block(*args, **kwargs) -> Block:
        """Abstract method. Subclasses must implement their own block switching logic.

        Returns:
            A new Block with sampled reward probabilities and length.
        """

        pass

    @abstractmethod
    def _is_block_switch_allowed(self) -> bool:
        """Determines whether all criteria are met to switch to the next block.

        Returns:
            True if all switch criteria are satisfied, False otherwise.
        """

        pass

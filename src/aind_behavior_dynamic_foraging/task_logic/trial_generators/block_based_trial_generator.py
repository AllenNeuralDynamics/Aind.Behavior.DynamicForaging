import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
from aind_behavior_services.task.distributions import (
    Distribution,
    ExponentialDistribution,
    ExponentialDistributionParameters,
    TruncationParameters,
)
from aind_behavior_services.task.distributions_utils import draw_sample
from aind_dynamic_foraging_models.logistic_regression import fit_logistic_regression
from pydantic import BaseModel, Field

from ..trial_models import Trial
from ._base import BaseTrialGeneratorSpecModel, ITrialGenerator, TrialOutcome

logger = logging.getLogger(__name__)


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


class BiasThreshold(BaseModel):
    upper: float = Field(default=0.7, ge=0, le=1, description="Absolute value of the upper bias threshold.")
    lower: float = Field(default=0.3, ge=0, le=1, description="Absolute value of the lower bias threshold.")


class AntiBiasParameters(BaseModel):
    threshold: BiasThreshold = Field(
        default=BiasThreshold(), validate_default=True, description="Thresholds for bias correction."
    )
    intervention_interval: int = Field(default=10, ge=0, description="Trials between bias intervention.")
    maximum_water_corrections: int = Field(default=5, ge=0, description="Number of water correction to attempt.")
    volume: int = Field(default=1, ge=0, description="Volume in ul of water given.")
    bias_window_length: int = Field(default=200, ge=0, description="Trials to calculate bias over.")
    lickspout_offset_delta: float = Field(default=0.05, ge=0, description="Absolute value of delta (mm) to move stage.")


class Block(BaseModel):
    p_right_reward: float = Field(ge=0, le=1, description="Reward probability for right side during block.")
    p_left_reward: float = Field(ge=0, le=1, description="Reward probability for left side during block.")
    right_length: int = Field(ge=0, description="Minimum number of trials in block.")
    left_length: int = Field(ge=0, description="Minimum number of trials in block.")


class BlockBasedTrialGeneratorSpec(BaseTrialGeneratorSpecModel):
    type: Literal["BlockBasedTrialGenerator"] = "BlockBasedTrialGenerator"

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
            truncation_parameters=TruncationParameters(min=1, max=8),
        ),
        description="Distribution describing the inter-trial interval (in seconds).",
    )

    block_length: Distribution = Field(
        default=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1 / 20),
            truncation_parameters=TruncationParameters(min=20, max=60),
        ),
        description="Distribution describing block length.",
    )

    autowater_parameters: Optional[AutoWaterParameters] = Field(
        default=AutoWaterParameters(),
        validate_default=True,
        description="Auto water settings. If set, free water is delivered when the animal exceeds the ignored or unrewarded trial thresholds.",
    )

    antibias_parameters: Optional[AntiBiasParameters] = Field(
        default=AntiBiasParameters(),
        validate_default=True,
        description="Antibias settings. If set, trial generator with give water and move lickspouts to combat bias.",
    )

    is_baiting: bool = Field(default=False, description="Whether uncollected rewards carry over to the next trial.")

    def create_generator(self) -> "BlockBasedTrialGenerator":
        return BlockBasedTrialGenerator(self)


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
        self.is_right_choice_history: list[bool | None] = []
        self.reward_history: list[bool] = []
        self.is_left_baited: bool = False
        self.is_right_baited: bool = False
        self.block: Block

        # antibias parameters
        self.trials_in_bias_intervention = 0
        self.water_corrections = 0
        self.bias: float
        self.total_lickspout_offset = 0

    def update(self, outcome: TrialOutcome | str):
        """Updates generator state from the previous trial outcome. Records choice and reward history and manages baiting state.
        Args:
            outcome: The TrialOutcome from the most recently completed trial.
        """
        logger.debug("Updating trial generator.")
        if isinstance(outcome, str):
            outcome = TrialOutcome.model_validate_json(outcome)

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

        is_auto_response_right = None

        # determine autowater
        if self._are_autowater_conditions_met():
            is_auto_response_right = True if self.block.p_right_reward > self.block.p_left_reward else False
            logger.debug("Delivering autowater: is_auto_response_right = %s" % is_auto_response_right)

        # determine bias correction. Overrides autowater
        lickspout_offset_delta = 0
        if self._are_antibias_conditions_met():
            is_auto_response_right, lickspout_offset_delta = self._determine_antibias_intervention()
            logger.debug(
                "Performing bias intervention: is_auto_response_right = %s, lickspout_offset_delta = %s."
                % (is_auto_response_right, lickspout_offset_delta)
            )

        return Trial(
            p_reward_left=1 if (self.is_left_baited and self.spec.is_baiting) else self.block.p_left_reward,
            p_reward_right=1 if (self.is_right_baited and self.spec.is_baiting) else self.block.p_right_reward,
            reward_consumption_duration=self.spec.reward_consumption_duration,
            response_deadline_duration=self.spec.response_duration,
            quiescence_period_duration=quiescent,
            inter_trial_interval_duration=iti,
            is_auto_response_right=is_auto_response_right,
            lickspout_offset_delta=lickspout_offset_delta,
        )

    def _are_autowater_conditions_met(self) -> bool:
        """Checks whether autowater should be given.

        Returns:
            True if autowater conditions are met, False otherwise.
        """

        if self.spec.autowater_parameters is None:  # autowater disabled
            logger.debug("Autowater not configured.")
            return False

        min_ignore = self.spec.autowater_parameters.min_ignored_trials
        min_unreward = self.spec.autowater_parameters.min_unrewarded_trials

        is_ignored = [choice is None for choice in self.is_right_choice_history]
        if all(is_ignored[-min_ignore:]):
            logger.debug("Past %s trials ignored." % min_ignore)
            return True

        is_unrewarded = [not reward for reward in self.reward_history]
        if all(is_unrewarded[-min_unreward:]):
            logger.debug("Past %s trials unrewarded." % min_unreward)
            return True

        return False

    def _are_antibias_conditions_met(self) -> bool:
        """Checks whether antibias conditions are met.

        Returns:
            True if antibias conditions are met, False otherwise.
        """

        if self.spec.antibias_parameters is None:  # antibias disabled
            logger.debug("Anitbias not configured.")
            return False

        if self.trials_in_bias_intervention > self.spec.antibias_parameters.intervention_interval:
            # update bias
            choice_history = (
                np.array(
                    self.is_right_choice_history[-self.spec.antibias_parameters.bias_window_length :], dtype=float
                ),
            )
            reward_history = self.reward_history[-self.spec.antibias_parameters.bias_window_length :]
            lr = fit_logistic_regression(
                choice_history=np.array(choice_history, dtype=float),
                reward_history=np.array(reward_history, dtype=float),
                n_trial_back=5,
                cv=10,
                fit_exponential=10,
            )
            self.bias = lr["df_beta"].loc["bias"]["cross_validation"].values[0]

            if self.bias <= self.spec.antibias_parameters.threshold.lower:
                logger.debug("Bias calculated below threshold: %s." % self.bias)
                return True

            if self.bias >= self.spec.antibias_parameters.threshold.upper:
                logger.debug("Bias calculated above threshold: %s." % self.bias)
                return True

        return False

    def _determine_antibias_intervention(self) -> tuple[bool | None, float]:
        """Determine anitbias interventions to perform: give water or move lickspouts

        Returns:
            Tuple dictating is_auto_response_right and lickspout_offset_delta of trial
        """

        is_right_autowater = None
        lickspout_offset_delta = 0
        ab_delta = self.spec.antibias_parameters.lickspout_offset_delta
        if abs(self.bias) > self.spec.antibias_parameters.threshold.upper:
            if self.water_corrections < self.spec.antibias_parameters.maximum_water_corrections:
                logger.debug("Correcting bias with water.")
                is_right_autowater = (
                    True if self.bias < 0 else False
                )  # - bias values corresponds to left, so give right and vice versa
                self.water_corrections += 1
            else:
                logger.debug("Correcting bias with lickspout offset.")
                lickspout_offset_delta = ab_delta if self.bias < 0 else -ab_delta  # + values move lickspout right
                self.water_corrections = 0

        elif (
            abs(self.bias) < self.spec.antibias_parameters.threshold.lower and self.total_lickspout_offset != 0
        ):  # bias below lower threshold, move back towards center
            logger.debug("Moving lickspout back toward center.")
            delta = min(ab_delta, abs(self.total_lickspout_offset))
            lickspout_offset_delta = -delta if self.total_lickspout_offset > 0 else delta

        self.total_lickspout_offset += lickspout_offset_delta

        return is_right_autowater, lickspout_offset_delta

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

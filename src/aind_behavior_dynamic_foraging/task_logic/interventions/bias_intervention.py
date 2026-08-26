import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BiasThreshold(BaseModel):
    upper: float = Field(default=0.7, ge=0, le=1, description="Absolute value of the upper bias threshold.")
    lower: float = Field(default=0.3, ge=0, le=1, description="Absolute value of the lower bias threshold.")


class BiasInterventionParameters(BaseModel):
    threshold: BiasThreshold = Field(
        default=BiasThreshold(), validate_default=True, description="Thresholds for bias correction intervention."
    )
    intervention_interval: int = Field(default=10, ge=0, description="Trials between bias intervention.")
    maximum_water_corrections: int = Field(default=5, ge=0, description="Number of water correction to attempt.")
    bias_window_length: int = Field(default=200, ge=0, description="Trials to calculate bias over.")
    lickspout_offset_delta: float = Field(
        default=0.05,
        ge=0,
        description="Distance (mm) to move the stage spouts by. This is a relative distance to the current value, not absolute.",
    )
    reward_fraction: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Fraction of full reward volume delivered for water intervention (0=none, 1=full).",
    )
    trial_threshold: int = Field(
        default=100,
        ge=0,
        description="Minimum number of trials that must elapse before anti-bias intervention can trigger. "
        "Bias is unreliable with few trials, so interventions are suppressed until this threshold is reached.",
    )


class BiasIntervention:
    """Manages bias correction interventions during a task.

    Tracks the animal's side bias and applies corrections — either automatic water
    rewards or lickspout position adjustments — when bias exceeds defined thresholds.
    Corrections are only considered after a minimum number of trials have elapsed
    since the last intervention.

    Attributes
    ----------
    parameters : Optional[BiasInterventionParameters]
        Configuration for bias thresholds, intervention interval, and correction
        magnitudes. If None, all interventions are disabled.
    trials_in_bias_intervention : int
        Trials elapsed since the last intervention. Must exceed
        ``parameters.intervention_interval`` before a new intervention can trigger.
    water_corrections : int
        Number of consecutive water corrections given in the current intervention.
        Resets to 0 when switching to lickspout correction.
    total_lickspout_offset : float
        Cumulative lickspout offset (mm) applied since instantiation.
        Used to track how far the spout has drifted from center.
    """

    def __init__(
        self,
        bias_intervention_parameters: Optional[BiasInterventionParameters] = None,
    ):

        self.parameters = bias_intervention_parameters

        self.trials_in_bias_intervention = 0
        self.water_corrections = 0
        self.total_lickspout_offset = 0

    def are_antibias_conditions_met(self, bias: float, n_trials: int = 0) -> bool:
        """Checks whether antibias conditions are met.

        Intervention is only considered once ``n_trials`` has reached
        ``parameters.trial_threshold`` and ``trials_in_bias_intervention`` exceeds
        ``parameters.intervention_interval``.
        If the bias is outside the threshold range at that point, returns True and
        leaves the counter unchanged (the caller is expected to call
        ``determine_antibias_intervention``, which resets it).
        If conditions are not met, increments ``trials_in_bias_intervention`` by 1.

        Args:
            bias: Current bias value.
            n_trials: Total number of trials elapsed in the session.

        Returns:
            True if antibias conditions are met, False otherwise.
        """
        if self.parameters is None:
            logger.debug("Bias intervention not configured.")
            return False

        if n_trials < self.parameters.trial_threshold:
            logger.debug("Minimum trial count not reached (%s/%s).", n_trials, self.parameters.trial_threshold)
            self.trials_in_bias_intervention += 1
            return False

        if self.trials_in_bias_intervention > self.parameters.intervention_interval:
            if abs(bias) >= self.parameters.threshold.upper:
                logger.debug("Bias calculated above threshold: %s." % bias)
                return True

            # bias intervention only when the spout is currently off-center.
            if abs(bias) < self.parameters.threshold.lower and self.total_lickspout_offset != 0:
                logger.debug("Bias calculated below threshold: %s." % bias)
                return True
        self.trials_in_bias_intervention += 1
        return False

    def determine_antibias_intervention(self, bias: float) -> tuple[Optional[bool], float]:
        """Determine anitbias interventions to perform: give water or move lickspouts

        Called after ``are_antibias_conditions_met`` returns True. Resets
        ``trials_in_bias_intervention`` to 0 regardless of which intervention is applied.

        Water corrections are attempted first, up to ``parameters.maximum_water_corrections``
        consecutive times. Once that limit is reached, the lickspout is moved instead and
        the water correction counter resets. If bias is below the lower threshold and the
        lickspout has drifted from center, it is nudged back by at most
        ``parameters.lickspout_offset_delta`` mm.

        Returns:
            Tuple dictating is_auto_reward_right and lickspout_offset_delta of trial
        """

        if self.parameters is None:
            logger.debug("Bias intervention not configured.")
            return None, 0

        is_right_autowater = None
        lickspout_offset_delta = 0
        ab_delta = self.parameters.lickspout_offset_delta
        if abs(bias) >= self.parameters.threshold.upper:
            if self.water_corrections < self.parameters.maximum_water_corrections:
                logger.debug("Correcting bias with water.")
                # - bias values corresponds to left, so give right and vice versa
                is_right_autowater = True if bias < 0 else False
                self.water_corrections += 1
            else:
                logger.debug("Correcting bias with lickspout offset.")
                # + values move lickspout right
                lickspout_offset_delta = ab_delta if bias < 0 else -ab_delta
                self.water_corrections = 0

        elif (
            abs(bias) < self.parameters.threshold.lower and self.total_lickspout_offset != 0
        ):  # bias below lower threshold, move back towards center
            logger.debug("Moving lickspout back toward center.")
            delta = min(self.parameters.lickspout_offset_delta, abs(self.total_lickspout_offset))
            lickspout_offset_delta = -delta if self.total_lickspout_offset > 0 else delta

        self.total_lickspout_offset += lickspout_offset_delta
        self.trials_in_bias_intervention = 0

        return is_right_autowater, lickspout_offset_delta

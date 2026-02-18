from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, TypeAliasType, Union

from pydantic import BaseModel, Field, SerializeAsAny


class AuditorySecondaryReinforcer(BaseModel):
    """Represents an auditory secondary reinforcer."""

    type: Literal["Auditory"] = "Auditory"


if TYPE_CHECKING:
    SecondaryReinforcer = Union[(AuditorySecondaryReinforcer,)]
else:
    SecondaryReinforcer = TypeAliasType(
        "SecondaryReinforcer",
        Annotated[
            Union[(AuditorySecondaryReinforcer,)],
            Field(discriminator="type", description="Type of secondary reinforcer"),
        ],
    )


class Trial(BaseModel):
    """Represents a single trial that can be instantiated by the Bonsai state machine."""

    p_reward_left: float = Field(
        default=1.0, ge=0, le=1, description="The probability of reward on the left side if response is made."
    )
    p_reward_right: float = Field(
        default=1.0, ge=0, le=1, description="The probability of reward on the right side if response is made."
    )
    reward_consumption_duration: float = Field(
        default=5.0, ge=0, description="Duration of reward consumption before transition to ITI (in seconds)."
    )
    reward_delay_duration: float = Field(
        default=0.0, ge=0, description="Delay before reward is delivered after the secondary reinforcer (in seconds)."
    )
    secondary_reinforcer: Optional[SecondaryReinforcer] = Field(
        default=None, description="Defines the secondary reinforcer used in the trial."
    )
    response_deadline_duration: float = Field(
        default=5.0, ge=0, description="Time allowed for the subject to make a response (in seconds)."
    )
    enable_fast_retract: bool = Field(
        default=False, description="If true, the opposite lickspout retracts quickly after a response is made."
    )
    quiescence_period_duration: float = Field(
        default=0.5,
        ge=0,
        description="Duration of the quiescence period before trial starts (in seconds). Each lick resets the timer.",
    )
    inter_trial_interval_duration: float = Field(
        default=5.0, ge=0.5, description="Duration of the inter-trial interval (in seconds)."
    )
    is_auto_response_right: Optional[bool] = Field(
        default=None,
        description="If set, the trial will automatically (and immediately) register a response to the right (True) or left (False).",
    )
    lickspout_offset: float = Field(
        default=0.0,
        description="Horizontal offset of the lickspouts (in mm). Positive values move the lickspouts right.",
    )
    extra_metadata: Optional[SerializeAsAny[Any]] = Field(
        default=None,
        description="Additional metadata to include with the trial. This field will NOT be used or validated by the task engine.",
    )


class TrialOutcome(BaseModel):
    """Represents the outcome of a single trial."""

    trial: Trial = Field(description="The trial associated with this outcome.")
    is_right_choice: Optional[bool] = Field(
        description="Reports the choice made by the subject. True for right, False for left, None for no choice."
    )
    is_rewarded: bool = Field(description="Indicates whether the subject received a reward on this trial.")

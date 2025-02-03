from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union

CONDITION_TYPES = Literal["Right choice", "Left reward", "Right reward", "Left no reward", "Right no reward"]
INTERVAL_CONDITION_TYPES = Literal[
    "Trial start",
    "Go cue",
    "Reward outcome",
    "Left choice",
    "Right choice",
    "Left reward",
    "Right reward",
    "Left no reward",
    "Right no reward"
]


class LaserBaseClass(BaseModel):
    name: str = Field(..., description="Name of laser")
    power: List[0, 1, 1.5, 2, 2.5, 3] = Field(..., description="Power of laser in mW")


class LaserOne(LaserBaseClass):
    name: Literal["LaserOne"] = "LaserOne"


class LaserTwo(LaserBaseClass):
    name: Literal["LaserTwo"] = "LaserTwo"


class AvailableLaserTypes(RootModel):
    root: Union[
        LaserOne,
        LaserTwo,
    ]


class IntervalConditions(BaseModel):
    condition: INTERVAL_CONDITION_TYPES = Field(..., description="Condition for interval")
    offset = float = Field(..., description="Offset in seconds.")


class _ProtocolBaseType(BaseModel):
    name: str = Field(..., description="Name of protocol.")


class SineProtocol(_ProtocolBaseType):
    name: Literal['Sine'] = 'Sine'
    frequency: float = Field(..., description="Frequency of sine wave.")


class PulseProtocol(_ProtocolBaseType):
    name: Literal['Pulse'] = 'Pulse'
    frequency: float = Field(..., description="Frequency of pulse.")
    duration: float = Field(..., description="Duration of pulse in seconds")


class ConstantProtocol(_ProtocolBaseType):
    name: Literal['Constant'] = 'Constant'
    ramp_down: float = Field(..., description="Ramp down of laser in seconds.")


class LaserColors(BaseModel):
    color: Literal['Blue', 'Red', 'Green', 'Orange'] = Field(..., description="Color of laser.")
    location: list[AvailableLaserTypes]
    probability: float = Field(..., title="Probability", ge=0, le=1)
    duration: float = Field(..., description="Duration of laser in seconds.", ge=0)
    condition: Optional[CONDITION_TYPES] = Field(..., description="Condition of laser.")
    condition_probability: float = Field(..., title="Condition Probability", ge=0, le=1)
    start: Optional[IntervalConditions] = Field(default=None, description="Start condition of laser")
    end: Optional[IntervalConditions] = Field(default=None, description="End condition of laser")
    protocol: Union[SineProtocol, PulseProtocol, ConstantProtocol] = Field(..., description="Protocol for laser.")


class SessionControl(BaseModel):
    session_fraction: float = Field(..., description="Fraction of session that will be optogenetic.", ge=0, le=1)
    optogenetic_start: bool = Field(..., description="If first trial will be opotgentic")
    alternating_sessions: bool = Field(..., description="Alternate if optogenetics is used session by session.")


class Optogenetics(BaseModel):
    experiment_type: Literal["Optogenetics"] = "Optogenetics"
    laser_colors: list[LaserColors] = Field(..., description="List of lasers used in experiment.")
    session_control: Optional[SessionControl] = Field(..., description="Field defining session wide parameters.")
    minimum_trial_interval: int = Field(..., description="Minimum trial count between two opto trials.")
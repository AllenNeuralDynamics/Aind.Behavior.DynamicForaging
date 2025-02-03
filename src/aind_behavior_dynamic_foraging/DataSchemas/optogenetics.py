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
    power: List[0, 1, 1.5, 2, 2.5, 3] = Field(default=1, description="Power of laser in mW")


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
    frequency: float = Field(default=40, description="Frequency of sine wave.")


class PulseProtocol(_ProtocolBaseType):
    name: Literal['Pulse'] = 'Pulse'
    frequency: float = Field(default=40, description="Frequency of pulse.")
    duration: float = Field(default=.002, description="Duration of pulse in seconds")


class ConstantProtocol(_ProtocolBaseType):
    name: Literal['Constant'] = 'Constant'
    ramp_down: float = Field(default=1, description="Ramp down of laser in seconds.")


class LaserColors(BaseModel):
    color: Literal['Blue', 'Red', 'Green', 'Orange'] = Field(..., description="Color of laser.")
    location: list[AvailableLaserTypes] = Field(default=[LaserOne(), LaserTwo()], description="List of lasers to use.")
    probability: float = Field(default=.25, title="Probability", ge=0, le=1)
    duration: float = Field(default=5, description="Duration of laser in seconds.", ge=0)
    condition: Optional[CONDITION_TYPES] = Field(default=None, description="Condition of laser.")
    condition_probability: float = Field(default=1, title="Condition Probability", ge=0, le=1)
    start: Optional[IntervalConditions] = Field(default=None, description="Start condition of laser")
    end: Optional[IntervalConditions] = Field(default=None, description="End condition of laser")
    protocol: Union[SineProtocol, PulseProtocol, ConstantProtocol] = Field(default=SineProtocol(),
                                                                           description="Protocol for laser.")


class SessionControl(BaseModel):
    session_fraction: float = Field(default=.5, description="Fraction of session that will be optogenetic.", ge=0, le=1)
    optogenetic_start: bool = Field(default=True, description="If first trial will be opotgentic")
    alternating_sessions: bool = Field(default=True, description="Alternate if optogenetics is used session by session.")


class Optogenetics(BaseModel):
    experiment_type: Literal["Optogenetics"] = "Optogenetics"
    laser_colors: list[LaserColors] = Field(default=[], description="List of lasers used in experiment.")
    session_control: Optional[SessionControl] = Field(default=None,
                                                      description="Field defining session wide parameters.")
    minimum_trial_interval: int = Field(default=10, description="Minimum trial count between two opto trials.")
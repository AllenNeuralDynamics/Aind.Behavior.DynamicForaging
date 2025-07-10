from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union
from typing_extensions import TypeAliasType

PULSE_CONDITIONS = Literal["Right choice", "Left reward", "Right reward", "Left no reward", "Right no reward"]
INTERVAL_CONDITIONS = Literal[
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
COLORS = Literal['Blue', 'Red', 'Green', 'Orange']

POWERS = [""]
AMPLITUDES = [""]

FREQUENCIES = Literal[40]

class LocationBaseClass(BaseModel):
    name: str = Field(..., description="Name of location")
    power: float = Field(default=0.0, description="Power of laser in mW")
    amplitude: float = Field(default=0.0, description="Amplitude of that correlates to power")

class LocationOne(LocationBaseClass):
    name: Literal["LocationOne"] = "LocationOne"


class LocationTwo(LocationBaseClass):
    name: Literal["LocationTwo"] = "LocationTwo"


AvailableLocations = TypeAliasType("AvailableLocations",
                                   Annotated[Union[LocationOne, LocationTwo], Field(discriminator="name")], )


class IntervalConditions(BaseModel):
    interval_condition: INTERVAL_CONDITIONS = Field(default="Trial start", description="Condition for interval")
    offset: float = Field(default=0, description="Offset in seconds.")


class _ProtocolBaseType(BaseModel):
    name: str = Field(..., description="Name of protocol.")


class SineProtocol(_ProtocolBaseType):
    name: Literal['Sine'] = 'Sine'
    frequency: FREQUENCIES = Field(default=40, description="Frequency of sine wave.")
    ramp_down: float = Field(default=1, description="Ramp down of laser in seconds.")

class PulseProtocol(_ProtocolBaseType):
    name: Literal['Pulse'] = 'Pulse'
    frequency: float = Field(default=40.0, description="Frequency of pulse.")
    duration: float = Field(default=.002, description="Duration of pulse in seconds")


class ConstantProtocol(_ProtocolBaseType):
    name: Literal['Constant'] = 'Constant'
    ramp_down: float = Field(default=1, description="Ramp down of laser in seconds.")


class LaserColors(BaseModel):
    name: str = Field(..., description="Name of Laser.")
    color: COLORS = Field(..., description="Color of laser.")
    location: list[AvailableLocations] = Field(default=[LocationOne(), LocationTwo()],
                                               description="List of lasers to use.")
    probability: float = Field(default=.25, title="Probability", ge=0, le=1)
    duration: float = Field(default=5.0, description="Duration of laser in seconds.", ge=0)
    pulse_condition: Optional[PULSE_CONDITIONS] = Field(default=None, description="Condition of laser.")
    condition_probability: float = Field(default=1.0, title="Condition Probability", ge=0, le=1)
    start: Optional[IntervalConditions] = Field(default=None, description="Start condition of laser")
    end: Optional[IntervalConditions] = Field(default=None, description="End condition of laser")
    protocol: Union[SineProtocol, PulseProtocol, ConstantProtocol] = Field(default=SineProtocol(),
                                                                           description="Protocol for laser.")


class LaserColorOne(LaserColors):
    name: Literal["LaserColorOne"] = "LaserColorOne"


class LaserColorTwo(LaserColors):
    name: Literal["LaserColorTwo"] = "LaserColorTwo"


class LaserColorThree(LaserColors):
    name: Literal["LaserColorThree"] = "LaserColorThree"


class LaserColorFour(LaserColors):
    name: Literal["LaserColorFour"] = "LaserColorFour"

class LaserColorFive(LaserColors):
    name: Literal["LaserColorFive"] = "LaserColorFive"

class LaserColorSix(LaserColors):
    name: Literal["LaserColorSix"] = "LaserColorSix"


AvailableLaserColors = TypeAliasType("AvailableLaserColors",
                                     Annotated[
                                         Union[LaserColorOne,
                                               LaserColorTwo,
                                               LaserColorThree,
                                               LaserColorFour,
                                               LaserColorFive,
                                               LaserColorSix], Field(
                                             discriminator="name")], )


class SessionControl(BaseModel):
    session_fraction: float = Field(default=.5, description="Fraction of session that will be optogenetic.", ge=0, le=1)
    optogenetic_start: bool = Field(default=True, description="If first trial will be opotgentic")
    alternating_sessions: bool = Field(default=True,
                                       description="Alternate if optogenetics is used session by session.")


class Optogenetics(BaseModel):
    name: Literal["Optogenetics"] = Field(default="Optogenetics", frozen=True)
    laser_colors: list[AvailableLaserColors] = Field(default=[],
                                                     description="List of lasers used in experiment.",
                                                     max_length=6)
    session_control: Optional[SessionControl] = Field(default=None,
                                                      description="Field defining session wide parameters.")
    minimum_trial_interval: int = Field(default=0, description="Minimum trial count between two opto trials.")
    sample_frequency: int = Field(defualt=5000, description="Sample frequency of wave")
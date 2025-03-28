from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union

class AutoStop(BaseModel):
    ignore_win: int = Field(default=30, title="Window of trials to check ignored responses")
    ignore_ratio_threshold: float = Field(default=.8,
                                          title="Threshold for acceptable ignored trials within window.",
                                          ge=0, le=1)
    max_trial: int = Field(default=1000, title="Maximal number of trials")
    max_time: int = Field(default=120, title="Maximal session time (min)")
    min_time: int = Field(default=30, title="Minimum session time (min)")

class StagePosition(BaseModel):
    x: int = Field(default=0, title="X position of stage")
    y: int = Field(default=0, title="Y position of stage")
    z: int = Field(default=0, title="Z position of stage")


class OperationalControl(BaseModel):
    name: Literal["OperationalControl"] = Field(default="OperationalControl", frozen=True)
    auto_stop: AutoStop = Field(default=AutoStop(), description="Parameters describing auto stop.")
    stage_position: Optional[StagePosition] = Field(default=None, description="Stage positions related to session.")
from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union

STAGE_NAMES = Literal["newscale", "AIND"]

class AutoStop(BaseModel):
    ignore_win: int = Field(default=30, title="Window of trials to check ignored responses")
    ignore_ratio_threshold: float = Field(default=.8,
                                          title="Threshold for acceptable ignored trials within window.",
                                          ge=0, le=1)
    max_trial: int = Field(default=1000, title="Maximal number of trials")
    max_time: int = Field(default=120, title="Maximal session time (min)")
    min_time: int = Field(default=30, title="Minimum session time (min)")

class StageSpecs(BaseModel):
    stage_name: STAGE_NAMES = Field(title="Name of stage used")
    rig_name: str = Field(title="Name of rig used")
    step_size: float = Field(default=None, title="Stage step size")
    x: Optional[float] = Field(default=None, title="X position of stage")
    y: Optional[float] = Field(default=None, title="Y position of stage")
    z: Optional[float] = Field(default=None, title="Z position of stage")


class OperationalControl(BaseModel):
    name: Literal["OperationalControl"] = Field(default="OperationalControl", frozen=True)
    auto_stop: AutoStop = Field(default=AutoStop(), description="Parameters describing auto stop.")
    stage_specs: Optional[StageSpecs] = Field(default=None, description="Stage positions related to session.")
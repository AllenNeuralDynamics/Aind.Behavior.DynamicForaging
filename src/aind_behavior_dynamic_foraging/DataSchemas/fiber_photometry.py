from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union

MODES = Literal["Normal", "Axon"]
STAGE_STARTS = Literal["stage_1_warmup", "stage_1", "stage_2", "stage_3", "stage_4", "final", "graduated"]

class FiberPhotometry(BaseModel):
    name: Literal["FiberPhotometry"] = Field(default="FiberPhotometry", frozen=True)
    enabled: bool = Field(default=True,  description="If FIP will be run during session")
    mode: MODES = Field(default="Normal", description="FIP mode for experiment.")
    baseline_time: int = Field(default=10, description="Baseline time before experiment starts in minutes.")
    stage_start: STAGE_STARTS = Field(default="stage_1_warmup", description="Stage to start FIP")
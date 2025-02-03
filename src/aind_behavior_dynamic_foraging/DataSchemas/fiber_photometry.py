from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union


class Fiber_Photometry(BaseModel):
    experiment_type: Literal["Fiber_Photometry"] = "Fiber_Photometry"
    mode: Literal["Normal", "Axon"] = Field(default="Normal", description="FIP mode for experiment.")
    baseline_time: int = Field(default=10, description="Baseline time before experiment starts in minutes.")

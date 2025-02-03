from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union


class Fiber_Photometry(BaseModel):
    experiment_type: Literal["Fiber_Photometry"] = "Fiber_Photometry"
    # more parameters to define experiment
    # mode
    # recording type
    # baseline time
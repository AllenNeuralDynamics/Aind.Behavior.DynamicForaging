from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator
from typing import Annotated, Dict, List, Literal, Optional, Self, Union


class Ephys(BaseModel):
    experiment_type: Literal["Ephys"] = "Ephys"

from pydantic import BaseModel, Field
from typing import Literal, Optional
RECORDING_TYPES = Literal["Behavior", "Surface Tagging", "Opto tagging", "Testing"]

class Ephys(BaseModel):
    experiment_type: Literal["Ephys"] = "Ephys"
    recording_type: Optional[RECORDING_TYPES] = Field(default="Behavior",
                                                      description="Recording type of ephys experiment.")
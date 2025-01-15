from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Self, Union
from pydantic import BaseModel, Field, NonNegativeFloat, RootModel, model_validator

from aind_behavior_services.task_logic import AindBehaviorTaskLogicModel, TaskParameters
from pydantic import Field

__version__ = "0.1.0"

class _NeuralExperimentTypeBase(BaseModel):
    experiment_type: str


class Fiber_Photometry(_NeuralExperimentTypeBase):
    experiment_type: Literal["Fiber_Photometry"] = "Fiber_Photometry"
    # more parameters to define experiment


class Optogenetics(_NeuralExperimentTypeBase):
    experiment_type: Literal["Optogenetics"] = "Optogenetics"
    # more parameters to define experiment


class Ephys(_NeuralExperimentTypeBase):
    experiment_type: Literal["Ephys"] = "Ephys"
    # more parameters to define experiment

class _BehaviorModalityBase(BaseModel):
    modality_type: str

class Licking(_BehaviorModalityBase):
    modality_type: Literal["Licking"] = "Licking"
    # more parameters to define experiment

class BehaviorModalities(RootModel):
    root: Annotated[
        Union[
            Licking
            ],
        Field(discriminator="modality_type"),
    ]

class Behavior(_NeuralExperimentTypeBase):
    experiment_type: Literal["Behavior"] = "Behavior"
    modalities: List[BehaviorModalities]
    # more parameters to define experiment

class NeuralExperimentTypes(RootModel):
    root: Annotated[
        Union[
            Behavior,
            Fiber_Photometry,
            Optogenetics,
            Ephys,
    ],
        Field(discriminator="experiment_type"),
    ]

class BlockParameters(BaseModel):
    probability_sum
    fa

class AindDynamicForagingTaskParameters(TaskParameters):

    neural_experiments: List[NeuralExperimentTypes] = Field(..., description="List of modalities included in session.")
    block_parameters: BlockParameters = Field(..., description="Parameters describing block conditions.")

class AindDynamicForagingTaskLogic(AindBehaviorTaskLogicModel):
    version: Literal[__version__] = __version__
    name: Literal["AindDynamicForaging"] = Field(default="AindDynamicForaging", description="Name of the task logic", frozen=True)
    task_parameters: AindDynamicForagingTaskParameters = Field(..., description="Parameters of the task logic")

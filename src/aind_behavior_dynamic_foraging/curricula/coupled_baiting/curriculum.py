from typing import Type, TypeVar

import numpy as np
import pydantic
from aind_behavior_curriculum import Curriculum, StageTransition, Trainer, create_curriculum

from aind_behavior_dynamic_foraging.task_logic import (
    AindDynamicForagingTaskLogic,
)

from ... import __semver__
from ..metrics import DynamicForagingMetrics
from .stages import s_final, s_graduated, s_stage_1, s_stage_1_warmup, s_stage_2, s_stage_3

CURRICULUM_NAME = "CoupledBaiting"
PKG_LOCATION = ".".join(__name__.split(".")[:-1])

TModel = TypeVar("TModel", bound=pydantic.BaseModel)


# --- STAGE TRANSITIONS ---


# warmup
@StageTransition
def st_stage_1_warmup_to_stage_1(metrics: DynamicForagingMetrics) -> bool:
    return metrics.session_at_current_stage >= 1


@StageTransition
def st_stage_1_warmup_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.finished_trials[-1] >= 200 and metrics.foraging_efficiency[-1] >= 0.6


# stage 1
@StageTransition
def st_stage_1_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] >= 0.6 and metrics.finished_trials[-1] >= 200


# stage 2
@StageTransition
def st_stage_2_to_stage_3(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] >= 0.65 and metrics.finished_trials[-1] >= 300


@StageTransition
def st_stage_2_to_stage_1(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] < 0.55 or metrics.finished_trials[-1] < 200


# stage 3
@StageTransition
def st_stage_3_to_final(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] >= 0.7 and metrics.finished_trials[-1] >= 400


@StageTransition
def st_stage_3_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] < 0.65 or metrics.finished_trials[-1] < 300


# stage final
@StageTransition
def st_final_to_graduated(metrics: DynamicForagingMetrics) -> bool:
    return (
        metrics.session_total >= 10
        and metrics.session_at_current_stage >= 5
        and np.mean(metrics.finished_trials[-5:]) >= 450
        and np.mean(metrics.foraging_efficiency[-5:]) >= 0.7
    )


@StageTransition
def st_final_to_stage_3(metrics: DynamicForagingMetrics) -> bool:
    return np.mean(metrics.foraging_efficiency[-5:]) < 0.60 or np.mean(metrics.finished_trials[-5:]) < 300


# --- CURRICULUM ---

curriculum_class: Type[Curriculum[AindDynamicForagingTaskLogic]] = create_curriculum(
    CURRICULUM_NAME, __semver__, (AindDynamicForagingTaskLogic,), pkg_location=PKG_LOCATION
)
CURRICULUM = curriculum_class()

# add stages
CURRICULUM.add_stage(s_stage_1_warmup)
CURRICULUM.add_stage(s_stage_1)
CURRICULUM.add_stage(s_stage_2)
CURRICULUM.add_stage(s_stage_3)
CURRICULUM.add_stage(s_final)
CURRICULUM.add_stage(s_graduated)

# add stage transitions
# warmup
CURRICULUM.add_stage_transition(
    s_stage_1_warmup, s_stage_2, st_stage_1_warmup_to_stage_2
)  # add 2 first to take priority

CURRICULUM.add_stage_transition(s_stage_1_warmup, s_stage_1, st_stage_1_warmup_to_stage_1)
# stage 1
CURRICULUM.add_stage_transition(s_stage_1, s_stage_2, st_stage_1_to_stage_2)
# stage 2
CURRICULUM.add_stage_transition(s_stage_2, s_stage_3, st_stage_2_to_stage_3)
CURRICULUM.add_stage_transition(s_stage_2, s_stage_1, st_stage_2_to_stage_1)
# stage 3
CURRICULUM.add_stage_transition(s_stage_3, s_final, st_stage_3_to_final)
CURRICULUM.add_stage_transition(s_stage_3, s_stage_2, st_stage_3_to_stage_2)
# final
CURRICULUM.add_stage_transition(s_final, s_graduated, st_final_to_graduated)
CURRICULUM.add_stage_transition(s_final, s_stage_3, st_final_to_stage_3)

TRAINER = Trainer(CURRICULUM)

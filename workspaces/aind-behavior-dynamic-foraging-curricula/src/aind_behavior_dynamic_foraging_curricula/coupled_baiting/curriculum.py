from typing import Any, Type, TypeVar

import numpy as np
import pydantic
from aind_behavior_curriculum import Curriculum, Metrics, StageTransition, Trainer, TrainerState, create_curriculum
from aind_behavior_dynamic_foraging.task_logic import (
    AindDynamicForagingTaskLogic,
)

from .. import __semver__
from ..cli import CurriculumCliArgs, CurriculumSuggestion
from ..metrics import DynamicForagingMetrics
from ..utils import metrics_from_dataset_path, trainer_state_from_file
from .stages import  make_s_stage_1_warmup, make_s_stage_1, make_s_stage_2, make_s_stage_3, make_s_stage_final, make_s_stage_graduated
CURRICULUM_NAME = "CoupledBaiting"
PKG_LOCATION = ".".join(__name__.split(".")[:-1])

TModel = TypeVar("TModel", bound=pydantic.BaseModel)


# --- STAGE TRANSITIONS ---


# warmup
@StageTransition
def st_stage_1_warmup_to_stage_1(metrics: DynamicForagingMetrics) -> bool:
    return metrics.consecutive_sessions_at_current_stage >= 1


@StageTransition
def st_stage_1_warmup_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.unignored_trials_per_session[-1] >= 200 and metrics.foraging_efficiency_per_session[-1] >= 0.6


# stage 1
@StageTransition
def st_stage_1_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency_per_session[-1] >= 0.6 and metrics.unignored_trials_per_session[-1] >= 200


# stage 2
@StageTransition
def st_stage_2_to_stage_3(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency_per_session[-1] >= 0.65 and metrics.unignored_trials_per_session[-1] >= 300


@StageTransition
def st_stage_2_to_stage_1(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency_per_session[-1] < 0.55 or metrics.unignored_trials_per_session[-1] < 200


# stage 3
@StageTransition
def st_stage_3_to_final(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency_per_session[-1] >= 0.7 and metrics.unignored_trials_per_session[-1] >= 400


@StageTransition
def st_stage_3_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency_per_session[-1] < 0.65 or metrics.unignored_trials_per_session[-1] < 300


# stage final
@StageTransition
def st_final_to_graduated(metrics: DynamicForagingMetrics) -> bool:
    return (
        metrics.total_sessions >= 10
        and metrics.consecutive_sessions_at_current_stage >= 5
        and np.mean(metrics.unignored_trials_per_session[-5:]) >= 450
        and np.mean(metrics.foraging_efficiency_per_session[-5:]) >= 0.7
    )


@StageTransition
def st_final_to_stage_3(metrics: DynamicForagingMetrics) -> bool:
    return (
        np.mean(metrics.foraging_efficiency_per_session[-5:]) < 0.60
        or np.mean(metrics.unignored_trials_per_session[-5:]) < 300
    )


# --- CURRICULUM ---
curriculum_class: Type[Curriculum[AindDynamicForagingTaskLogic]] = create_curriculum(
    CURRICULUM_NAME, __semver__, (AindDynamicForagingTaskLogic,), pkg_location=PKG_LOCATION
)
CURRICULUM = curriculum_class()

# add stages
CURRICULUM.add_stage(make_s_stage_1_warmup())
CURRICULUM.add_stage(make_s_stage_1())
CURRICULUM.add_stage(make_s_stage_2())
CURRICULUM.add_stage(make_s_stage_3())
CURRICULUM.add_stage(make_s_stage_final())
CURRICULUM.add_stage(make_s_stage_graduated())

# add stage transitions
# warmup
CURRICULUM.add_stage_transition(
    make_s_stage_1_warmup(), make_s_stage_2(), st_stage_1_warmup_to_stage_2
)  # add 2 first to take priority

CURRICULUM.add_stage_transition(make_s_stage_1_warmup(), make_s_stage_1(), st_stage_1_warmup_to_stage_1)
# stage 1
CURRICULUM.add_stage_transition(make_s_stage_1(), make_s_stage_2(), st_stage_1_to_stage_2)
# stage 2
CURRICULUM.add_stage_transition(make_s_stage_2(), make_s_stage_3(), st_stage_2_to_stage_3)
CURRICULUM.add_stage_transition(make_s_stage_2(), make_s_stage_1(), st_stage_2_to_stage_1)
# stage 3
CURRICULUM.add_stage_transition(make_s_stage_3(), make_s_stage_final(), st_stage_3_to_final)
CURRICULUM.add_stage_transition(make_s_stage_3(), make_s_stage_2(), st_stage_3_to_stage_2)
# final
CURRICULUM.add_stage_transition(make_s_stage_final(), make_s_stage_graduated(), st_final_to_graduated)
CURRICULUM.add_stage_transition(make_s_stage_final(), make_s_stage_3(), st_final_to_stage_3)

TRAINER = Trainer(CURRICULUM)


def run_curriculum(args: CurriculumCliArgs) -> CurriculumSuggestion[TrainerState[Any], Any]:
    trainer_state = trainer_state_from_file(args.input_trainer_state, TRAINER)
    metrics: Metrics = metrics_from_dataset_path(
        stage_changed=args.stage_changed,
        previous_metrics=args.previous_metrics,
        dataset_path=args.data_directory,
        trainer_state=trainer_state,
    )
    trainer_state = TRAINER.evaluate(trainer_state, metrics)
    return CurriculumSuggestion(trainer_state=trainer_state, metrics=metrics, version=__semver__)

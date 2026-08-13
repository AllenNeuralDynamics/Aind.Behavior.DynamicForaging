import logging
import os
from typing import Annotated, List, Literal, Optional

from aind_behavior_curriculum import Metrics
from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_dynamic_foraging.task_logic.utils.calculate_foraging_efficiency import calculate_foraging_efficiency
from pydantic import BeforeValidator, Field

STAGE_NAMES = Literal["stage_1_warmup", "stage_1", "stage_2", "stage_3", "final", "graduated"]

logger = logging.getLogger(__name__)


def coerce_none_to_nan(v: Optional[float]) -> float:
    if v is None:
        return float("nan")
    return v


NoneToNan = Annotated[float, BeforeValidator(coerce_none_to_nan)]


class DynamicForagingMetrics(Metrics):
    """Metrics for dynamic foraging"""

    foraging_efficiency_per_session: List[NoneToNan] = Field(
        min_length=1, description="Full history of foraging efficiency per session"
    )
    unignored_trials_per_session: List[int] = Field(
        min_length=1, description="Full history of trials finished per session"
    )
    total_sessions: int = Field(ge=0, description="Total sessions completed.")
    consecutive_sessions_at_current_stage: int = Field(ge=0, description="Last consecutive sessions at current stage.")
    stage_name: STAGE_NAMES = Field(description="Stage name of session.")


def metrics_from_dataset(
    data_directory: os.PathLike,
) -> DynamicForagingMetrics:
    """
    Create metrics for completed session.

    Args:
        data_directory (os.PathLike):
            Path to the directory containing the dataset to analyze. This
            directory is expected to include all required behavioral data files. Also includes metrics and trainer state

    Returns:
        DynamicForagingMetrics:
            Metrics for session

    Raises:
        FileNotFoundError:
            If the specified data directory or required files do not exist.

        ValueError:
            If the dataset is malformed or missing required fields for
            computing metrics.
    """

    dataset = df_foraging_dataset(data_directory)
    software_events = dataset["Behavior"]["SoftwareEvents"]
    software_events.load_all()

    trial_generator_spec = software_events["TrialGeneratorSpec"].data["data"].iloc[-1]
    is_baiting = trial_generator_spec.get("is_baiting", False)
    trial_outcomes = software_events["TrialOutcome"].data["data"].iloc
    # exclude auto response and ignored trials
    filtered = [
        t for t in trial_outcomes if t["is_right_choice"] is not None and t["trial"]["is_auto_reward_right"] is None
    ]
    is_right_choice = [to["is_right_choice"] for to in filtered]
    is_rewarded = [to["is_rewarded"] for to in filtered]
    p_right_reward = [to["trial"]["metadata"]["p_reward_right"] for to in filtered]
    p_left_reward = [to["trial"]["metadata"]["p_reward_left"] for to in filtered]
    foraging_efficiency = calculate_foraging_efficiency(
        is_baiting=is_baiting, is_rewarded=is_rewarded, p_left_reward=p_left_reward, p_right_reward=p_right_reward
    )
    logger.debug(f"Calculated foraging efficiency as {foraging_efficiency}")

    try:
        prev_metrics = DynamicForagingMetrics.model_validate(dataset["Behavior"]["PreviousMetrics"].data)
        prev_stage = prev_metrics.stage_name
    except FileNotFoundError:
        logger.info("No previous metrics found.")
        prev_metrics = None
        prev_stage = None

    foraging_efficiency_per_session = [] if not prev_metrics else prev_metrics.foraging_efficiency_per_session
    unignored_trials_per_session = [] if not prev_metrics else prev_metrics.unignored_trials_per_session
    total_sessions = 0 if not prev_metrics else prev_metrics.total_sessions
    stage_name = dataset["Behavior"]["TrainerState"].data.stage.name
    consecutive_sessions_at_current_stage = (
        0 if not prev_metrics or stage_name != prev_stage else prev_metrics.consecutive_sessions_at_current_stage
    )

    return DynamicForagingMetrics(
        foraging_efficiency_per_session=foraging_efficiency_per_session + [coerce_none_to_nan(foraging_efficiency)],
        unignored_trials_per_session=unignored_trials_per_session + [sum(x is not None for x in is_right_choice)],
        total_sessions=total_sessions + 1,
        consecutive_sessions_at_current_stage=consecutive_sessions_at_current_stage + 1,
        stage_name=stage_name,
    )


if __name__ == "__main__":
    print(metrics_from_dataset(r"C:\Users\micah.woodard\Downloads\864253_2026-07-24T194251Z").model_dump_json(indent=4))

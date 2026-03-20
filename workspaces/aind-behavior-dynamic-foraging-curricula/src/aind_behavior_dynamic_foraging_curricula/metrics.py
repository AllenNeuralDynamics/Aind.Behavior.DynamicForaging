import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from aind_behavior_curriculum import Metrics
from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from pydantic import Field


class DynamicForagingMetrics(Metrics):
    """Metrics for dynamic foraging"""

    foraging_efficiency_per_session: List[float] = Field(
        min_length=1, description="Full history of foraging efficiency per session"
    )
    unignored_trials_per_session: List[int] = Field(
        min_length=1, description="Full history of trials finished per session"
    )
    total_sessions: int = Field(ge=0, description="Total sessions completed.")
    sessions_at_current_stage: int = Field(ge=0, escription="Last consecutive sessions at current stage.")


def metrics_from_dataset(
    data_directory: os.PathLike,
    previous_metrics: Optional[os.PathLike] = None,
) -> DynamicForagingMetrics:
    """
    Create metrics for completed session.

    Args:
        data_directory (os.PathLike):
            Path to the directory containing the dataset to analyze. This
            directory is expected to include all required behavioral data files.

        previous_metrics (Optional[os.PathLike]):
            Path to a previously computed metrics file as metrics depend on previous sessions. If not provided, metrics will be based only on current session.
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

    is_baiting = software_events["TrialGeneratorSpec"].data["data"].iloc[-1]["is_baiting"]
    trial_outcomes = software_events["TrialOutcome"].data["data"].iloc
    # exclude auto response and ignored trials
    filtered = [
        t for t in trial_outcomes if t["is_right_choice"] is not None and not t["trial"]["is_auto_response_right"]
    ]
    is_right_choice = [to["is_right_choice"] for to in filtered]
    is_rewarded = [to["is_rewarded"] for to in filtered]
    p_right_reward = [to["trial"]["p_reward_right"] for to in filtered]
    p_left_reward = [to["trial"]["p_reward_left"] for to in filtered]
    foraging_efficiency = compute_foraging_efficiency(
        is_baiting=is_baiting, is_rewarded=is_rewarded, p_left_reward=p_left_reward, p_right_reward=p_right_reward
    )

    if previous_metrics:
        metrics = DynamicForagingMetrics.model_validate_json(Path(previous_metrics).read_text())
    else:
        metrics = None

    foraging_efficiency_per_session = [] if not metrics else metrics.foraging_efficiency_per_session
    unignored_trials_per_session = [] if not metrics else metrics.foraging_efficiency_per_session
    total_sessions = 0 if not metrics else metrics.foraging_efficiency_per_session
    sessions_at_current_stage = 0 if not metrics else metrics.foraging_efficiency_per_session

    return DynamicForagingMetrics(
        foraging_efficiency_per_session=foraging_efficiency_per_session + [foraging_efficiency],
        unignored_trials_per_session=unignored_trials_per_session + [sum(x is not None for x in is_right_choice)],
        total_sessions=total_sessions + 1,
        sessions_at_current_stage=sessions_at_current_stage + 1,
    )


def compute_foraging_efficiency(
    is_baiting: bool, is_rewarded: list[bool], p_right_reward: list[float], p_left_reward: list[float]
) -> float:
    """
    Compute foraging efficiency for a two-arm bandit task.

    This function calculates the ratio of actual rewards obtained to the
    optimal expected rewards for a session. The implementation is adapted from the Allen Institute dynamic foraging
    analysis codebase.

    Args:
        is_baiting (bool):
            Whether the task uses a baiting schedule. If True, rewards can
            accumulate on unchosen options; if False, rewards are independent
            per trial.

        is_rewarded (list[bool | None]):
            List indicating whether each trial resulted in a reward. `True`
            indicates a rewarded trial, `False` indicates no reward.

        p_right_reward (list[float]):
            Probability of reward for the right option on each trial.

        p_left_reward (list[float]):
            Probability of reward for the left option on each trial.

    Returns:
        float:
            Foraging efficiency, defined as the ratio of the number of
            rewarded trials to the optimal expected number of rewards for
            the session.

    Raises:
        ValueError:
            If input lists have mismatched lengths.

    Notes:
        Adapted from:
        https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis/blob/main/src/aind_dynamic_foraging_basic_analysis/metrics/foraging_efficiency.py
    """

    if not is_baiting:
        optimal_rewards_per_session = np.nanmean(np.max([p_right_reward], axis=0)) * len(p_left_reward)

    else:
        p_max = np.maximum(p_left_reward, p_right_reward)
        p_min = np.minimum(p_left_reward, p_right_reward)

        optimal_visit_ratio = np.floor(np.log(1 - p_max) / np.log(1 - p_min))
        optimal_general_reward_rates = p_max + (1 - (1 - p_min) ** (optimal_visit_ratio + 1) - p_max**2) / (
            optimal_visit_ratio + 1
        )

        simple_case = (p_min == 0) | (p_max >= 1)
        optimal_reward_per_trial = np.where(simple_case, p_max, optimal_general_reward_rates)

        optimal_rewards_per_session = np.nanmean(optimal_reward_per_trial) * len(p_left_reward)

    return is_rewarded.count(True) / optimal_rewards_per_session

import os
from typing import List

from aind_behavior_curriculum import Metrics
from pydantic import Field


class DynamicForagingMetrics(Metrics):
    """Metrics for dynamic foraging"""

    foraging_efficiency_per_session: List[float] = Field(
        min_length=1, description="Full history of foraging efficiency per session"
    )
    finished_trials_per_session: List[int] = Field(
        min_length=1, description="Full history of trials finished per session"
    )
    total_sessions: int = Field(ge=0, description="Total sessions completed.")
    sessions_at_current_stage: int = Field(ge=0, escription="Last consecutive sessions at current stage.")


def metrics_from_dataset(data_directory: os.PathLike) -> DynamicForagingMetrics:
    """TODO: query docdb for metrics from the previous session
    https://github.com/AllenNeuralDynamics/aind-physio-arch/blob/dyf-curriculum-doc/doc/curriculum/architecture.md

    could maybe do UPath do be able to do local and remote files.

    docdb_path = https://api.allenneuraldynamics.org/v1/metadata_index/data_assets
    """

    return DynamicForagingMetrics(
        foraging_efficiency_per_session=[0],
        finished_trials_per_session=[0],
        total_sessions=0,
        sessions_at_current_stage=0,
    )

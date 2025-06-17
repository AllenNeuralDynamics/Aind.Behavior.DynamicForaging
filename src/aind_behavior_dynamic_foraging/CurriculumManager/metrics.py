from aind_behavior_curriculum import Metrics

from typing import List


class DynamicForagingMetrics(Metrics):
    """
        Metrics for dynamic foraging
    """
    foraging_efficiency: List[float]  # Full history of foraging efficiency
    finished_trials: List[int]  # Full history of finished trials
    session_total: int
    session_at_current_stage: int
    ignore_rate: List[float]

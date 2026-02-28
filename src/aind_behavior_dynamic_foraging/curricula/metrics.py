from typing import List

from aind_behavior_curriculum import Metrics


class DynamicForagingMetrics(Metrics):
    """Metrics for dynamic foraging"""

    foraging_efficiency: List[float]  # Full history of foraging efficiency
    finished_trials: List[int]  # Full history of finished trials
    session_total: int
    session_at_current_stage: int

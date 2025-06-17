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


if __name__ == "__main__":
    warmup_stay = DynamicForagingMetrics(
        session_total=1,
        session_at_current_stage=0,
        foraging_efficiency=[0.0],
        finished_trials=[0],
        ignore_rate=[.1]
    )
    print(warmup_stay.ignore_rate)
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def calculate_foraging_efficiency(
    is_baiting: bool, is_rewarded: list[bool], p_right_reward: list[float], p_left_reward: list[float]
) -> Optional[float]:
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
        logger.debug("Calculated non baiting foraging efficiency.")
        optimal_rewards_per_session = np.nanmean(np.max([p_right_reward, p_left_reward], axis=0)) * len(p_left_reward)
    else:
        logger.debug("Calculated baiting foraging efficiency.")
        p_max = np.maximum(p_left_reward, p_right_reward)
        p_min = np.minimum(p_left_reward, p_right_reward)

        with np.errstate(divide="ignore", invalid="ignore"):
            optimal_visit_ratio = np.floor(np.log(1 - p_max) / np.log(1 - p_min))
            optimal_general_reward_rates = p_max + (1 - (1 - p_min) ** (optimal_visit_ratio + 1) - p_max**2) / (
                optimal_visit_ratio + 1
            )

        simple_case = (p_min == 0) | (p_max >= 1)
        optimal_reward_per_trial = np.where(simple_case, p_max, optimal_general_reward_rates)

        optimal_rewards_per_session = np.nanmean(optimal_reward_per_trial) * len(p_left_reward)
    foraging_efficiency = float(is_rewarded.count(True) / optimal_rewards_per_session)
    return round(foraging_efficiency, 3)

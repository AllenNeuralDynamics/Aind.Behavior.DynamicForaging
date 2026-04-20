import os
from typing import Optional

from aind_behavior_dynamic_foraging.data_contract import dataset
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic


def calculate_consumed_water(session_path: os.PathLike) -> Optional[float]:
    """Calculate the total volume of water consumed during a session.

    Args:
        session_path (os.PathLike): Path to the session directory.

    Returns:
        Optional[float]: Total volume of water consumed in milliliters, or None if unavailable.
    """

    trial_outcomes = dataset(session_path)["Behavior"]["SoftwareEvents"]["TrialOutcome"].load().data["data"]
    is_right_choice = [to["is_right_choice"] for to in trial_outcomes]
    is_rewarded = [to["is_rewarded"] for to in trial_outcomes]

    task_logic_data = dataset(session_path)["Behavior"]["InputSchemas"]["TaskLogic"].load().data
    task_logic = AindDynamicForagingTaskLogic.model_validate(task_logic_data)
    right_reward_size = task_logic.task_parameters.reward_size.right_value_volume
    left_reward_size = task_logic.task_parameters.reward_size.left_value_volume

    total = 0
    for choice, rewarded in zip(is_right_choice, is_rewarded):
        if rewarded:
            if choice is True:
                total += right_reward_size * 1e-3
            if choice is False:
                total += left_reward_size * 1e-3
    return total

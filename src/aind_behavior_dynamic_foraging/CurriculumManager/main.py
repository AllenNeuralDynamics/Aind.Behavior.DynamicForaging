from aind_behavior_curriculum import TrainerState, Trainer
from aind_behavior_dynamic_foraging import DynamicForagingMetrics

def calculate_metrics(data: dict) -> DynamicForagingMetrics:
    """
    Calculate session data
    :param data: dictionary object of data produced in session
    :return: metrics from session data
    """

    return

def run_curriculum(data: dict, current_trainer_state: TrainerState) -> TrainerState:
    """
    Get next trainer state based on current session metrics and trainer state
    :param data: dictionary object of data produced in session
    :param current_trainer_state: TrainerState of session
    :return: TrainerState for next session
    """


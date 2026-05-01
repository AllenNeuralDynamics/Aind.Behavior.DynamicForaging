import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome


def simulate_response(
    previous_reward: bool, previous_choice: bool | None, previous_left_bait: bool, previous_right_bait: bool
) -> TrialOutcome:

    np.random.seed(42)

    if np.random.random(1) < 0.1:
        is_right_choice = None
    elif np.random.random(1) < 0:
        is_right_choice = False
    elif previous_choice is None:
        is_right_choice = np.random.choice([True, False])
    else:
        is_right_choice = previous_choice if previous_reward else not previous_choice

    if is_right_choice is None:
        is_rewarded = False
    else:
        is_rewarded = previous_right_bait if is_right_choice else previous_left_bait

    return TrialOutcome(trial=Trial(), is_right_choice=is_right_choice, is_rewarded=is_rewarded)

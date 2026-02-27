import logging
import random

import numpy as np

from aind_behavior_dynamic_foraging.task_logic.trial_generators import WarmupTrialGeneratorSpec
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
        is_right_choice = random.choice([True, False])
    else:
        is_right_choice = previous_choice if previous_reward else not previous_choice

    if is_right_choice is None:
        is_rewarded = False
    else:
        is_rewarded = previous_right_bait if is_right_choice else previous_left_bait

    return TrialOutcome(trial=Trial(), is_right_choice=is_right_choice, is_rewarded=is_rewarded)


def main():
    warmup_trial_generator = WarmupTrialGeneratorSpec().create_generator()
    trial = Trial()
    outcome = TrialOutcome(
        trial=trial, is_right_choice=random.choice([True, False, None]), is_rewarded=random.choice([True, False])
    )
    for i in range(100):
        trial = warmup_trial_generator.next()
        warmup_trial_generator.update(outcome)
        outcome = simulate_response(
            previous_reward=outcome.is_rewarded,
            previous_choice=outcome.is_right_choice,
            previous_left_bait=warmup_trial_generator.is_left_baited,
            previous_right_bait=warmup_trial_generator.is_right_baited,
        )
        if not trial:
            print("Warmup finished")
            return

        print(f"Next trial: {trial}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()

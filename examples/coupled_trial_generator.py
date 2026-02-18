import logging
import random

from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generator import CoupledTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome


def main():
    coupled_trial_generator = CoupledTrialGeneratorSpec().create_generator()
    trial = Trial()
    for i in range(100):
        trial_outcome = TrialOutcome(
            trial=trial, is_right_choice=random.choice([True, False, None]), is_rewarded=random.choice([True, False])
        )
        coupled_trial_generator.update(trial_outcome)
        trial = coupled_trial_generator.next()
        print(f"Next trial: {trial}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

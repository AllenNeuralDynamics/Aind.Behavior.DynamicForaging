import os
import random

from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generator import CoupledTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome, Trial

def main():
    coupled_trial_generator = CoupledTrialGeneratorSpec().create_generator()
    
    for i in range(100):
        trial_outcome = TrialOutcome(trial=Trial(), 
                                     is_right_choice=random.choice([True, False, None]),
                                     is_rewarded=random.choice([True, False]))
        coupled_trial_generator.update(trial_outcome)
        coupled_trial_generator.next()


if __name__ == "__main__":
    main()

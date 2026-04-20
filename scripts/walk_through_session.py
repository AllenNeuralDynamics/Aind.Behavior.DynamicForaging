import logging
import os

from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_warmup_trial_generator import CoupledWarmupTrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

logging.basicConfig(
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def walk_through_session(data_directory: os.PathLike):
    dataset = df_foraging_dataset(data_directory)
    software_events = dataset["Behavior"]["SoftwareEvents"]
    software_events.load_all()

    trial_outcomes = software_events["TrialOutcome"].data["data"].iloc
    warmup_trial_generator = CoupledWarmupTrialGeneratorSpec().create_generator()
    for i, outcome in enumerate(trial_outcomes):
        warmup_trial_generator.update(TrialOutcome.model_validate(outcome))
        trial = warmup_trial_generator.next()

        if not trial:
            print(f"Session finished at trial {i}")
            return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk through a behavior session.")
    parser.add_argument("--data-directory", help="Path to the session directory")
    args = parser.parse_args()

    walk_through_session(args.data_directory)

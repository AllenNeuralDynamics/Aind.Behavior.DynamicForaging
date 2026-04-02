import logging
import os

from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_dynamic_foraging.task_logic.trial_generators import WarmupTrialGeneratorSpec, CoupledTrialGeneratorSpec
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
    trial_generator = CoupledTrialGeneratorSpec().create_generator()
    for i, outcome in enumerate(trial_outcomes):
        trial_generator.update(TrialOutcome.model_validate(outcome))
        trial = trial_generator.next()

        if not trial:
            print(f"Session finished at trial {i}")
            return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk through a behavior session.")
    parser.add_argument("--data-directory", help="Path to the session directory")
    args = parser.parse_args()

    walk_through_session(args.data_directory)

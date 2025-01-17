import datetime
import os

import aind_behavior_services.rig as rig
import aind_behavior_services.task_logic.distributions as distributions
from aind_behavior_services import db_utils as db
from aind_behavior_services.session import AindBehaviorSessionModel
from DataSchemas.aind_behavior_dynamic_foraging.task_logic import (
    AindDynamicForagingTaskParameters,
    AindDynamicForagingTaskLogic,
)
from DataSchemas.aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig


def mock_session() -> AindBehaviorSessionModel:
    """Generates a mock AindBehaviorSessionModel model"""
    return AindBehaviorSessionModel(
        date=datetime.datetime.now(tz=datetime.timezone.utc),
        experiment="AindDynamicForaging",
        root_path="c://",
        remote_path="c://remote",
        subject="test",
        notes="test session",
        experiment_version="0.1.0",
        allow_dirty_repo=True,
        skip_hardware_validation=False,
        experimenter=["Foo", "Bar"],
    )


def mock_rig() -> AindDynamicForagingRig:
    """Generates a mock AindVrForagingRig model"""


def mock_task_logic() -> AindDynamicForagingTaskLogic:
    """Generates a mock AindVrForagingTaskLogic model"""

    return AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            base_reward_sum=.8,
            reward_family=1,
            reward_pairs_n=1,

            uncoupled_reward="0.1,0.3,0.7",
            randomness='Exponential',

            # Block length
            block_min=0,
            block_max=60,
            block_beta=20,
            block_min_reward=0,

            # Delay period
            delay_min=0,
            delay_max=1,
            delay_beta=1,

            # Reward delay
            reward_delay=0,

            # Auto water
            auto_reward=True,
            auto_water_type="Natural",
            multiplier=0,
            unrewarded=200,
            ignored=120,

            # ITI
            iti_min=1,
            iti_max=8,
            iti_beta=2,
            iti_increase=0,

            # Response time
            response_time=1,
            reward_consume_time=3,
            stop_ignores=32,

            # Auto block
            advanced_block_auto='off',
            switch_thr=.5,
            points_in_a_row=5,

            # Auto stop
            max_trial=1000,
            max_time=120,

            # Reward size
            right_value_volume=3,
            left_value_volume=3,

            # Warmup
            warmup='off',
            warm_min_trial=60,
            warm_max_choice_ratio_bias=.1,
            warm_min_finish_ratio=.8,
            warm_windowsize=20
        )
    )


def mock_subject_database() -> db.SubjectDataBase:
    """Generates a mock database object"""
    database = db.SubjectDataBase()
    database.add_subject("test", db.SubjectEntry(task_logic_target="preward_intercept_stageA"))
    database.add_subject("test2", db.SubjectEntry(task_logic_target="does_notexist"))
    return database


def main(path_seed: str = "./local/{schema}.json"):
    example_session = mock_session()
    example_rig = mock_rig()
    example_task_logic = mock_task_logic()
    example_database = mock_subject_database()

    os.makedirs(os.path.dirname(path_seed), exist_ok=True)

    models = [example_task_logic, example_session, example_rig, example_database]

    for model in models:
        with open(path_seed.format(schema=model.__class__.__name__), "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

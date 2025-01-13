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
            BaseRewardSum=.8,
            RewardFamily=1,
            RewardPairsN=1,

            UncoupledReward="0.1,0.3,0.7",
            Randomness='Exponential',

            # Block length
            BlockMin=0,
            BlockMax=60,
            BlockBeta=20,
            BlockMinReward=0,

            # Delay period
            DelayMin=0,
            DelayMax=1,
            DelayBeta=1,

            # Reward delay
            RewardDelay=0,

            # Auto water
            AutoReward=True,
            AutoWaterType="Natural",
            Multiplier=0,
            Unrewarded=200,
            Ignored=120,

            # ITI
            ITIMin=1,
            ITIMax=8,
            ITIBeta=2,
            ITIIncrease=0,

            # Response time
            ResponseTime=1,
            RewardConsumeTime=3,
            StopIgnores=32,

            # Auto block
            AdvancedBlockAuto='off',
            SwitchThr=.5,
            PointsInARow=5,

            # Auto stop
            MaxTrial=1000,
            MaxTime=120,

            # Reward size
            RightValue_volume=3,
            LeftValue_volume=3,

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

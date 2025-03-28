import datetime
import os

from aind_behavior_services import db_utils as db
from aind_behavior_services.session import AindBehaviorSessionModel
from aind_behavior_dynamic_foraging.DataSchemas.task_logic import (
    AindDynamicForagingTaskLogic,
    AindDynamicForagingTaskParameters,
    BlockParameters,
    RewardProbability,
    DelayPeriod,
    AutoWater,
    InterTrialInterval,
    ResponseTime,
    AutoBlock,
    RewardSize,
    Warmup
)
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
        experimenter=["Chris", "P", "Bacon"],
    )


def mock_task_logic() -> AindDynamicForagingTaskLogic:
    """Generates a mock AindVrForagingTaskLogic model"""

    return AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            # Warmup ON
            warmup=Warmup(
                min_trial=50,
                max_choice_ratio_bias=0.1,
                min_finish_ratio=0.8,
                windowsize=20
            ),
            reward_probability=RewardProbability(
                base_reward_sum=0.8,
                family=3,
                pairs_n=1
            ),
            block_parameters=BlockParameters(
                min=10,
                max=30,
                beta=10,
                min_reward=0
            ),
            inter_trial_interval=InterTrialInterval(
                min=1,
                max=7,
                beta=3
            ),
            delay_period=DelayPeriod(
                min=0,
                max=0,
                beta=0
            ),
            reward_delay=0.1,
            reward_size=RewardSize(
                right_value_volume=4.0,
                left_value_volume=4.0
            ),
            auto_water=AutoWater(
                auto_water_type="Natural",
                multiplier=0.5,
                unrewarded=3,
                ignored=3,
            ),
            auto_block=AutoBlock(
                advanced_block_auto="now",
                switch_thr=0.5,
                points_in_a_row=5
            ),
            response_time=ResponseTime(
                response_time=5,
                reward_consume_time=1
            ),
            uncoupled_reward=[0.1, 0.3, 0.7]
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
    example_task_logic = mock_task_logic()
    example_database = mock_subject_database()

    os.makedirs(os.path.dirname(path_seed), exist_ok=True)

    models = [example_task_logic, example_session, example_database]

    for model in models:
        with open(path_seed.format(schema=model.__class__.__name__), "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))


if __name__ == "__main__":
    from pprint import pprint
    pprint(AindDynamicForagingTaskLogic.model_fields)
    main()

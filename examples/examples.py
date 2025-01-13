import datetime
import os

import aind_behavior_services.rig as rig
import aind_behavior_services.task_logic.distributions as distributions
from aind_behavior_services import db_utils as db
from aind_behavior_services.session import AindBehaviorSessionModel
from DataSchemas.aind_behavior_dynamic_foraging.task_logic import (
    AindDynamicForagingTaskParameters,
    NumericalUpdater,
    AindDynamicForagingTaskLogic,
    NumericalUpdaterParameters,
    NumericalUpdaterOperation,
    OperationControl,
    AudioControl,
    PositionControl,
    Vector3,
    OperantLogic,
    PatchRewardFunction,
    ConstantFunction,
    LinearFunction,
    DepletionRule,
    PatchStatistics,
    RewardSpecification,
    EnvironmentStatistics,
    BlockStructure,
    Block,
    ForagingSettings
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

    def NumericalUpdaterParametersHelper(initial_value, increment, decrement, minimum, maximum):
        return NumericalUpdaterParameters(
            initial_value=initial_value, increment=increment, decrement=decrement, minimum=minimum, maximum=maximum
        )

    updaters = {
        "RewardDelayOffset": NumericalUpdater(
            operation=NumericalUpdaterOperation.OFFSET,
            parameters=NumericalUpdaterParametersHelper(0, 0.005, 0, 0, 0.2),
        )
    }

    operation_control = OperationControl(
        audio_control=AudioControl(),
        position_control=PositionControl(
            gain=Vector3(x=1, y=1, z=1),
            initial_position=Vector3(x=0, y=2.56, z=0),
            frequency_filter_cutoff=5,
            velocity_threshold=40,
        ),
    )

    def OperantLogicHelper(stop_duration: float = 0.2, is_operant: bool = False):
        return OperantLogic(
            is_operant=is_operant,
            stop_duration=stop_duration,
            time_to_collect_reward=1000000,
            grace_distance_threshold=10,
        )

    def ExponentialDistributionHelper(rate=1, minimum=0, maximum=1000):
        return distributions.ExponentialDistribution(
            distribution_parameters=distributions.ExponentialDistributionParameters(rate=rate),
            truncation_parameters=distributions.TruncationParameters(min=minimum, max=maximum, is_truncated=True),
            scaling_parameters=distributions.ScalingParameters(scale=1.0, offset=0.0),
        )

    reward_function = PatchRewardFunction(
        amount=ConstantFunction(value=1),
        probability=ConstantFunction(value=1),
        available=LinearFunction(a=-1, b=5),
        depletion_rule=DepletionRule.ON_CHOICE,
    )

    patch1 = PatchStatistics(
        label="Amyl Acetate",
        state_index=0,
        reward_specification=RewardSpecification(
            reward_function=reward_function,
            operant_logic=OperantLogicHelper(),
            delay=ExponentialDistributionHelper(1, 0, 10),
        ),
    )

    patch2 = PatchStatistics(
        label="Alpha-pinene",
        state_index=1,
        reward_specification=RewardSpecification(
            reward_function=reward_function,
            operant_logic=OperantLogicHelper(),
            delay=ExponentialDistributionHelper(1, 0, 10),
        ),
    )

    environment_statistics = EnvironmentStatistics(
        patches=[patch1, patch2]
    )

    warm_up_block

    return AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            rng_seed=None,
            updaters=updaters,
            environment=BlockStructure(
                blocks=[Block(environment_statistics=environment_statistics, end_conditions=[])],
                sampling_mode="Random",
            ),
            task_mode_settings=ForagingSettings(),
            operation_control=operation_control,
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

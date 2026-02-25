import os

from aind_behavior_curriculum import Stage, TrainerState

import aind_behavior_dynamic_foraging.task_logic as df_task_logic
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic, AindDynamicForagingTaskParameters

task_logic = AindDynamicForagingTaskLogic(
    task_parameters=AindDynamicForagingTaskParameters(
        rng_seed=42,
        reward_size=df_task_logic.RewardSize(right_value_volume=4.0, left_value_volume=4.0),
            )
)


def main(path_seed: str = "./local/DynamicForaging_{schema}.json"):
    example_task_logic = task_logic
    example_trainer_state = TrainerState(
        stage=Stage(name="example_stage", task=example_task_logic), curriculum=None, is_on_curriculum=False
    )
    os.makedirs(os.path.dirname(path_seed), exist_ok=True)
    models = [example_task_logic, example_trainer_state]

    for model in models:
        with open(path_seed.format(schema=model.__class__.__name__), "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

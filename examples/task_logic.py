import os

from aind_behavior_curriculum import Stage, TrainerState

import aind_behavior_dynamic_foraging.task_logic as df_task_logic
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic, AindDynamicForagingTaskParameters

task_logic = AindDynamicForagingTaskLogic(
    task_parameters=AindDynamicForagingTaskParameters(
        rng_seed=42,
        warmup=df_task_logic.Warmup(min_trial=50, max_choice_ratio_bias=0.1, min_finish_ratio=0.8, windowsize=20),
        reward_probability=df_task_logic.RewardProbability(base_reward_sum=0.8, family=3, pairs_n=1),
        block_parameters=df_task_logic.BlockParameters(min=10, max=30, beta=10, min_reward=0),
        inter_trial_interval=df_task_logic.InterTrialInterval(min=1, max=7, beta=3),
        delay_period=df_task_logic.DelayPeriod(min=0, max=0, beta=0),
        reward_delay=0.1,
        reward_size=df_task_logic.RewardSize(right_value_volume=4.0, left_value_volume=4.0),
        auto_water=df_task_logic.AutoWater(
            auto_water_type="Natural",
            multiplier=0.5,
            unrewarded=3,
            ignored=3,
        ),
        auto_block=df_task_logic.AutoBlock(advanced_block_auto="now", switch_thr=0.5, points_in_a_row=5),
        response_time=df_task_logic.Response(response_time=5, reward_consume_time=1),
        uncoupled_reward=[0.1, 0.3, 0.7],
    )
)


def main(path_seed: str = "./local/PatchForaging_{schema}.json"):
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

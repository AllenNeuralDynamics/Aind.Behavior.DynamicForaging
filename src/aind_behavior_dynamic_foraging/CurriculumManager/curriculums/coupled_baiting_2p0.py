from aind_behavior_curriculum import (
    Curriculum,
    Stage,
    StageTransition,

)

from aind_behavior_dynamic_foraging import (
    AindDynamicForagingTaskParameters,
    AutoWaterMode,
    AdvancedBlockMode,
    AindDynamicForagingTaskLogic,
    DynamicForagingMetrics
)

from typing import List, Literal
import numpy as np

# --- Stages  ---
s_stage_1_warmup = Stage(
    name="stage_1_warmup",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup OFF
            warmup='on',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            # p_sum = 0.8, p_ratio = [1:0]
            base_reward_sum=0.8,
            reward_family=3,
            reward_paird_n=1,

            # block = [10, 20, 5]
            block_min=10,
            block_max=20,
            block_beta=5,
            block_min_reward=0,

            # Small ITI at the beginning to better engage the animal
            iti_min=1,
            iti_max=7,
            iti_beta=3,

            # Add a (fixed) small delay period at the beginning  # TODO: automate delay period
            delay_min=0.5,
            delay_max=0.5,
            delay_beta=0,

            # Reward size and reward delay
            reward_delay=0.0,
            right_value_volume=4.0,
            left_value_volume=4.0,

            # -- Within session automation --
            # Auto water
            auto_reward=True,
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=3,
            ignored=3,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.NOW,
            switch_thr=0.5,
            points_in_a_row=5,

            # Auto stop; set stop_ignores to a large number at the beginning
            max_trial=1000,
            Max_time=90,
            stop_ignores=20000,

            # -- Miscs --
            response_time=5, 
            reward_consume_time=1,  # Very long response time at the beginning
            uncoupled_reward="",  # Only valid in uncoupled task
        )
    )
)

s_stage_1 = Stage(
    name="stage_1",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup OFF
            warmup='off',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            # p_sum = 0.8, p_ratio = [1:0]
            base_reward_sum=0.8,
            reward_family=3,
            reward_paird_n=1,

            # block = [10, 20, 5]
            block_min=10,
            block_max=20,
            block_beta=5,
            block_min_reward=0,

            # Small ITI at the beginning to better engage the animal
            iti_min=1,
            iti_max=7,
            iti_beta=3,

            # Add a (fixed) small delay period at the beginning  # TODO: automate delay period
            delay_min=0.5,
            delay_max=0.5,
            delay_beta=0,

            # Reward size and reward delay
            reward_delay=0.0,
            right_value_volume=2.0,
            left_value_volume=2.0,

            # -- Within session automation --
            # Auto water
            auto_reward=True,
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=5,
            ignored=5,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.NOW,
            switch_thr=0.5,
            points_in_a_row=5,

            # Auto stop; set stop_ignores to a large number at the beginning
            max_trial=1000,
            Max_time=90,
            stop_ignores=20000,

            # -- Miscs --
            response_time=5, reward_consume_time=3,  # Very long response time at the beginning
            uncoupled_reward="",  # Only valid in uncoupled task
        )
    )
)

s_stage_2 = Stage(
    name="stage_2",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup OFF
            warmup='off',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            # p_sum = 0.8 --> 0.6, p_ratio = [1:0] -> [8:1]
            base_reward_sum=0.6,
            reward_family=1,
            reward_paird_n=1,

            # block length [10, 20, 5] --> [10, 40, 10]
            block_min=10,
            block_max=40,
            block_beta=10,
            block_min_reward=0,

            # ITI [1, 7, 3] --> [1, 10, 5]
            iti_min=1,
            iti_max=10,
            iti_beta=3,

            # Delay 0.5 --> 1.0
            delay_min=1.0,
            delay_max=1.0,
            delay_beta=0,

            # Reward size and reward delay
            reward_delay=0.0,
            right_value_volume=2.0,
            left_value_volume=2.0,

            # -- Within session automation --
            # Auto water
            auto_reward=True,
            auto_water_type=AutoWaterMode.NATURAL,
            # Decrease auto water: unrewarded 5 --> 10, ignored 5 --> 10
            unrewarded=10,
            ignored=10,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.NOW,
            # Increase auto block switch threshold: 0.5 --> 0.6
            switch_thr=0.6,
            points_in_a_row=5,

            # Auto stop
            max_trial=1000,
            Max_time=90,
            stop_ignores=50,  # Auto stop on ignores-in-a-row starts to take effect

            # -- Miscs --
            response_time=3,  # Decrease response time: 5 --> 3
            reward_consume_time=3,
            uncoupled_reward="",  # Only valid in uncoupled task
        )
    )
)

s_stage_3 = Stage(
    name="stage_3",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup OFF
            warmup='off',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            # p_sum = 0.6 --> 0.45, p_ratio still [8:1]
            base_reward_sum=0.45,
            reward_family=1,
            reward_paird_n=1,

            # block length [10, 40, 10] --> [20, 60, 20]
            block_min=20,
            block_max=60,
            block_beta=20,
            block_min_reward=0,

            # ITI [2, 10, 5] --> [3, 15, 5]
            iti_min=1,
            iti_max=15,
            iti_beta=3,

            # Delay 1.0 --> 1.5
            delay_min=1.5,
            delay_max=1.5,
            delay_beta=0,

            # Reward size and reward delay
            reward_delay=0.0,
            right_value_volume=2.0,
            left_value_volume=2.0,

            # -- Within session automation --
            # Auto water
            auto_reward=True,
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=15,
            ignored=15,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.NOW,
            switch_thr=0.6,
            points_in_a_row=5,

            # Auto stop
            max_trial=1000,
            Max_time=90,
            stop_ignores=50,

            # -- Miscs --
            response_time=2,  # Decrease response time:  3 --> 2
            reward_consume_time=3,
            uncoupled_reward="",  # Only valid in uncoupled task
        )
    )
)

s_final = Stage(
    name="final",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup 0FF
            warmup='off',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            # p_sum = 0.45, p_ratio = [8:1] --> [8:1], [6:1], [3:1], [1:1]
            base_reward_sum=0.45,
            reward_family=1,
            reward_paird_n=4,

            # block = [10, 20, 5] (mean ~ 33 trials)
            block_min=20,
            block_max=60,
            block_beta=20,
            block_min_reward=0,

            # ITI [1, 15, 5] --> [1, 30, 5] (mean ~ 6.0 s, not included 1-s no lick window before ITI start)
            iti_min=1,
            iti_max=30,
            iti_beta=3,

            # Delay 1.5 --> 2.0 (Bari et al. 2019)
            delay_min=2.0,
            delay_max=2.0,
            delay_beta=0,

            # Reward size and reward delay
            reward_delay=0.0,
            right_value_volume=2.0,
            left_value_volume=2.0,

            # -- Within session automation --
            # Auto water
            auto_reward=False, # Turn off auto water
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=10,
            ignored=10,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.OFF,  # Turn off auto block
            switch_thr=0.6,
            points_in_a_row=5,

            # Auto stop
            max_trial=1000,
            Max_time=90,
            stop_ignores=50,

            # -- Miscs --
            response_time=1,
            reward_consume_time=3,
            uncoupled_reward="",  # Only valid in uncoupled task
        )
    )
)

# graduated same is identical to final but an absorbing state
s_graduated = Stage(
    name="final",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup 0FF
            warmup='off',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            # p_sum = 0.45, p_ratio = [8:1] --> [8:1], [6:1], [3:1], [1:1]
            base_reward_sum=0.45,
            reward_family=1,
            reward_paird_n=4,

            # block = [10, 20, 5] (mean ~ 33 trials)
            block_min=20,
            block_max=60,
            block_beta=20,
            block_min_reward=0,

            # ITI [1, 15, 5] --> [1, 30, 5] (mean ~ 6.0 s, not included 1-s no lick window before ITI start)
            iti_min=1,
            iti_max=30,
            iti_beta=3,

            # Delay 1.5 --> 2.0 (Bari et al. 2019)
            delay_min=2.0,
            delay_max=2.0,
            delay_beta=0,

            # Reward size and reward delay
            reward_delay=0.0,
            right_value_volume=2.0,
            left_value_volume=2.0,

            # -- Within session automation --
            # Auto water
            auto_reward=False, # Turn off auto water
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=10,
            ignored=10,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.OFF,  # Turn off auto block
            switch_thr=0.6,
            points_in_a_row=5,

            # Auto stop
            max_trial=1000,
            Max_time=90,
            stop_ignores=50,

            # -- Miscs --
            response_time=1,
            reward_consume_time=3,
            uncoupled_reward="",  # Only valid in uncoupled task
        )
    )
)


# --- STAGE TRANSITIONS ---

# warmup
@StageTransition
def st_stage_1_warmup_to_stage_1(metrics: DynamicForagingMetrics) -> bool:
    return metrics.session_at_current_stage >= 1


@StageTransition
def st_stage_1_warmup_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] >= 0.6 and metrics.finished_trials[-1] >= 200


# stage 1
@StageTransition
def st_stage_1_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] >= 0.6 and metrics.finished_trials[-1] >= 200


# stage 2
@StageTransition
def st_stage_2_to_stage_3(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] >= 0.65 and metrics.finished_trials[-1] >= 300


@StageTransition
def st_stage_2_to_stage_1(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] < 0.55 or metrics.finished_trials[-1] < 200


# stage 3
@StageTransition
def st_stage_3_to_final(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] >= 0.7 and metrics.finished_trials[-1] >= 400


@StageTransition
def st_stage_3_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] < 0.65 or metrics.finished_trials[-1] < 300


# stage final
@StageTransition
def st_final_to_graduated(metrics: DynamicForagingMetrics) -> bool:
    return metrics.session_total >= 10 and \
           metrics.session_at_current_stage >= 5 and \
           np.mean(metrics.finished_trials[-5:]) >= 500 and \
           np.mean(metrics.foraging_efficiency[-5:]) >= 0.7


@StageTransition
def st_final_to_stage_3(metrics: DynamicForagingMetrics) -> bool:
    return np.mean(metrics.foraging_efficiency[-2:]) < 0.65 or np.mean(metrics.finished_trials[-2:]) < 350


# --- Curriculum ---
class CoupledBaiting1p0Curriculum(Curriculum):
    name: Literal["Coupled Baiting 1p0 Curriculum"] = "Coupled Baiting 1p0 Curriculum"


def construct_coupled_baiting_1p0_curriculum() -> CoupledBaiting1p0Curriculum:

    cb_curriculum = CoupledBaiting1p0Curriculum(name="Coupled Baiting 1p0 Curriculum")

    # add stages
    cb_curriculum.add_stage(s_stage_1_warmup)
    cb_curriculum.add_stage(s_stage_1)
    cb_curriculum.add_stage(s_stage_2)
    cb_curriculum.add_stage(s_stage_3)
    cb_curriculum.add_stage(s_final)
    cb_curriculum.add_stage(s_graduated)

    # add stage transitions
    cb_curriculum.add_stage_transition(s_stage_1_warmup, s_stage_1, st_stage_1_warmup_to_stage_1)
    cb_curriculum.add_stage_transition(s_stage_1_warmup, s_stage_2, st_stage_1_warmup_to_stage_2)
    cb_curriculum.add_stage_transition(s_stage_1, s_stage_2, st_stage_1_to_stage_2)
    cb_curriculum.add_stage_transition(s_stage_2, s_stage_3, st_stage_2_to_stage_3)
    cb_curriculum.add_stage_transition(s_stage_2, s_stage_1, st_stage_2_to_stage_1)
    cb_curriculum.add_stage_transition(s_stage_3, s_final, st_stage_3_to_final)
    cb_curriculum.add_stage_transition(s_stage_3, s_stage_2, st_stage_3_to_stage_2)
    cb_curriculum.add_stage_transition(s_final, s_graduated, st_final_to_graduated)
    cb_curriculum.add_stage_transition(s_final, s_graduated, st_final_to_graduated)
    cb_curriculum.add_stage_transition(s_final, s_stage_3, st_final_to_stage_3)


    return cb_curriculum


from aind_behavior_curriculum import (
    Curriculum,
    Stage,
    StageTransition,
    create_curriculum
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

__version__ = "0.2.3"

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
            reward_pairs_n=1,

            # block = [10, 30, 10]
            block_min=10,
            block_max=30,
            block_beta=10,
            block_min_reward=0,

            # Small ITI at the beginning to better engage the animal
            iti_min=1,
            iti_max=7,
            iti_beta=3,

            # Add a (fixed) small delay period at the beginning  # TODO: automate delay period
            delay_min=0.1,
            delay_max=0.1,
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
            max_time=75,
            auto_stop_ignore_win=20000,
            auto_stop_ignore_ratio_threshold=1,

            # -- Miscs --
            response_time=5,    # Very long response time at the beginning
            reward_consume_time=1,  # Shorter RewardConsumeTime to increase the number of trials
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
            reward_pairs_n=1,

            # block = [10, 20, 5]
            block_min=10,
            block_max=30,
            block_beta=10,
            block_min_reward=0,

            # Small ITI at the beginning to better engage the animal
            iti_min=1,
            iti_max=7,
            iti_beta=3,

            # Add a (fixed) small delay period at the beginning  # TODO: automate delay period
            delay_min=0.1,
            delay_max=0.1,
            delay_beta=0,

            # Decrease water size to 2.0 from now on
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
            max_time=75,
            auto_stop_ignore_win=20000,
            auto_stop_ignore_ratio_threshold=1,

            # -- Miscs --
            response_time=5,
            reward_consume_time=1,  # Very long response time at the beginning
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

            # p_ratio [1:0] -> [8:1]
            base_reward_sum=0.8,
            reward_family=1,
            reward_pairs_n=1,

            # block length [10, 30, 10] --> [20, 35, 20]
            block_min=20,
            block_max=35,
            block_beta=10,
            block_min_reward=0,

            # ITI [1, 7, 3] --> [1, 10, 3]
            iti_min=1,
            iti_max=10,
            iti_beta=3,

            delay_min=0.3,
            delay_max=0.3,
            delay_beta=0,

            # Reward size and reward delay
            reward_delay=0.0,
            right_value_volume=2.0,
            left_value_volume=2.0,

            # -- Within session automation --
            # Auto water
            auto_reward=True,
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=7,
            ignored=7,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.NOW,
            switch_thr=0.5,
            points_in_a_row=5,

            # Auto stop
            max_trial=1000,
            max_time=75,
            auto_stop_ignore_win=30,
            auto_stop_ignore_ratio_threshold=.83,

            # -- Miscs --
            response_time=3,     # Decrease response time: 5 --> 3
            reward_consume_time=1,
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

            base_reward_sum=0.8,
            reward_family=1,
            reward_pairs_n=1,

            block_min=20,
            block_max=35,
            block_beta=10,
            block_min_reward=0,

            # ITI [1, 10, 3] --> [1, 15, 3]
            iti_min=1,
            iti_max=15,
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
            # Turn on auto water for the first day after switching to uncoupled task
            auto_reward=True,
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=10,
            ignored=10,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.OFF,  # Turn off auto block
            switch_thr=0.5,
            points_in_a_row=5,

            # Auto stop
            max_trial=1000,
            max_time=75,
            auto_stop_ignore_win=30,
            auto_stop_ignore_ratio_threshold=.83,

            # -- Miscs --
            response_time=2,    # Decrease response time: 3 --> 2
            reward_consume_time=1,  # Very long response time at the beginning
            uncoupled_reward="0.1, 0.4, 0.7",
        )
    )
)

s_final = Stage(
    name="final",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup OFF
            warmup='off',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            base_reward_sum=0.8,
            reward_family=1,
            reward_pairs_n=1,


            block_min=20,
            block_max=35,
            block_beta=10,
            block_min_reward=0,

            iti_min=1,
            iti_max=30,
            iti_beta=3,

            delay_min=1,
            delay_max=1,
            delay_beta=0,

            reward_delay=0.0,
            right_value_volume=2.0,
            left_value_volume=2.0,

            # -- Within session automation --
            # Auto water
            auto_reward=False,  # Turn off auto water
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=10,
            ignored=10,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.OFF,  # Turn off auto block
            switch_thr=0.5,
            points_in_a_row=5,

            # Auto stop
            max_trial=1000,
            max_time=75,
            auto_stop_ignore_win=30,
            auto_stop_ignore_ratio_threshold=.83,

            # -- Miscs --
            response_time=1,
            reward_consume_time=3,
            uncoupled_reward="0.1, 0.4, 0.7",
        )
    )
)

# graduated same is identical to final but an absorbing state
s_graduated = Stage(
    name="graduated",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup OFF
            warmup='off',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            base_reward_sum=0.8,
            reward_family=1,
            reward_pairs_n=1,

            block_min=20,
            block_max=35,
            block_beta=10,
            block_min_reward=0,

            iti_min=1,
            iti_max=30,
            iti_beta=3,

            delay_min=1,
            delay_max=1,
            delay_beta=0,

            reward_delay=0.0,
            right_value_volume=2.0,
            left_value_volume=2.0,

            # -- Within session automation --
            # Auto water
            auto_reward=False,
            auto_water_type=AutoWaterMode.NATURAL,
            unrewarded=10,
            ignored=10,
            multiplier=0.5,

            # Auto block
            advanced_block_auto=AdvancedBlockMode.OFF,
            switch_thr=0.5,
            points_in_a_row=5,

            max_trial=1000,
            max_time=75,
            auto_stop_ignore_win=30,
            auto_stop_ignore_ratio_threshold=.83,

            # -- Miscs --
            response_time=1,
            reward_consume_time=3,
            uncoupled_reward="0.1, 0.4, 0.7",
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
    return metrics.foraging_efficiency[-1] >= 0.65 and\
           metrics.finished_trials[-1] >= 300 and metrics.session_at_current_stage >= 2


@StageTransition
def st_stage_2_to_stage_1(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] < 0.55 or metrics.finished_trials[-1] < 200


# stage 3
@StageTransition
def st_stage_3_to_final(metrics: DynamicForagingMetrics) -> bool:
    return metrics.session_at_current_stage >= 1


# stage final
@StageTransition
def st_final_to_graduated(metrics: DynamicForagingMetrics) -> bool:
    return metrics.session_total >= 10 and \
           metrics.session_at_current_stage >= 5 and \
           np.mean(metrics.finished_trials[-5:]) >= 450 and \
           np.mean(metrics.foraging_efficiency[-5:]) >= 0.7


@StageTransition
def st_final_to_stage_3(metrics: DynamicForagingMetrics) -> bool:
    return np.mean(metrics.foraging_efficiency[-5:]) < 0.60 or np.mean(metrics.finished_trials[-5:]) < 300


# --- Curriculum ---
def construct_uncoupled_baiting_2p3_curriculum() -> Curriculum:

    cb_curriculum = create_curriculum("UnCoupledBaiting2p3Curriculum", __version__, [AindDynamicForagingTaskLogic])()

    # add stages
    cb_curriculum.add_stage(s_stage_1_warmup)
    cb_curriculum.add_stage(s_stage_1)
    cb_curriculum.add_stage(s_stage_2)
    cb_curriculum.add_stage(s_stage_3)
    cb_curriculum.add_stage(s_final)
    cb_curriculum.add_stage(s_graduated)

    # add stage transitions
    # warmup
    cb_curriculum.add_stage_transition(s_stage_1_warmup, s_stage_2, st_stage_1_warmup_to_stage_2)   # first to set priority
    cb_curriculum.add_stage_transition(s_stage_1_warmup, s_stage_1, st_stage_1_warmup_to_stage_1)
    # stage 1
    cb_curriculum.add_stage_transition(s_stage_1, s_stage_2, st_stage_1_to_stage_2)
    # stage 2
    cb_curriculum.add_stage_transition(s_stage_2, s_stage_3, st_stage_2_to_stage_3)
    cb_curriculum.add_stage_transition(s_stage_2, s_stage_1, st_stage_2_to_stage_1)
    # stage 3
    cb_curriculum.add_stage_transition(s_stage_3, s_final, st_stage_3_to_final)
    # final
    cb_curriculum.add_stage_transition(s_final, s_graduated, st_final_to_graduated)
    cb_curriculum.add_stage_transition(s_final, s_stage_3, st_final_to_stage_3)


    return cb_curriculum


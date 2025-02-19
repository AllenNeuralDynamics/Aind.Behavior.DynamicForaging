from aind_behavior_curriculum import (
    Curriculum,
    Stage,
    StageTransition,
    create_curriculum
)

from aind_behavior_dynamic_foraging import (
    AindDynamicForagingTaskParameters,
    AindDynamicForagingTaskLogic,
    DynamicForagingMetrics
)
from aind_behavior_dynamic_foraging.DataSchemas.task_logic import (
    BlockParameters,
    RewardProbability,
    DelayPeriod,
    AutoWaterMode,
    AutoWater,
    InterTrialInterval,
    ResponseTime,
    AutoStop,
    AutoBlock,
    RewardSize,
    Warmup
)
from typing import List, Literal
import numpy as np

__version__ = "0.2.3"

# --- Stages  ---
s_stage_1_warmup = Stage(name="stage_1_warmup", task=AindDynamicForagingTaskLogic(
    task_parameters=AindDynamicForagingTaskParameters(
        warmup=Warmup(min_trial=50, max_choice_ratio_bias=0.1, min_finish_ratio=0.8, windowsize=20),
        reward_probability=RewardProbability(base_reward_sum=0.8, family=3, pairs_n=1),
        block_parameters=BlockParameters(min=10, max=30, beta=10, min_reward=0),
        inter_trial_interval=InterTrialInterval(min=1, max=7, beta=3),
        delay_period=DelayPeriod(min=0.1, max=0.1, beta=0),
        reward_delay=0.0,
        reward_size=RewardSize(right_value_volume=4.0, left_value_volume=4.0),
        auto_water=AutoWater(auto_water_type=AutoWaterMode.NATURAL, unrewarded=3, ignored=3, multiplier=0.5),
        auto_block=AutoBlock(advanced_block_auto="now", switch_thr=0.5, points_in_a_row=5),
        auto_stop=AutoStop(ignore_win=20000, ignore_ratio_threshold=1, max_trial=1000, max_time=75),
        response_time=ResponseTime(response_time=5, reward_consume_time=1),
        uncoupled_reward=None
    )
))

s_stage_1 = Stage(name="stage_1", task=AindDynamicForagingTaskLogic(
    task_parameters=AindDynamicForagingTaskParameters(
        reward_probability=RewardProbability(base_reward_sum=0.8, family=3, pairs_n=1),
        block_parameters=BlockParameters(min=10, max=30, beta=10, min_reward=0),
        inter_trial_interval=InterTrialInterval(min=1, max=7, beta=3),
        delay_period=DelayPeriod(min=0.1, max=0.1, beta=0),
        reward_delay=0.0,
        reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
        auto_water=AutoWater(auto_water_type=AutoWaterMode.NATURAL, unrewarded=5, ignored=5, multiplier=0.5),
        auto_block=AutoBlock(advanced_block_auto="now", switch_thr=0.5, points_in_a_row=5),
        auto_stop=AutoStop(ignore_win=20000, ignore_ratio_threshold=1, max_trial=1000, max_time=75),
        response_time=ResponseTime(response_time=5, reward_consume_time=1),
        uncoupled_reward=None
    )
))

s_stage_2 = Stage(name="stage_2", task=AindDynamicForagingTaskLogic(
    task_parameters=AindDynamicForagingTaskParameters(
        reward_probability=RewardProbability(base_reward_sum=0.8, family=1, pairs_n=1),
        block_parameters=BlockParameters(min=20, max=35, beta=10, min_reward=0),
        inter_trial_interval=InterTrialInterval(min=1, max=10, beta=3),
        delay_period=DelayPeriod(min=0.3, max=0.3, beta=0),
        reward_delay=0.0,
        reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
        auto_water=AutoWater(auto_water_type=AutoWaterMode.NATURAL, unrewarded=7, ignored=7, multiplier=0.5),
        auto_block=AutoBlock(advanced_block_auto="now", switch_thr=0.5, points_in_a_row=5),
        auto_stop=AutoStop(ignore_win=30, ignore_ratio_threshold=0.83, max_trial=1000, max_time=75),
        response_time=ResponseTime(response_time=3, reward_consume_time=1),
        uncoupled_reward=None
    )
))

s_stage_3 = Stage(name="stage_3", task=AindDynamicForagingTaskLogic(
    task_parameters=AindDynamicForagingTaskParameters(
        reward_probability=RewardProbability(base_reward_sum=0.8, family=1, pairs_n=1),
        block_parameters=BlockParameters(min=20, max=35, beta=10, min_reward=0),
        inter_trial_interval=InterTrialInterval(min=1, max=15, beta=3),
        delay_period=DelayPeriod(min=0.5, max=0.5, beta=0),
        reward_delay=0.0,
        reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
        auto_water=AutoWater(auto_water_type=AutoWaterMode.NATURAL, unrewarded=10, ignored=10, multiplier=0.5),
        auto_stop=AutoStop(ignore_win=30, ignore_ratio_threshold=0.83, max_trial=1000, max_time=75),
        response_time=ResponseTime(response_time=2, reward_consume_time=1),
        uncoupled_reward=[0.1, 0.4, 0.7]
    )
))

s_final = Stage(name="final", task=AindDynamicForagingTaskLogic(
    task_parameters=AindDynamicForagingTaskParameters(
        reward_probability=RewardProbability(base_reward_sum=0.8, family=1, pairs_n=1),
        block_parameters=BlockParameters(min=20, max=35, beta=10, min_reward=0),
        inter_trial_interval=InterTrialInterval(min=1, max=30, beta=3),
        delay_period=DelayPeriod(min=1, max=1, beta=0),
        reward_delay=0.0,
        reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
        auto_stop=AutoStop(ignore_win=30, ignore_ratio_threshold=0.83, max_trial=1000, max_time=75),
        response_time=ResponseTime(response_time=1, reward_consume_time=3),
        uncoupled_reward=[0.1, 0.4, 0.7]
    )
))

s_graduated = Stage(name="graduated", task=AindDynamicForagingTaskLogic(
    task_parameters=AindDynamicForagingTaskParameters(
        reward_probability=RewardProbability(base_reward_sum=0.8, family=1, pairs_n=1),
        block_parameters=BlockParameters(min=20, max=35, beta=10, min_reward=0),
        inter_trial_interval=InterTrialInterval(min=1, max=30, beta=3),
        delay_period=DelayPeriod(min=1, max=1, beta=0),
        reward_delay=0.0,
        reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
        auto_stop=AutoStop(ignore_win=30, ignore_ratio_threshold=0.83, max_trial=1000, max_time=75),
        response_time=ResponseTime(response_time=1, reward_consume_time=3),
        uncoupled_reward=[0.1, 0.4, 0.7]
    )
))



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


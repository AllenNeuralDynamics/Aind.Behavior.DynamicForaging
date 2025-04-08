from aind_behavior_curriculum import (
    Curriculum,
    Stage,
    StageTransition,
    create_curriculum
)
from aind_behavior_dynamic_foraging import DynamicForagingMetrics
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

import numpy as np

__version__ = "2.3.0"

# --- Stages ---

# Stage 1: Warmup ON
s_stage_1_warmup = Stage(
    name="stage_1_warmup",
    task=AindDynamicForagingTaskLogic(
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
            uncoupled_reward=None
        )
    )
)

# Stage 1: Warmup OFF
s_stage_1 = Stage(
    name="stage_1",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
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
                right_value_volume=2.0,
                left_value_volume=2.0
            ),
            auto_water=AutoWater(
                auto_water_type="Natural",
                multiplier=0.5,
                unrewarded=5,
                ignored=5,
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
            uncoupled_reward=None
        )
    )
)

# Stage 2: Adjustments
s_stage_2 = Stage(
    name="stage_2",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_probability=RewardProbability(
                base_reward_sum=0.8,
                family=1,
                pairs_n=1
            ),
            block_parameters=BlockParameters(
                min=20,
                max=35,
                beta=10,
                min_reward=0
            ),
            inter_trial_interval=InterTrialInterval(
                min=1,
                max=10,
                beta=3
            ),
            delay_period=DelayPeriod(
                min=0.25,
                max=0.25,
                beta=0
            ),
            reward_delay=0.1,
            reward_size=RewardSize(
                right_value_volume=2.0,
                left_value_volume=2.0
            ),
            auto_water=AutoWater(
                auto_water_type="Natural",
                multiplier=0.5,
                unrewarded=7,
                ignored=7,
            ),
            auto_block=AutoBlock(
                advanced_block_auto="now",
                switch_thr=0.5,
                points_in_a_row=5
            ),
            response_time=ResponseTime(
                response_time=1.5,
                reward_consume_time=1
            ),
            uncoupled_reward=None
        )
    )
)

# Stage 3: Further adjustments
s_stage_3 = Stage(
    name="stage_3",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_probability=RewardProbability(
                base_reward_sum=0.8,
                family=1,
                pairs_n=1
            ),
            block_parameters=BlockParameters(
                min=20,
                max=35,
                beta=10,
                min_reward=0
            ),
            inter_trial_interval=InterTrialInterval(
                min=1,
                max=10,
                beta=3
            ),
            delay_period=DelayPeriod(
                min=1,
                max=1,
                beta=0
            ),
            reward_delay=0.1,
            reward_size=RewardSize(
                right_value_volume=2.0,
                left_value_volume=2.0
            ),
            auto_water=AutoWater(
                auto_water_type="Natural",
                multiplier=0.5,
                unrewarded=10,
                ignored=10,
            ),
            auto_block=AutoBlock(
                advanced_block_auto="now",
                switch_thr=0.5,
                points_in_a_row=5
            ),
            response_time=ResponseTime(
                response_time=1.5,
                reward_consume_time=1
            ),
            uncoupled_reward=None
        )
    )
)

# Stage 4: Further adjustments with increased delay
s_stage_4 = Stage(
    name="stage_4",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_probability=RewardProbability(
                base_reward_sum=0.8,
                family=1,
                pairs_n=1
            ),
            block_parameters=BlockParameters(
                min=20,
                max=35,
                beta=10,
                min_reward=0
            ),
            inter_trial_interval=InterTrialInterval(
                min=1,
                max=15,
                beta=3
            ),
            delay_period=DelayPeriod(
                min=1,
                max=1,
                beta=0
            ),
            reward_delay=0.15,
            reward_size=RewardSize(
                right_value_volume=2.0,
                left_value_volume=2.0
            ),
            auto_water=AutoWater(
                auto_water_type="Natural",
                multiplier=0.5,
                unrewarded=10,
                ignored=10,
            ),
            response_time=ResponseTime(
                response_time=1.5,
                reward_consume_time=1
            ),
            uncoupled_reward=[0.1, 0.5, 0.9]
        )
    )
)

# Final Stage
s_final = Stage(
    name="final",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_probability=RewardProbability(
                base_reward_sum=0.8,
                family=1,
                pairs_n=1
            ),
            block_parameters=BlockParameters(
                min=20,
                max=35,
                beta=10,
                min_reward=0
            ),
            inter_trial_interval=InterTrialInterval(
                min=2,
                max=15,
                beta=3
            ),
            delay_period=DelayPeriod(
                min=1,
                max=1,
                beta=0
            ),
            reward_delay=0.2,
            reward_size=RewardSize(
                right_value_volume=2.0,
                left_value_volume=2.0
            ),
            response_time=ResponseTime(
                response_time=1.5,
                reward_consume_time=1
            ),
            uncoupled_reward=[0.1, 0.5, 0.9]
        )
    )
)

s_graduated = Stage(
    name="graduated",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_probability=RewardProbability(
                base_reward_sum=0.8,
                family=1,
                pairs_n=1
            ),
            block_parameters=BlockParameters(
                min=20,
                max=35,
                beta=10,
                min_reward=0
            ),
            inter_trial_interval=InterTrialInterval(
                min=2,
                max=15,
                beta=3
            ),
            delay_period=DelayPeriod(
                min=1,
                max=1,
                beta=0
            ),
            reward_delay=0.2,
            reward_size=RewardSize(
                right_value_volume=2.0,
                left_value_volume=2.0
            ),
            response_time=ResponseTime(
                response_time=1.5,
                reward_consume_time=1
            ),
            uncoupled_reward=[0.1, 0.5, 0.9]
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
    return metrics.session_at_current_stage >= 3


@StageTransition
def st_stage_2_to_stage_1(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] < 0.55 or metrics.finished_trials[-1] < 200


# stage 3
@StageTransition
def st_stage_3_to_stage_4(metrics: DynamicForagingMetrics) -> bool:
    return metrics.finished_trials[-1] >= 300 and \
           metrics.foraging_efficiency[-1] >= 0.65 and \
           metrics.session_at_current_stage >= 3

@StageTransition
def st_stage_3_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return (metrics.finished_trials[-1] < 250 or metrics.foraging_efficiency[-1] < 0.50) and \
           metrics.session_at_current_stage >= 3

@StageTransition
def st_stage_4_to_final(metrics: DynamicForagingMetrics) -> bool:
    return metrics.session_at_current_stage >= 2

# stage final
@StageTransition
def st_final_to_graduated(metrics: DynamicForagingMetrics) -> bool:
    return metrics.session_total >= 10 and \
           metrics.session_at_current_stage >= 5 and \
           np.mean(metrics.finished_trials[-5:]) >= 400 and \
           np.mean(metrics.foraging_efficiency[-5:]) >= 0.65


@StageTransition
def st_final_to_stage_4(metrics: DynamicForagingMetrics) -> bool:
    return np.mean(metrics.finished_trials[-5:]) < 250 or np.mean(metrics.foraging_efficiency[-5:]) < 0.60

# --Curriculum--
def construct_uncoupled_no_baiting_rd_2p3p1_curriculum() -> Curriculum:

    cb_curriculum = create_curriculum("UncoupledNoBaitingRewardDelayCurriculum2p3p1",
                                      __version__,
                                      [AindDynamicForagingTaskLogic])()

    # add stages
    cb_curriculum.add_stage(s_stage_1_warmup)
    cb_curriculum.add_stage(s_stage_1)
    cb_curriculum.add_stage(s_stage_2)
    cb_curriculum.add_stage(s_stage_3)
    cb_curriculum.add_stage(s_stage_4)
    cb_curriculum.add_stage(s_final)
    cb_curriculum.add_stage(s_graduated)

    # add stage transitions
    #   warmup
    cb_curriculum.add_stage_transition(s_stage_1_warmup, s_stage_2, st_stage_1_warmup_to_stage_2)
    cb_curriculum.add_stage_transition(s_stage_1_warmup, s_stage_1, st_stage_1_warmup_to_stage_1)
    # stage 1
    cb_curriculum.add_stage_transition(s_stage_1, s_stage_2, st_stage_1_to_stage_2)
    # stage 2
    cb_curriculum.add_stage_transition(s_stage_2, s_stage_3, st_stage_2_to_stage_3)
    cb_curriculum.add_stage_transition(s_stage_2, s_stage_1, st_stage_2_to_stage_1)
    # stage 3
    cb_curriculum.add_stage_transition(s_stage_3, s_stage_4, st_stage_3_to_stage_4)
    cb_curriculum.add_stage_transition(s_stage_3, s_stage_2, st_stage_3_to_stage_2)
    # stage 4
    cb_curriculum.add_stage_transition(s_stage_4, s_final, st_stage_4_to_final)
    # final
    cb_curriculum.add_stage_transition(s_final, s_graduated, st_final_to_graduated)
    cb_curriculum.add_stage_transition(s_final, s_stage_4, st_final_to_stage_4)

    return cb_curriculum
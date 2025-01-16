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
            BaseRewardSum=0.8,
            RewardFamily=3,
            RewardPairsN=1,

            # block = [10, 20, 5]
            BlockMin=10,
            BlockMax=20,
            BlockBeta=5,
            BlockMinReward=0,

            # Small ITI at the beginning to better engage the animal
            ITIMin=1,
            ITIMax=7,
            ITIBeta=3,

            # Add a (fixed) small delay period at the beginning  # TODO: automate delay period
            DelayMin=0.5,
            DelayMax=0.5,
            DelayBeta=0,

            # Reward size and reward delay
            RewardDelay=0.0,
            RightValue_volume=5.0,
            LeftValue_volume=5.0,

            # -- Within session automation --
            # Auto water
            AutoReward=True,
            AutoWaterType=AutoWaterMode.NATURAL,
            Unrewarded=5,
            Ignored=5,
            Multiplier=0.5,

            # Auto block
            AdvancedBlockAuto=AdvancedBlockMode.NOW,
            SwitchThr=0.5,
            PointsInARow=5,

            # Auto stop; set StopIgnores to a large number at the beginning
            MaxTrial=1000,
            MaxTime=90,
            StopIgnores=20000,

            # -- Miscs --
            ResponseTime=5, RewardConsumeTime=3,  # Very long response time at the beginning
            UncoupledReward="",  # Only valid in uncoupled task
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
            BaseRewardSum=0.8,
            RewardFamily=3,
            RewardPairsN=1,

            # block = [10, 20, 5]
            BlockMin=10,
            BlockMax=20,
            BlockBeta=5,
            BlockMinReward=0,

            # Small ITI at the beginning to better engage the animal
            ITIMin=1,
            ITIMax=7,
            ITIBeta=3,

            # Add a (fixed) small delay period at the beginning  # TODO: automate delay period
            DelayMin=0.5,
            DelayMax=0.5,
            DelayBeta=0,

            # Reward size and reward delay
            RewardDelay=0.0,
            RightValue_volume=3.0,
            LeftValue_volume=3.0,

            # -- Within session automation --
            # Auto water
            AutoReward=True,
            AutoWaterType=AutoWaterMode.NATURAL,
            Unrewarded=5,
            Ignored=5,
            Multiplier=0.5,

            # Auto block
            AdvancedBlockAuto=AdvancedBlockMode.NOW,
            SwitchThr=0.5,
            PointsInARow=5,

            # Auto stop; set StopIgnores to a large number at the beginning
            MaxTrial=1000,
            MaxTime=90,
            StopIgnores=20000,

            # -- Miscs --
            ResponseTime=5, RewardConsumeTime=3,  # Very long response time at the beginning
            UncoupledReward="",  # Only valid in uncoupled task
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
            BaseRewardSum=0.6,
            RewardFamily=1,
            RewardPairsN=1,

            # block length [10, 20, 5] --> [10, 40, 10]
            BlockMin=10,
            BlockMax=40,
            BlockBeta=10,
            BlockMinReward=0,

            # ITI [1, 7, 3] --> [1, 10, 5]
            ITIMin=1,
            ITIMax=10,
            ITIBeta=3,

            # Delay 0.5 --> 1.0
            DelayMin=1.0,
            DelayMax=1.0,
            DelayBeta=0,

            # Reward size and reward delay
            RewardDelay=0.0,
            RightValue_volume=3.0,
            LeftValue_volume=3.0,

            # -- Within session automation --
            # Auto water
            AutoReward=True,
            AutoWaterType=AutoWaterMode.NATURAL,
            # Decrease auto water: unrewarded 5 --> 10, ignored 5 --> 10
            Unrewarded=10,
            Ignored=10,
            Multiplier=0.5,

            # Auto block
            AdvancedBlockAuto=AdvancedBlockMode.NOW,
            # Increase auto block switch threshold: 0.5 --> 0.6
            SwitchThr=0.6,
            PointsInARow=5,

            # Auto stop
            MaxTrial=1000,
            MaxTime=90,
            StopIgnores=50,  # Auto stop on ignores-in-a-row starts to take effect

            # -- Miscs --
            ResponseTime=3,  # Decrease response time: 5 --> 3
            RewardConsumeTime=3,
            UncoupledReward="",  # Only valid in uncoupled task
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
            BaseRewardSum=0.45,
            RewardFamily=1,
            RewardPairsN=1,

            # block length [10, 40, 10] --> [20, 60, 20]
            BlockMin=20,
            BlockMax=60,
            BlockBeta=20,
            BlockMinReward=0,

            # ITI [2, 10, 5] --> [3, 15, 5]
            ITIMin=1,
            ITIMax=15,
            ITIBeta=3,

            # Delay 1.0 --> 1.5
            DelayMin=1.5,
            DelayMax=1.5,
            DelayBeta=0,

            # Reward size and reward delay
            RewardDelay=0.0,
            RightValue_volume=3.0,
            LeftValue_volume=3.0,

            # -- Within session automation --
            # Auto water
            AutoReward=True,
            AutoWaterType=AutoWaterMode.NATURAL,
            Unrewarded=10,
            Ignored=10,
            Multiplier=0.5,

            # Auto block
            AdvancedBlockAuto=AdvancedBlockMode.NOW,
            SwitchThr=0.6,
            PointsInARow=5,

            # Auto stop
            MaxTrial=1000,
            MaxTime=90,
            StopIgnores=50,

            # -- Miscs --
            ResponseTime=2,  # Decrease response time:  3 --> 2
            RewardConsumeTime=3,
            UncoupledReward="",  # Only valid in uncoupled task
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
            BaseRewardSum=0.45,
            RewardFamily=1,
            RewardPairsN=4,

            # block = [10, 20, 5] (mean ~ 33 trials)
            BlockMin=20,
            BlockMax=60,
            BlockBeta=20,
            BlockMinReward=0,

            # ITI [1, 15, 5] --> [1, 30, 5] (mean ~ 6.0 s, not included 1-s no lick window before ITI start)
            ITIMin=1,
            ITIMax=30,
            ITIBeta=3,

            # Delay 1.5 --> 2.0 (Bari et al. 2019)
            DelayMin=2.0,
            DelayMax=2.0,
            DelayBeta=0,

            # Reward size and reward delay
            RewardDelay=0.0,
            RightValue_volume=3.0,
            LeftValue_volume=3.0,

            # -- Within session automation --
            # Auto water
            AutoReward=False, # Turn off auto water
            AutoWaterType=AutoWaterMode.NATURAL,
            Unrewarded=10,
            Ignored=10,
            Multiplier=0.5,

            # Auto block
            AdvancedBlockAuto=AdvancedBlockMode.OFF,  # Turn off auto block
            SwitchThr=0.6,
            PointsInARow=5,

            # Auto stop
            MaxTrial=1000,
            MaxTime=90,
            StopIgnores=50,

            # -- Miscs --
            ResponseTime=2,
            RewardConsumeTime=3,
            UncoupledReward="",  # Only valid in uncoupled task
        )
    )
)

# graduated same is identical to final but an absorbing state
s_graduated = Stage(
    name="graduated",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(

            # Warmup 0FF
            warmup='off',
            warm_min_trial=50,
            warm_max_choice_ratio_bias=0.1,
            warm_min_finish_ratio=0.8,
            warm_windowsize=20,

            # p_sum = 0.45, p_ratio = [8:1] --> [8:1], [6:1], [3:1], [1:1]
            BaseRewardSum=0.45,
            RewardFamily=1,
            RewardPairsN=4,

            # block = [10, 20, 5] (mean ~ 33 trials)
            BlockMin=20,
            BlockMax=60,
            BlockBeta=20,
            BlockMinReward=0,

            # ITI [1, 15, 5] --> [1, 30, 5] (mean ~ 6.0 s, not included 1-s no lick window before ITI start)
            ITIMin=1,
            ITIMax=30,
            ITIBeta=3,

            # Delay 1.5 --> 2.0 (Bari et al. 2019)
            DelayMin=2.0,
            DelayMax=2.0,
            DelayBeta=0,

            # Reward size and reward delay
            RewardDelay=0.0,
            RightValue_volume=3.0,
            LeftValue_volume=3.0,

            # -- Within session automation --
            # Auto water
            AutoReward=False, # Turn off auto water
            AutoWaterType=AutoWaterMode.NATURAL,
            Unrewarded=10,
            Ignored=10,
            Multiplier=0.5,

            # Auto block
            AdvancedBlockAuto=AdvancedBlockMode.OFF,  # Turn off auto block
            SwitchThr=0.6,
            PointsInARow=5,

            # Auto stop
            MaxTrial=1000,
            MaxTime=90,
            StopIgnores=50,

            # -- Miscs --
            ResponseTime=2,
            RewardConsumeTime=3,
            UncoupledReward="",  # Only valid in uncoupled task
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
    return metrics.foraging_efficiency[-1] >= 0.7 and metrics.finished_trials[-1] >= 350


@StageTransition
def st_stage_3_to_stage_2(metrics: DynamicForagingMetrics) -> bool:
    return metrics.foraging_efficiency[-1] >= 0.6 or metrics.finished_trials[-1] < 250


# stage final
@StageTransition
def st_final_to_graduated(metrics: DynamicForagingMetrics) -> bool:
    return metrics.session_total >= 10 and \
           metrics.session_at_current_stage >= 5 and \
           np.mean(metrics.finished_trials[-5:]) >= 400 and \
           np.mean(metrics.foraging_efficiency[-5:]) >= 0.7


@StageTransition
def st_final_to_stage_3(metrics: DynamicForagingMetrics) -> bool:
    return np.mean(metrics.foraging_efficiency[-2:]) < 0.6 or np.mean(metrics.finished_trials[-2:]) < 300


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


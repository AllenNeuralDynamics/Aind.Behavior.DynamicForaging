from aind_behavior_curriculum import MetricsProvider, Stage
from aind_behavior_dynamic_foraging.task_logic import (
    AindDynamicForagingTaskLogic,
    AindDynamicForagingTaskParameters,
    RewardSize,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators import (
    CoupledTrialGeneratorSpec,
    TrialGeneratorCompositeSpec,
    WarmupTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    RewardProbabilityParameters,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generator import (
    BehaviorStabilityParameters,
    CoupledTrialGenerationEndConditions,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.warmup_trial_generator import (
    WarmupTrialGenerationEndConditions,
)
from aind_behavior_services.task.distributions import (
    ExponentialDistribution,
    ExponentialDistributionParameters,
    Scalar,
    TruncationParameters,
)

from ..metrics import metrics_from_dataset

# --- STAGES ---
# adapted from https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/curriculums/coupled_baiting_2p3.py


def make_s_stage_1_warmup():
    return Stage(
        name="stage_1_warmup",
        task=AindDynamicForagingTaskLogic(
            task_parameters=AindDynamicForagingTaskParameters(
                reward_size=RewardSize(right_value_volume=4.0, left_value_volume=4.0),
                trial_generator=TrialGeneratorCompositeSpec(
                    generators=[
                        WarmupTrialGeneratorSpec(
                            trial_generation_end_parameters=WarmupTrialGenerationEndConditions(
                                min_trial=50,
                                max_choice_bias=0.1,
                                min_response_rate=0.8,
                                evaluation_window=20,
                            ),
                            reward_probability_parameters=RewardProbabilityParameters(
                                base_reward_sum=1, reward_pairs=[[1.0, 0.0]]
                            ),
                            block_len=Scalar(value=1),
                            inter_trial_interval_duration=ExponentialDistribution(
                                distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                                truncation_parameters=TruncationParameters(min=1, max=7),
                            ),
                            quiescent_duration=Scalar(value=0.1),
                            min_block_reward=1,
                            is_baiting=True,
                            response_duration=5.0,
                            reward_consumption_duration=1.0,
                            kernel_size=2,
                            extend_block_on_no_response=True,
                        ),
                        CoupledTrialGeneratorSpec(
                            trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                                max_trial=1000,
                                max_time=75,
                                min_time=30,
                                ignore_win=20000,
                                ignore_ratio_threshold=1,
                            ),
                            behavior_stability_parameters=BehaviorStabilityParameters(
                                behavior_evaluation_mode="end",
                                behavior_stability_fraction=0.5,
                                min_consecutive_stable_trials=5,
                            ),
                            reward_probability_parameters=RewardProbabilityParameters(
                                base_reward_sum=0.8, reward_pairs=[[1.0, 0.0]]
                            ),
                            block_len=ExponentialDistribution(
                                distribution_parameters=ExponentialDistributionParameters(rate=0.2),
                                truncation_parameters=TruncationParameters(min=10, max=20),
                            ),
                            inter_trial_interval_duration=ExponentialDistribution(
                                distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                                truncation_parameters=TruncationParameters(min=1, max=7),
                            ),
                            quiescent_duration=Scalar(value=0.1),
                            min_block_reward=0,
                            is_baiting=True,
                            extend_block_on_no_response=True,
                            response_duration=5.0,
                            reward_consumption_duration=1.0,
                            kernel_size=2,
                        ),
                    ]
                ),
            )
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_1():
    return Stage(
        name="stage_1",
        task=AindDynamicForagingTaskLogic(
            task_parameters=AindDynamicForagingTaskParameters(
                reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
                lick_spout_retraction=False,
                trial_generator=CoupledTrialGeneratorSpec(
                    trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=75,
                        min_time=30,
                        ignore_win=20000,
                        ignore_ratio_threshold=1,
                    ),
                    behavior_stability_parameters=BehaviorStabilityParameters(
                        behavior_evaluation_mode="end",
                        behavior_stability_fraction=0.5,
                        min_consecutive_stable_trials=5,
                    ),
                    reward_probability_parameters=RewardProbabilityParameters(
                        base_reward_sum=0.8, reward_pairs=[[1.0, 0.0]]
                    ),
                    block_len=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=0.2),
                        truncation_parameters=TruncationParameters(min=10, max=20),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(min=1, max=7),
                    ),
                    quiescent_duration=Scalar(value=0.1),
                    min_block_reward=0,
                    is_baiting=False,
                    extend_block_on_no_response=True,
                    response_duration=5.0,
                    reward_consumption_duration=1.0,
                    kernel_size=2,
                ),
            )
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_2():
    return Stage(
        name="stage_2",
        task=AindDynamicForagingTaskLogic(
            task_parameters=AindDynamicForagingTaskParameters(
                reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
                lick_spout_retraction=False,
                trial_generator=CoupledTrialGeneratorSpec(
                    trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=75,
                        min_time=30,
                        ignore_win=30,
                        ignore_ratio_threshold=0.83,
                    ),
                    behavior_stability_parameters=BehaviorStabilityParameters(
                        behavior_evaluation_mode="end",
                        behavior_stability_fraction=0.6,
                        min_consecutive_stable_trials=5,
                    ),
                    reward_probability_parameters=RewardProbabilityParameters(
                        base_reward_sum=0.6, reward_pairs=[[8, 1]]
                    ),
                    block_len=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=0.1),
                        truncation_parameters=TruncationParameters(min=10, max=40),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=0.2),
                        truncation_parameters=TruncationParameters(min=1, max=10),
                    ),
                    quiescent_duration=Scalar(value=0.3),
                    min_block_reward=0,
                    is_baiting=True,
                    extend_block_on_no_response=True,
                    response_duration=3.0,
                    reward_consumption_duration=1.0,
                    kernel_size=2,
                ),
            )
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_3():
    return Stage(
        name="stage_3",
        task=AindDynamicForagingTaskLogic(
            task_parameters=AindDynamicForagingTaskParameters(
                reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
                lick_spout_retraction=False,
                trial_generator=CoupledTrialGeneratorSpec(
                    trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=75,
                        min_time=30,
                        ignore_win=30,
                        ignore_ratio_threshold=0.83,
                    ),
                    behavior_stability_parameters=BehaviorStabilityParameters(
                        behavior_evaluation_mode="end",
                        behavior_stability_fraction=0.6,
                        min_consecutive_stable_trials=5,
                    ),
                    reward_probability_parameters=RewardProbabilityParameters(
                        base_reward_sum=0.45, reward_pairs=[[8, 1]]
                    ),
                    block_len=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=0.05),
                        truncation_parameters=TruncationParameters(min=20, max=60),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(min=1, max=15),
                    ),
                    quiescent_duration=Scalar(value=0.5),
                    min_block_reward=0,
                    is_baiting=True,
                    extend_block_on_no_response=True,
                    response_duration=2.0,
                    reward_consumption_duration=1.0,
                    kernel_size=2,
                ),
            )
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_final():
    return Stage(
        name="final",
        task=AindDynamicForagingTaskLogic(
            task_parameters=AindDynamicForagingTaskParameters(
                reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
                lick_spout_retraction=False,
                trial_generator=CoupledTrialGeneratorSpec(
                    trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=75,
                        min_time=30,
                        ignore_win=30,
                        ignore_ratio_threshold=0.83,
                    ),
                    behavior_stability_parameters=None,
                    reward_probability_parameters=RewardProbabilityParameters(
                        base_reward_sum=0.45, reward_pairs=[[8, 1], [6, 1], [3, 1], [1, 1]]
                    ),
                    block_len=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=0.05),
                        truncation_parameters=TruncationParameters(min=20, max=60),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(min=1, max=30),
                    ),
                    quiescent_duration=Scalar(value=1),
                    min_block_reward=0,
                    is_baiting=True,
                    extend_block_on_no_response=True,
                    response_duration=1.0,
                    reward_consumption_duration=3.0,
                    kernel_size=2,
                ),
            )
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_graduated():
    return Stage(
        name="graduated",
        task=AindDynamicForagingTaskLogic(
            task_parameters=AindDynamicForagingTaskParameters(
                reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
                lick_spout_retraction=False,
                trial_generator=CoupledTrialGeneratorSpec(
                    trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=75,
                        min_time=30,
                        ignore_win=30,
                        ignore_ratio_threshold=0.83,
                    ),
                    behavior_stability_parameters=None,
                    reward_probability_parameters=RewardProbabilityParameters(
                        base_reward_sum=0.45, reward_pairs=[[8, 1], [6, 1], [3, 1], [1, 1]]
                    ),
                    block_len=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=0.05),
                        truncation_parameters=TruncationParameters(min=20, max=60),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(min=1, max=30),
                    ),
                    quiescent_duration=Scalar(value=1),
                    min_block_reward=0,
                    is_baiting=True,
                    extend_block_on_no_response=True,
                    response_duration=1.0,
                    reward_consumption_duration=3.0,
                    kernel_size=2,
                ),
            )
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )

from aind_behavior_curriculum import MetricsProvider, Stage
from aind_behavior_dynamic_foraging.task_logic import (
    AindDynamicForagingTaskLogic,
    AindDynamicForagingTaskParameters,
)
from aind_behavior_dynamic_foraging.task_logic.interventions.bias_intervention import (
    BiasInterventionParameters,
    BiasThreshold,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators import (
    CoupledTrialGeneratorSpec,
    CoupledWarmupTrialGeneratorSpec,
    TrialGeneratorCompositeSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    AutoWaterParameters,
    RewardSize,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.base_coupled_trial_generator import (
    RewardProbabilityParameters,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_trial_generator import (
    BehaviorStabilityParameters,
    CoupledTrialGenerationEndConditions,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generators.coupled_warmup_trial_generator import (
    CoupledWarmupTrialGenerationEndConditions,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators.uncoupled_trial_gnerator import (
    UncoupledTrialGenerationEndConditions,
    UncoupledTrialGeneratorSpec,
)
from aind_behavior_services.task.distributions import (
    ExponentialDistribution,
    ExponentialDistributionParameters,
    Scalar,
    ScalarDistributionParameter,
    ScalingParameters,
    TruncationParameters,
    UniformDistribution,
    UniformDistributionParameters,
)

from ..metrics import metrics_from_dataset

# --- STAGES ---
# adapted from https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/curriculums/uncoupled_no_baiting_2p3p1rwdDelay159.py


def make_s_stage_1_warmup():
    return Stage(
        name="STAGE_1_WARMUP",
        task=AindDynamicForagingTaskLogic(
            stage_name="STAGE_1_WARMUP",
            task_parameters=AindDynamicForagingTaskParameters(
                trial_generator=TrialGeneratorCompositeSpec(
                    generators=[
                        CoupledWarmupTrialGeneratorSpec(
                            min_block_reward=1,
                            reward_size=RewardSize(right=4.0, left=4.0),
                            trial_generation_end_parameters=CoupledWarmupTrialGenerationEndConditions(
                                min_trial=50,
                                max_choice_bias=0.1,
                                min_response_rate=0.8,
                                evaluation_window=20,
                            ),
                            reward_probability_parameters=RewardProbabilityParameters(
                                base_reward_sum=1, reward_pairs=[[1.0, 0.0]]
                            ),
                            block_length=Scalar(distribution_parameters=ScalarDistributionParameter(value=1)),
                            inter_trial_interval_duration=ExponentialDistribution(
                                distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                                truncation_parameters=TruncationParameters(truncation_mode="clamp", min=0, max=7),
                                scaling_parameters=ScalingParameters(offset=1),
                            ),
                            quiescent_duration=Scalar(distribution_parameters=ScalarDistributionParameter(value=0.1)),
                            is_baiting=True,
                            response_duration=5.0,
                            reward_consumption_duration=1.0,
                            autowater_parameters=AutoWaterParameters(
                                reward_fraction=0.8, min_ignored_trials=0, min_unrewarded_trials=0
                            ),
                            bias_intervention_parameters=BiasInterventionParameters(
                                threshold=BiasThreshold(upper=0.5, lower=0.0),
                                intervention_interval=10,
                                maximum_water_corrections=2,
                                bias_window_length=200,
                                lickspout_offset_delta=0.05,
                                reward_fraction=0.8,
                            ),
                        ),
                        CoupledTrialGeneratorSpec(
                            reward_size=RewardSize(right=4.0, left=4.0),
                            trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                                max_trial=1000,
                                max_time=4500,
                                min_time=1800,
                                ignore_window_length=20000,
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
                            block_length=ExponentialDistribution(
                                distribution_parameters=ExponentialDistributionParameters(rate=0.1),
                                truncation_parameters=TruncationParameters(min=10, max=30),
                            ),
                            inter_trial_interval_duration=ExponentialDistribution(
                                distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                                truncation_parameters=TruncationParameters(truncation_mode="clamp", min=0, max=7),
                                scaling_parameters=ScalingParameters(offset=1),
                            ),
                            quiescent_duration=Scalar(distribution_parameters=ScalarDistributionParameter(value=0.1)),
                            is_baiting=True,
                            extend_block_on_no_response=True,
                            response_duration=5.0,
                            reward_consumption_duration=1.0,
                            kernel_size=2,
                            autowater_parameters=AutoWaterParameters(
                                reward_fraction=0.5, min_ignored_trials=3, min_unrewarded_trials=3
                            ),
                            bias_intervention_parameters=BiasInterventionParameters(
                                threshold=BiasThreshold(upper=0.5, lower=0.0),
                                intervention_interval=10,
                                maximum_water_corrections=2,
                                bias_window_length=200,
                                lickspout_offset_delta=0.05,
                                reward_fraction=0.5,
                            ),
                        ),
                    ]
                ),
            ),
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_1():
    return Stage(
        name="STAGE_1",
        task=AindDynamicForagingTaskLogic(
            stage_name="STAGE_1",
            task_parameters=AindDynamicForagingTaskParameters(
                trial_generator=CoupledTrialGeneratorSpec(
                    reward_size=RewardSize(right=2.0, left=2.0),
                    trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=4500,
                        min_time=1800,
                        ignore_window_length=20000,
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
                    block_length=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=0.1),
                        truncation_parameters=TruncationParameters(min=10, max=30),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(truncation_mode="clamp", min=0, max=7),
                        scaling_parameters=ScalingParameters(offset=1),
                    ),
                    quiescent_duration=Scalar(distribution_parameters=ScalarDistributionParameter(value=0.1)),
                    is_baiting=True,
                    extend_block_on_no_response=True,
                    response_duration=5.0,
                    reward_consumption_duration=1.0,
                    kernel_size=2,
                    autowater_parameters=AutoWaterParameters(
                        reward_fraction=0.5, min_ignored_trials=5, min_unrewarded_trials=5
                    ),
                    bias_intervention_parameters=BiasInterventionParameters(
                        threshold=BiasThreshold(upper=0.5, lower=0.0),
                        intervention_interval=10,
                        maximum_water_corrections=2,
                        bias_window_length=200,
                        lickspout_offset_delta=0.05,
                        reward_fraction=0.5,
                    ),
                ),
            ),
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_2():
    return Stage(
        name="STAGE_2",
        task=AindDynamicForagingTaskLogic(
            stage_name="STAGE_2",
            task_parameters=AindDynamicForagingTaskParameters(
                trial_generator=CoupledTrialGeneratorSpec(
                    reward_size=RewardSize(right=2.0, left=2.0),
                    trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=4500,
                        min_time=1800,
                        ignore_window_length=30,
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
                    block_length=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=0.1),
                        truncation_parameters=TruncationParameters(min=10, max=40),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(truncation_mode="clamp", min=0, max=10),
                        scaling_parameters=ScalingParameters(offset=1),
                    ),
                    quiescent_duration=Scalar(distribution_parameters=ScalarDistributionParameter(value=0.3)),
                    is_baiting=True,
                    extend_block_on_no_response=True,
                    response_duration=3.0,
                    reward_consumption_duration=1.0,
                    kernel_size=2,
                    autowater_parameters=AutoWaterParameters(
                        reward_fraction=0.5, min_ignored_trials=7, min_unrewarded_trials=7
                    ),
                    bias_intervention_parameters=BiasInterventionParameters(
                        threshold=BiasThreshold(upper=0.5, lower=0.0),
                        intervention_interval=10,
                        maximum_water_corrections=2,
                        bias_window_length=200,
                        lickspout_offset_delta=0.05,
                        reward_fraction=0.5,
                    ),
                ),
            ),
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_3():
    return Stage(
        name="STAGE_3",
        task=AindDynamicForagingTaskLogic(
            stage_name="STAGE_3",
            task_parameters=AindDynamicForagingTaskParameters(
                trial_generator=UncoupledTrialGeneratorSpec(
                    reward_size=RewardSize(right=2.0, left=2.0),
                    trial_generation_end_parameters=UncoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=4500,
                        min_time=1800,
                        ignore_window_length=30,
                        ignore_ratio_threshold=0.83,
                    ),
                    reward_probabilities=[0.1, 0.4, 0.7],
                    block_length=UniformDistribution(
                        distribution_parameters=UniformDistributionParameters(min=20, max=36),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(truncation_mode="clamp", min=0, max=15),
                        scaling_parameters=ScalingParameters(offset=1),
                    ),
                    quiescent_duration=Scalar(distribution_parameters=ScalarDistributionParameter(value=0.5)),
                    is_baiting=True,
                    response_duration=2.0,
                    reward_consumption_duration=1.0,
                    autowater_parameters=AutoWaterParameters(
                        reward_fraction=0.5, min_ignored_trials=10, min_unrewarded_trials=10
                    ),
                    bias_intervention_parameters=BiasInterventionParameters(
                        threshold=BiasThreshold(upper=0.5, lower=0.0),
                        intervention_interval=10,
                        maximum_water_corrections=2,
                        bias_window_length=200,
                        lickspout_offset_delta=0.05,
                        reward_fraction=0.5,
                    ),
                ),
            ),
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_final():
    return Stage(
        name="STAGE_FINAL",
        task=AindDynamicForagingTaskLogic(
            stage_name="STAGE_FINAL",
            task_parameters=AindDynamicForagingTaskParameters(
                trial_generator=UncoupledTrialGeneratorSpec(
                    reward_size=RewardSize(right=2.0, left=2.0),
                    trial_generation_end_parameters=UncoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=4500,
                        min_time=1800,
                        ignore_window_length=30,
                        ignore_ratio_threshold=0.83,
                    ),
                    reward_probabilities=[0.1, 0.4, 0.7],
                    block_length=UniformDistribution(
                        distribution_parameters=UniformDistributionParameters(min=20, max=36),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(truncation_mode="clamp", min=0, max=30),
                        scaling_parameters=ScalingParameters(offset=1),
                    ),
                    quiescent_duration=Scalar(distribution_parameters=ScalarDistributionParameter(value=1)),
                    is_baiting=True,
                    response_duration=1.0,
                    reward_consumption_duration=3.0,
                    autowater_parameters=None,
                    bias_intervention_parameters=BiasInterventionParameters(
                        threshold=BiasThreshold(upper=0.5, lower=0.0),
                        intervention_interval=10,
                        maximum_water_corrections=2,
                        bias_window_length=200,
                        lickspout_offset_delta=0.05,
                        reward_fraction=0.5,
                    ),
                ),
            ),
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_stage_graduated():
    return Stage(
        name="GRADUATED",
        task=AindDynamicForagingTaskLogic(
            stage_name="GRADUATED",
            task_parameters=AindDynamicForagingTaskParameters(
                trial_generator=UncoupledTrialGeneratorSpec(
                    reward_size=RewardSize(right=2.0, left=2.0),
                    trial_generation_end_parameters=UncoupledTrialGenerationEndConditions(
                        max_trial=1000,
                        max_time=4500,
                        min_time=1800,
                        ignore_window_length=30,
                        ignore_ratio_threshold=0.83,
                    ),
                    reward_probabilities=[0.1, 0.4, 0.7],
                    block_length=UniformDistribution(
                        distribution_parameters=UniformDistributionParameters(min=20, max=36),
                    ),
                    inter_trial_interval_duration=ExponentialDistribution(
                        distribution_parameters=ExponentialDistributionParameters(rate=1.0 / 3),
                        truncation_parameters=TruncationParameters(truncation_mode="clamp", min=0, max=30),
                        scaling_parameters=ScalingParameters(offset=1),
                    ),
                    quiescent_duration=Scalar(distribution_parameters=ScalarDistributionParameter(value=1)),
                    is_baiting=True,
                    response_duration=1.0,
                    reward_consumption_duration=3.0,
                    autowater_parameters=None,
                    bias_intervention_parameters=BiasInterventionParameters(
                        threshold=BiasThreshold(upper=0.5, lower=0.0),
                        intervention_interval=10,
                        maximum_water_corrections=2,
                        bias_window_length=200,
                        lickspout_offset_delta=0.05,
                        reward_fraction=0.5,
                    ),
                ),
            ),
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )

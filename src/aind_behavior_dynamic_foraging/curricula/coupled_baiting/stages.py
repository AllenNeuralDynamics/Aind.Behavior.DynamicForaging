from datetime import timedelta

from aind_behavior_curriculum import Stage
from aind_behavior_services.task.distributions import (
    ExponentialDistribution,
    ExponentialDistributionParameters,
    TruncationParameters,
    UniformDistribution,
    UniformDistributionParameters,
)

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

# --- STAGES ---

s_stage_1_warmup = Stage(
    name="stage_1_warmup",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_size=RewardSize(right_value_volume=4.0, left_value_volume=4.0),
            lick_spout_retraction=False,
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
                            base_reward_sum=1,
                            family=3,
                            pairs_n=1,
                        ),
                        block_len=ExponentialDistribution(
                            distribution_parameters=ExponentialDistributionParameters(rate=1),
                            truncation_parameters=TruncationParameters(min=1, max=1),
                        ),
                        inter_trial_interval_duration=ExponentialDistribution(
                            distribution_parameters=ExponentialDistributionParameters(rate=1 / 3),
                            truncation_parameters=TruncationParameters(min=1, max=7),
                        ),
                        quiescent_duration=UniformDistribution(
                            distribution_parameters=UniformDistributionParameters(min=0.1, max=0.1),
                        ),
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
                            max_time=timedelta(minutes=75),
                            min_time=timedelta(minutes=30),
                            ignore_win=20000,
                            ignore_ratio_threshold=1,
                        ),
                        behavior_stability_parameters=BehaviorStabilityParameters(
                            behavior_evaluation_mode="end",
                            behavior_stability_fraction=0.5,
                            min_consecutive_stable_trials=5,
                        ),
                        reward_probability_parameters=RewardProbabilityParameters(
                            base_reward_sum=0.8,
                            family=3,
                            pairs_n=1,
                        ),
                        block_len=ExponentialDistribution(
                            distribution_parameters=ExponentialDistributionParameters(rate=1 / 5),
                            truncation_parameters=TruncationParameters(min=10, max=20),
                        ),
                        inter_trial_interval_duration=ExponentialDistribution(
                            distribution_parameters=ExponentialDistributionParameters(rate=1 / 3),
                            truncation_parameters=TruncationParameters(min=1, max=7),
                        ),
                        quiescent_duration=UniformDistribution(
                            distribution_parameters=UniformDistributionParameters(min=0.1, max=0.1),
                        ),
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
)

s_stage_1 = Stage(
    name="stage_1",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
            lick_spout_retraction=False,
            trial_generator=CoupledTrialGeneratorSpec(
                trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                    max_trial=1000,
                    max_time=timedelta(minutes=75),
                    min_time=timedelta(minutes=30),
                    ignore_win=20000,
                    ignore_ratio_threshold=1,
                ),
                behavior_stability_parameters=BehaviorStabilityParameters(
                    behavior_evaluation_mode="end",
                    behavior_stability_fraction=0.5,
                    min_consecutive_stable_trials=5,
                ),
                reward_probability_parameters=RewardProbabilityParameters(
                    base_reward_sum=0.8,
                    family=3,
                    pairs_n=1,
                ),
                block_len=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 5),
                    truncation_parameters=TruncationParameters(min=10, max=20),
                ),
                inter_trial_interval_duration=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 3),
                    truncation_parameters=TruncationParameters(min=1, max=7),
                ),
                quiescent_duration=UniformDistribution(
                    distribution_parameters=UniformDistributionParameters(min=0.1, max=0.1),
                ),
                min_block_reward=0,
                is_baiting=False,
                extend_block_on_no_response=True,
                response_duration=5.0,
                reward_consumption_duration=1.0,
                kernel_size=2,
            ),
        )
    ),
)

s_stage_2 = Stage(
    name="stage_2",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
            lick_spout_retraction=False,
            trial_generator=CoupledTrialGeneratorSpec(
                trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                    max_trial=1000,
                    max_time=timedelta(minutes=75),
                    min_time=timedelta(minutes=30),
                    ignore_win=30,
                    ignore_ratio_threshold=0.83,
                ),
                behavior_stability_parameters=BehaviorStabilityParameters(
                    behavior_evaluation_mode="end",
                    behavior_stability_fraction=0.6,
                    min_consecutive_stable_trials=5,
                ),
                reward_probability_parameters=RewardProbabilityParameters(
                    base_reward_sum=0.6,
                    family=1,
                    pairs_n=1,
                ),
                block_len=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 10),
                    truncation_parameters=TruncationParameters(min=10, max=40),
                ),
                inter_trial_interval_duration=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 5),
                    truncation_parameters=TruncationParameters(min=1, max=10),
                ),
                quiescent_duration=UniformDistribution(
                    distribution_parameters=UniformDistributionParameters(min=0.3, max=0.3),
                ),
                min_block_reward=0,
                is_baiting=True,
                extend_block_on_no_response=True,
                response_duration=3.0,
                reward_consumption_duration=1.0,
                kernel_size=2,
            ),
        )
    ),
)

s_stage_3 = Stage(
    name="stage_3",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
            lick_spout_retraction=False,
            trial_generator=CoupledTrialGeneratorSpec(
                trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                    max_trial=1000,
                    max_time=timedelta(minutes=75),
                    min_time=timedelta(minutes=30),
                    ignore_win=30,
                    ignore_ratio_threshold=0.83,
                ),
                behavior_stability_parameters=BehaviorStabilityParameters(
                    behavior_evaluation_mode="end",
                    behavior_stability_fraction=0.6,
                    min_consecutive_stable_trials=5,
                ),
                reward_probability_parameters=RewardProbabilityParameters(
                    base_reward_sum=0.45,
                    family=1,
                    pairs_n=1,
                ),
                block_len=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 20),
                    truncation_parameters=TruncationParameters(min=20, max=60),
                ),
                inter_trial_interval_duration=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 3),
                    truncation_parameters=TruncationParameters(min=1, max=15),
                ),
                quiescent_duration=UniformDistribution(
                    distribution_parameters=UniformDistributionParameters(min=0.5, max=0.5),
                ),
                min_block_reward=0,
                is_baiting=True,
                extend_block_on_no_response=True,
                response_duration=2.0,
                reward_consumption_duration=1.0,
                kernel_size=2,
            ),
        )
    ),
)

s_final = Stage(
    name="final",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
            lick_spout_retraction=False,
            trial_generator=CoupledTrialGeneratorSpec(
                trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                    max_trial=1000,
                    max_time=timedelta(minutes=75),
                    min_time=timedelta(minutes=30),
                    ignore_win=30,
                    ignore_ratio_threshold=0.83,
                ),
                behavior_stability_parameters=None,
                reward_probability_parameters=RewardProbabilityParameters(
                    base_reward_sum=0.45,
                    family=1,
                    pairs_n=4,
                ),
                block_len=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 20),
                    truncation_parameters=TruncationParameters(min=20, max=60),
                ),
                inter_trial_interval_duration=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 3),
                    truncation_parameters=TruncationParameters(min=1, max=30),
                ),
                quiescent_duration=UniformDistribution(
                    distribution_parameters=UniformDistributionParameters(min=1.0, max=1.0),
                ),
                min_block_reward=0,
                is_baiting=True,
                extend_block_on_no_response=True,
                response_duration=1.0,
                reward_consumption_duration=3.0,
                kernel_size=2,
            ),
        )
    ),
)

s_graduated = Stage(
    name="graduated",
    task=AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            reward_size=RewardSize(right_value_volume=2.0, left_value_volume=2.0),
            lick_spout_retraction=False,
            trial_generator=CoupledTrialGeneratorSpec(
                trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                    max_trial=1000,
                    max_time=timedelta(minutes=75),
                    min_time=timedelta(minutes=30),
                    ignore_win=30,
                    ignore_ratio_threshold=0.83,
                ),
                behavior_stability_parameters=None,
                reward_probability_parameters=RewardProbabilityParameters(
                    base_reward_sum=0.45,
                    family=1,
                    pairs_n=4,
                ),
                block_len=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 20),
                    truncation_parameters=TruncationParameters(min=20, max=60),
                ),
                inter_trial_interval_duration=ExponentialDistribution(
                    distribution_parameters=ExponentialDistributionParameters(rate=1 / 3),
                    truncation_parameters=TruncationParameters(min=1, max=30),
                ),
                quiescent_duration=UniformDistribution(
                    distribution_parameters=UniformDistributionParameters(min=1.0, max=1.0),
                ),
                min_block_reward=0,
                is_baiting=True,
                extend_block_on_no_response=True,
                response_duration=1.0,
                reward_consumption_duration=3.0,
                kernel_size=2,
            ),
        )
    ),
)

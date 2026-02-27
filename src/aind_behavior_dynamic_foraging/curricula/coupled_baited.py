from datetime import timedelta

from aind_behavior_curriculum import (
    Stage,
    StageTransition,
    create_curriculum,
    Curriculum
)

from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic, AindDynamicForagingTaskParameters, RewardSize
from aind_behavior_dynamic_foraging.task_logic.trial_generators.warmup_trial_generator import (
    WarmupTrialGenerationEndConditions,
)

from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    RewardProbabilityParameters,
)

from aind_behavior_dynamic_foraging.task_logic.trial_generators.coupled_trial_generator import (
    CoupledTrialGenerationEndConditions,
)

from aind_behavior_services.task.distributions import (
    ExponentialDistribution,
    ExponentialDistributionParameters,
    TruncationParameters,
)

from aind_behavior_dynamic_foraging.task_logic.trial_generators import (
    TrialGeneratorCompositeSpec,
    WarmupTrialGeneratorSpec,
    CoupledTrialGeneratorSpec,
)

s_stage_1_warmup = Stage(
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
                            base_reward_sum=0.8,
                            family=3,
                            pairs_n=1,
                        ),
                        block_len_distribution=ExponentialDistribution(
                            distribution_parameters=ExponentialDistributionParameters(rate=1 / 5),
                            truncation_parameters=TruncationParameters(min=10, max=20),
                        ),
                        inter_trial_interval_duration_distribution=ExponentialDistribution(
                            distribution_parameters=ExponentialDistributionParameters(rate=1 / 3),
                            truncation_parameters=TruncationParameters(min=1, max=7),
                        ),
                        min_block_reward=0,
                        response_duration=5.0,
                        reward_consumption_duration=1.0,
                    ),
                    CoupledTrialGeneratorSpec(
                        trial_generation_end_parameters=CoupledTrialGenerationEndConditions(
                            max_trial=1000,
                            max_time=timedelta(minutes=75),
                            auto_stop_ignore_win=20000,
                            ignore_ratio_threshold=1,
                        ),
                        reward_probability_parameters=RewardProbabilityParameters(
                            base_reward_sum=0.8,
                            family=3,
                            pairs_n=1,
                        ),
                        inter_trial_interval_duration_distribution=ExponentialDistribution(
                            distribution_parameters=ExponentialDistributionParameters(rate=1 / 3),
                            truncation_parameters=TruncationParameters(min=1, max=7),
                        ),
                        response_duration=5.0,
                        reward_consumption_duration=1.0,
                    ),
                ]
            ),
        )
    ),
)
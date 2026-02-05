from typing import Literal

from ..trial_models import Trial, TrialOutcome
from ._base import ITrialGenerator, _BaseTrialGeneratorSpecModel


class IntegrationTestTrialGeneratorSpec(_BaseTrialGeneratorSpecModel):
    type: Literal["IntegrationTestTrialGenerator"] = "IntegrationTestTrialGenerator"

    def create_generator(self) -> "IntegrationTestTrialGenerator":
        return IntegrationTestTrialGenerator(self)


class IntegrationTestTrialGenerator(ITrialGenerator):
    def __init__(self, spec: IntegrationTestTrialGeneratorSpec) -> None:
        self._spec = spec
        self._idx = 0

        self.trial_opts = [
            Trial(),  # 0: left and right reward
            Trial(p_reward_left=1.0, p_reward_right=0.0),  # 1: left reward
            Trial(p_reward_left=0.0, p_reward_right=1.0),  # 2: right reward
            Trial(p_reward_left=0.0, p_reward_right=0.0),  # 3: no reward
            Trial(p_reward_left=1.0, p_reward_right=1.0),  # 4: both reward
            # auto response right
            Trial(
                p_reward_left=1.0, p_reward_right=0.0, is_auto_response_right=True
            ),  # 5: left reward, auto response right
            Trial(
                p_reward_left=0.0, p_reward_right=1.0, is_auto_response_right=True
            ),  # 6: right reward, auto response right
            Trial(
                p_reward_left=1.0, p_reward_right=1.0, is_auto_response_right=True
            ),  # 9: both reward, auto response right,
            Trial(
                p_reward_left=0.0, p_reward_right=0.0, is_auto_response_right=True
            ),  # 10: no reward, auto response right,
            # auto response left
            Trial(
                p_reward_left=1.0, p_reward_right=0.0, is_auto_response_right=False
            ),  # 11: left reward, auto response left,
            Trial(
                p_reward_left=0.0, p_reward_right=1.0, is_auto_response_right=False
            ),  # 12: right reward, auto response left,
            Trial(
                p_reward_left=0.0, p_reward_right=1.0, is_auto_response_right=False
            ),  # 13: both reward, auto response left,
            Trial(
                p_reward_left=0.0, p_reward_right=0.0, is_auto_response_right=False
            ),  # 14: no reward, auto response left,
            # fast retract
            Trial(enable_fast_retract=True),  # 15: enable fast retract
            # secondary reinforcer
            # Trial(secondary_reinforcer=SecondaryReinforcer()),   # 16: enable secondary reinforcer
            # no reward consumption duration
            Trial(reward_consumption_duration=0),  # 17: no reward consumption duration
            # no reward delay
            Trial(reward_delay_duration=0),  # 18: no reward delay duration
            # no response deadline duration
            Trial(response_deadline_duration=0),  # 19: no response deadline duration
            # no quiescence period duration
            Trial(quiescence_period_duration=0),  # 20: no quiescence period duration
            # no inter trial interval
            Trial(inter_trial_interval_duration=0.5),  # 21: no inter trial interval duration
        ]

    def next(self) -> Trial | None:
        return self.trial_opts[self._idx]

    def update(self, outcome: TrialOutcome) -> None:
        self._idx += 1

from typing import Literal

from ..trial_models import Trial, TrialOutcome
from ._base import BaseTrialGeneratorSpecModel, ITrialGenerator


class IntegrationTestTrialGeneratorSpec(BaseTrialGeneratorSpecModel):
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
            ),  # 7: both reward, auto response right,
            Trial(
                p_reward_left=0.0, p_reward_right=0.0, is_auto_response_right=True
            ),  # 8: no reward, auto response right,
            # auto response left
            Trial(
                p_reward_left=1.0, p_reward_right=0.0, is_auto_response_right=False
            ),  # 9: left reward, auto response left,
            Trial(
                p_reward_left=0.0, p_reward_right=1.0, is_auto_response_right=False
            ),  # 10: right reward, auto response left,
            Trial(
                p_reward_left=0.0, p_reward_right=1.0, is_auto_response_right=False
            ),  # 11: both reward, auto response left,
            Trial(
                p_reward_left=0.0, p_reward_right=0.0, is_auto_response_right=False
            ),  # 12: no reward, auto response left,
            # fast retract
            Trial(enable_fast_retract=True),  # 13: enable fast retract
            # secondary reinforcer
            # Trial(secondary_reinforcer=SecondaryReinforcer()),   # 14: enable secondary reinforcer
            # no reward consumption duration
            Trial(reward_consumption_duration=0),  # 15: no reward consumption duration
            # no reward delay
            Trial(reward_delay_duration=0),  # 16: no reward delay duration
            # no response deadline duration
            Trial(response_deadline_duration=0),  # 17: no response deadline duration
            # no quiescence period duration
            Trial(quiescence_period_duration=0),  # 18: no quiescence period duration
            # no inter trial interval
            Trial(inter_trial_interval_duration=0.5),  # 19: no inter trial interval duration
        ]

    def next(self) -> Trial | None:
        if self._idx >= len(self.trial_opts):
            return None
        return self.trial_opts[self._idx]

    def update(self, outcome: TrialOutcome) -> None:
        self._idx += 1

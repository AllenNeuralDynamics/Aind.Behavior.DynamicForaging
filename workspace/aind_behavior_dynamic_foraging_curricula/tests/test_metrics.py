import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, PropertyMock, patch

from aind_behavior_dynamic_foraging_curricula.metrics import (
    metrics_from_dataset,
)


def _make_trial(
    is_right_choice: Optional[bool],
    is_rewarded: bool,
    p_reward_right: float,
    p_reward_left: float,
    is_auto_response_right: Optional[bool] = False,
) -> dict:
    return {
        "is_right_choice": is_right_choice,
        "is_rewarded": is_rewarded,
        "trial": {
            "p_reward_right": p_reward_right,
            "p_reward_left": p_reward_left,
            "is_auto_response_right": is_auto_response_right,
        },
    }


def _patch_dataset(
    trials: list[dict], is_baiting: bool = True, prev_metrics: Optional[dict] = None, stage_name: str = "stage_1"
):
    """Patch df_foraging_dataset with a mock matching the access pattern in metrics_from_dataset."""

    # software events
    mock_trial_spec = MagicMock()
    mock_trial_spec.data = {"data": MagicMock()}
    mock_trial_spec.data["data"].iloc.__getitem__ = MagicMock(return_value={"is_baiting": is_baiting})
    mock_trial_spec.data["data"].iloc[-1] = {"is_baiting": is_baiting}

    # trial outcomes
    mock_trial_outcome = MagicMock()
    mock_trial_outcome.data = {"data": MagicMock(iloc=trials)}

    # trial generator
    mock_software_events = MagicMock()
    mock_software_events.__getitem__ = MagicMock(
        side_effect=lambda key: mock_trial_spec if key == "TrialGeneratorSpec" else mock_trial_outcome
    )

    # trainer state
    mock_trainer_state = MagicMock(**{"data.stage.name": stage_name})

    # previous metrics
    mock_previous_metrics = MagicMock()
    if prev_metrics is None:
        type(mock_previous_metrics).data = PropertyMock(side_effect=FileNotFoundError)
    else:
        mock_previous_metrics.data = prev_metrics

    mock_behavior = MagicMock()
    mock_behavior.__getitem__ = MagicMock(
        side_effect=lambda key: {
            "SoftwareEvents": mock_software_events,
            "TrainerState": mock_trainer_state,
            "PreviousMetrics": mock_previous_metrics,
        }[key]
    )

    mock_dataset = MagicMock()
    mock_dataset.__getitem__ = MagicMock(return_value=mock_behavior)
    return patch(
        "aind_behavior_dynamic_foraging_curricula.metrics.df_foraging_dataset",
        return_value=mock_dataset,
    )


class TestMetricsFromDataset(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

    def test_no_previous_metrics_total_sessions_is_one(self):
        trials = [_make_trial(True, True, 0.7, 0.3)]
        with _patch_dataset(trials):
            result = metrics_from_dataset(self.tmp_path)
        self.assertEqual(result.total_sessions, 1)

    def test_no_previous_metrics_consecutive_sessions_is_one(self):
        trials = [_make_trial(True, True, 0.7, 0.3)]
        with _patch_dataset(trials):
            result = metrics_from_dataset(self.tmp_path)
        self.assertEqual(result.consecutive_sessions_at_current_stage, 1)

    def test_previous_metrics_accumulate(self):
        trials = [_make_trial(True, True, 0.7, 0.3)]
        metrics = {
            "foraging_efficiency_per_session": [0.5],
            "unignored_trials_per_session": [10],
            "total_sessions": 1,
            "consecutive_sessions_at_current_stage": 1,
            "stage_name": "stage_1_warmup",
        }
        with _patch_dataset(trials, prev_metrics=metrics):
            result = metrics_from_dataset(self.tmp_path)
        self.assertEqual(result.total_sessions, 2)
        self.assertEqual(len(result.foraging_efficiency_per_session), 2)
        self.assertEqual(len(result.unignored_trials_per_session), 2)


if __name__ == "__main__":
    unittest.main()

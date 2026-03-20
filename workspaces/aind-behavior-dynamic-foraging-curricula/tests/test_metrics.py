import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from aind_behavior_dynamic_foraging_curricula.metrics import (
    DynamicForagingMetrics,
    metrics_from_dataset,
)


def _make_trial(is_right_choice, is_rewarded, p_reward_right, p_reward_left, is_auto_response_right=False):
    return {
        "is_right_choice": is_right_choice,
        "is_rewarded": is_rewarded,
        "trial": {
            "p_reward_right": p_reward_right,
            "p_reward_left": p_reward_left,
            "is_auto_response_right": is_auto_response_right,
        },
    }


def _make_previous_metrics(tmp_path, **kwargs) -> Path:
    defaults = dict(
        foraging_efficiency_per_session=[0.6],
        unignored_trials_per_session=[150],
        total_sessions=1,
        consecutive_sessions_at_current_stage=1,
    )
    defaults.update(kwargs)
    previous = DynamicForagingMetrics(**defaults)
    prev_path = tmp_path / "previous_metrics.json"
    prev_path.write_text(previous.model_dump_json())
    return prev_path


def _patch_dataset(trials, is_baiting=True):
    """Patch df_foraging_dataset with a mock matching the access pattern in metrics_from_dataset."""
    mock_trial_spec = MagicMock()
    mock_trial_spec.data = {"data": MagicMock()}
    mock_trial_spec.data["data"].iloc.__getitem__ = MagicMock(return_value={"is_baiting": is_baiting})
    mock_trial_spec.data["data"].iloc[-1] = {"is_baiting": is_baiting}

    mock_trial_outcome = MagicMock()
    mock_trial_outcome.data = {"data": MagicMock(iloc=trials)}

    mock_software_events = MagicMock()
    mock_software_events.__getitem__ = MagicMock(
        side_effect=lambda key: mock_trial_spec if key == "TrialGeneratorSpec" else mock_trial_outcome
    )

    mock_dataset = MagicMock()
    mock_dataset.__getitem__ = MagicMock(
        side_effect=lambda key: MagicMock(__getitem__=MagicMock(return_value=mock_software_events))
    )

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

        prev_path = _make_previous_metrics(self.tmp_path)
        trials = [_make_trial(True, True, 0.7, 0.3)]
        with _patch_dataset(trials):
            result = metrics_from_dataset(self.tmp_path, previous_metrics=prev_path)
        self.assertEqual(result.total_sessions, 2)
        self.assertEqual(len(result.foraging_efficiency_per_session), 2)
        self.assertEqual(len(result.unignored_trials_per_session), 2)

    def test_stage_changed_resets_consecutive_sessions(self):

        prev_path = _make_previous_metrics(self.tmp_path, consecutive_sessions_at_current_stage=3)
        trials = [_make_trial(True, True, 0.7, 0.3)]
        with _patch_dataset(trials):
            result = metrics_from_dataset(self.tmp_path, previous_metrics=prev_path, stage_changed=True)
        self.assertEqual(result.consecutive_sessions_at_current_stage, 1)

    def test_stage_not_changed_increments_consecutive_sessions(self):

        prev_path = _make_previous_metrics(self.tmp_path, consecutive_sessions_at_current_stage=3)
        trials = [_make_trial(True, True, 0.7, 0.3)]
        with _patch_dataset(trials):
            result = metrics_from_dataset(self.tmp_path, previous_metrics=prev_path, stage_changed=False)
        self.assertEqual(result.consecutive_sessions_at_current_stage, 4)

    def test_foraging_efficiency_is_finite_and_positive(self):
        trials = [
            _make_trial(True, True, 0.7, 0.3),
            _make_trial(True, False, 0.7, 0.3),
            _make_trial(False, True, 0.7, 0.3),
        ]
        with _patch_dataset(trials):
            result = metrics_from_dataset(self.tmp_path)
        self.assertGreater(result.foraging_efficiency_per_session[-1], 0)
        self.assertTrue(np.isfinite(result.foraging_efficiency_per_session[-1]))


if __name__ == "__main__":
    unittest.main()

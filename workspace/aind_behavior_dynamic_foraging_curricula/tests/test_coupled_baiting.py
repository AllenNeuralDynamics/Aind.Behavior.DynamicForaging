import unittest

from aind_behavior_dynamic_foraging_curricula.coupled_baiting import CURRICULUM, TRAINER
from aind_behavior_dynamic_foraging_curricula.coupled_baiting.stages import (
    make_s_stage_1,
    make_s_stage_1_warmup,
    make_s_stage_2,
    make_s_stage_3,
    make_s_stage_final,
    make_s_stage_graduated,
)
from aind_behavior_dynamic_foraging_curricula.metrics import DynamicForagingMetrics


def make_metrics(
    foraging_efficiency_per_session: list[float] = None,
    unignored_trials_per_session: list[int] = None,
    total_sessions: int = 1,
    consecutive_sessions_at_current_stage: int = 1,
    stage_name: str = "stage_1_warmup",
) -> DynamicForagingMetrics:
    return DynamicForagingMetrics(
        foraging_efficiency_per_session=foraging_efficiency_per_session or [0.0],
        unignored_trials_per_session=unignored_trials_per_session or [0],
        total_sessions=total_sessions,
        consecutive_sessions_at_current_stage=consecutive_sessions_at_current_stage,
        stage_name=stage_name,
    )


class TestCurriculumStructure(unittest.TestCase):
    def test_all_stages_in_curriculum(self):
        stages = CURRICULUM.see_stages()
        stage_names = [s.name for s in stages]
        self.assertIn("stage_1_warmup", stage_names)
        self.assertIn("stage_1", stage_names)
        self.assertIn("stage_2", stage_names)
        self.assertIn("stage_3", stage_names)
        self.assertIn("final", stage_names)
        self.assertIn("graduated", stage_names)

    def test_enrollment_starts_at_stage_1_warmup(self):
        trainer_state = TRAINER.create_enrollment()
        self.assertEqual(trainer_state.stage.name, "stage_1_warmup")


class TestWarmupTransitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_1_warmup())

    def test_warmup_to_stage_2_on_good_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[250], foraging_efficiency_per_session=[0.65], stage_name="stage_1_warmup"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")

    def test_warmup_to_stage_1_after_first_session(self):
        metrics = make_metrics(
            unignored_trials_per_session=[100],
            foraging_efficiency_per_session=[0.4],
            consecutive_sessions_at_current_stage=1,
            stage_name="stage_1_warmup",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1")


class TestStage1Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_1())

    def test_stage_1_to_stage_2_on_good_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[200], foraging_efficiency_per_session=[0.6], stage_name="stage_1"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")

    def test_stage_1_no_transition_on_poor_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[100], foraging_efficiency_per_session=[0.4], stage_name="stage_1"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1")


class TestStage2Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_2())

    def test_stage_2_to_stage_3_on_good_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[300], foraging_efficiency_per_session=[0.65], stage_name="stage_2"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_3")

    def test_stage_2_rollback_to_stage_1_on_poor_trials(self):
        metrics = make_metrics(
            unignored_trials_per_session=[150], foraging_efficiency_per_session=[0.6], stage_name="stage_2"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1")

    def test_stage_2_rollback_to_stage_1_on_poor_efficiency(self):
        metrics = make_metrics(
            unignored_trials_per_session=[199], foraging_efficiency_per_session=[0.5], stage_name="stage_2"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1")

    def test_stage_2_no_transition_on_middle_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[250], foraging_efficiency_per_session=[0.6], stage_name="stage_2"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")


class TestStage3Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_3())

    def test_stage_3_to_final_on_good_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[400], foraging_efficiency_per_session=[0.7], stage_name="stage_3"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "final")

    def test_stage_3_rollback_to_stage_2_on_poor_trials(self):
        metrics = make_metrics(
            unignored_trials_per_session=[250], foraging_efficiency_per_session=[0.7], stage_name="stage_3"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")

    def test_stage_3_rollback_to_stage_2_on_poor_efficiency(self):
        metrics = make_metrics(
            unignored_trials_per_session=[299], foraging_efficiency_per_session=[0.6], stage_name="stage_3"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")

    def test_stage_3_no_transition_on_middle_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[350], foraging_efficiency_per_session=[0.67], stage_name="stage_3"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_3")


class TestFinalTransitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_final())

    def test_final_to_graduated_on_excellent_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[450] * 5,
            foraging_efficiency_per_session=[0.70] * 5,
            total_sessions=10,
            consecutive_sessions_at_current_stage=5,
            stage_name="final",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "graduated")

    def test_final_rollback_to_stage_3_on_poor_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[250] * 5,
            foraging_efficiency_per_session=[0.55] * 5,
            total_sessions=10,
            consecutive_sessions_at_current_stage=5,
            stage_name="final",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_3")

    def test_final_no_graduation_without_enough_sessions(self):
        metrics = make_metrics(
            unignored_trials_per_session=[450] * 5,
            foraging_efficiency_per_session=[0.70] * 5,
            total_sessions=5,
            consecutive_sessions_at_current_stage=3,
            stage_name="final",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertNotEqual(updated.stage.name, "graduated")

    def test_graduated_is_absorbing(self):
        trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_graduated())
        metrics = make_metrics(
            unignored_trials_per_session=[500] * 5,
            foraging_efficiency_per_session=[0.9] * 5,
            total_sessions=20,
            consecutive_sessions_at_current_stage=10,
            stage_name="final",
        )
        updated = TRAINER.evaluate(trainer_state, metrics)
        self.assertEqual(updated.stage.name, "graduated")


if __name__ == "__main__":
    unittest.main()

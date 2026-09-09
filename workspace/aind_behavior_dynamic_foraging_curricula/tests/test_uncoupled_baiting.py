import unittest

from aind_behavior_dynamic_foraging_curricula.metrics import DynamicForagingMetrics
from aind_behavior_dynamic_foraging_curricula.uncoupled_baiting import CURRICULUM, TRAINER
from aind_behavior_dynamic_foraging_curricula.uncoupled_baiting.stages import (
    make_s_stage_1,
    make_s_stage_1_warmup,
    make_s_stage_2,
    make_s_stage_3,
    make_s_stage_final,
    make_s_stage_graduated,
)


def make_metrics(
    foraging_efficiency_per_session: list[float] = None,
    unignored_trials_per_session: list[int] = None,
    total_sessions: int = 1,
    consecutive_sessions_at_current_stage: int = 1,
    stage_name: str = "STAGE_1_WARMUP",
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
        self.assertIn("STAGE_1_WARMUP", stage_names)
        self.assertIn("STAGE_1", stage_names)
        self.assertIn("STAGE_2", stage_names)
        self.assertIn("STAGE_3", stage_names)
        self.assertIn("STAGE_FINAL", stage_names)
        self.assertIn("GRADUATED", stage_names)

    def test_enrollment_starts_at_stage_1_warmup(self):
        trainer_state = TRAINER.create_enrollment()
        self.assertEqual(trainer_state.stage.name, "STAGE_1_WARMUP")


class TestWarmupTransitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_1_warmup())

    def test_warmup_to_stage_2_on_good_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[250], foraging_efficiency_per_session=[0.65], stage_name="STAGE_1_WARMUP"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_2")

    def test_warmup_to_stage_1_after_first_session(self):
        metrics = make_metrics(
            unignored_trials_per_session=[100],
            foraging_efficiency_per_session=[0.4],
            consecutive_sessions_at_current_stage=1,
            stage_name="STAGE_1_WARMUP",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_1")


class TestStage1Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_1())

    def test_stage_1_to_stage_2_on_good_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[200], foraging_efficiency_per_session=[0.6], stage_name="STAGE_1"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_2")

    def test_stage_1_no_transition_on_poor_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[100], foraging_efficiency_per_session=[0.4], stage_name="STAGE_1"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_1")


class TestStage2Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_2())

    def test_stage_2_to_stage_3_on_good_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[300],
            foraging_efficiency_per_session=[0.65],
            consecutive_sessions_at_current_stage=3,
            stage_name="STAGE_2",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_3")

    def test_stage_2_requires_two_sessions_before_stage_3(self):
        metrics = make_metrics(
            unignored_trials_per_session=[300],
            foraging_efficiency_per_session=[0.65],
            consecutive_sessions_at_current_stage=1,
            stage_name="STAGE_2",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_2")

    def test_stage_2_rollback_to_stage_1_on_poor_trials(self):
        metrics = make_metrics(
            unignored_trials_per_session=[150], foraging_efficiency_per_session=[0.6], stage_name="STAGE_2"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_1")

    def test_stage_2_rollback_to_stage_1_on_poor_efficiency(self):
        metrics = make_metrics(
            unignored_trials_per_session=[199], foraging_efficiency_per_session=[0.5], stage_name="STAGE_2"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_1")

    def test_stage_2_no_transition_on_middle_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[250], foraging_efficiency_per_session=[0.6], stage_name="STAGE_2"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_2")


class TestStage3Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_3())

    def test_stage_3_to_final_one_trial_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[400], foraging_efficiency_per_session=[0.7], stage_name="STAGE_3"
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_FINAL")


class TestFinalTransitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_final())

    def test_final_to_graduated_on_excellent_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[450] * 5,
            foraging_efficiency_per_session=[0.70] * 5,
            total_sessions=10,
            consecutive_sessions_at_current_stage=5,
            stage_name="STAGE_FINAL",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "GRADUATED")

    def test_final_rollback_to_stage_3_on_poor_performance(self):
        metrics = make_metrics(
            unignored_trials_per_session=[250] * 5,
            foraging_efficiency_per_session=[0.55] * 5,
            total_sessions=10,
            consecutive_sessions_at_current_stage=5,
            stage_name="STAGE_FINAL",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "STAGE_3")

    def test_final_no_graduation_without_enough_sessions(self):
        metrics = make_metrics(
            unignored_trials_per_session=[450] * 5,
            foraging_efficiency_per_session=[0.70] * 5,
            total_sessions=5,
            consecutive_sessions_at_current_stage=3,
            stage_name="STAGE_FINAL",
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertNotEqual(updated.stage.name, "GRADUATED")

    def test_graduated_is_absorbing(self):
        trainer_state = TRAINER.create_trainer_state(stage=make_s_stage_graduated())
        metrics = make_metrics(
            unignored_trials_per_session=[500] * 5,
            foraging_efficiency_per_session=[0.9] * 5,
            total_sessions=20,
            consecutive_sessions_at_current_stage=10,
            stage_name="GRADUATED",
        )
        updated = TRAINER.evaluate(trainer_state, metrics)
        self.assertEqual(updated.stage.name, "GRADUATED")


if __name__ == "__main__":
    unittest.main()

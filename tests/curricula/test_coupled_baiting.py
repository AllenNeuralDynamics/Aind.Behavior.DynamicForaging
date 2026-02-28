import unittest

from aind_behavior_dynamic_foraging.curricula.coupled_baiting import CURRICULUM, TRAINER
from aind_behavior_dynamic_foraging.curricula.coupled_baiting.stages import (
    s_final,
    s_graduated,
    s_stage_1,
    s_stage_1_warmup,
    s_stage_2,
    s_stage_3,
)
from aind_behavior_dynamic_foraging.curricula.metrics import DynamicForagingMetrics


def make_metrics(
    finished_trials: list[int] = None,
    foraging_efficiency: list[float] = None,
    session_total: int = 1,
    session_at_current_stage: int = 1,
) -> DynamicForagingMetrics:
    return DynamicForagingMetrics(
        finished_trials=finished_trials or [0],
        foraging_efficiency=foraging_efficiency or [0.0],
        session_total=session_total,
        session_at_current_stage=session_at_current_stage,
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
        self.trainer_state = TRAINER.create_trainer_state(stage=s_stage_1_warmup)

    def test_warmup_to_stage_2_on_good_performance(self):
        metrics = make_metrics(finished_trials=[250], foraging_efficiency=[0.65])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")

    def test_warmup_to_stage_1_after_first_session(self):
        # does not meet stage_2 criteria, but has completed a session
        metrics = make_metrics(finished_trials=[100], foraging_efficiency=[0.4], session_at_current_stage=1)
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1")

    def test_warmup_no_transition_on_first_session_bad_performance(self):
        metrics = make_metrics(finished_trials=[100], foraging_efficiency=[0.4], session_at_current_stage=0)
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1_warmup")


class TestStage1Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=s_stage_1)

    def test_stage_1_to_stage_2_on_good_performance(self):
        metrics = make_metrics(finished_trials=[200], foraging_efficiency=[0.6])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")

    def test_stage_1_no_transition_on_poor_performance(self):
        metrics = make_metrics(finished_trials=[100], foraging_efficiency=[0.4])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1")


class TestStage2Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=s_stage_2)

    def test_stage_2_to_stage_3_on_good_performance(self):
        metrics = make_metrics(finished_trials=[300], foraging_efficiency=[0.65])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_3")

    def test_stage_2_rollback_to_stage_1_on_poor_trials(self):
        metrics = make_metrics(finished_trials=[150], foraging_efficiency=[0.6])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1")

    def test_stage_2_rollback_to_stage_1_on_poor_efficiency(self):
        metrics = make_metrics(finished_trials=[250], foraging_efficiency=[0.5])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_1")

    def test_stage_2_no_transition_on_middle_performance(self):
        metrics = make_metrics(finished_trials=[250], foraging_efficiency=[0.6])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")


class TestStage3Transitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=s_stage_3)

    def test_stage_3_to_final_on_good_performance(self):
        metrics = make_metrics(finished_trials=[400], foraging_efficiency=[0.7])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "final")

    def test_stage_3_rollback_to_stage_2_on_poor_trials(self):
        metrics = make_metrics(finished_trials=[250], foraging_efficiency=[0.7])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")

    def test_stage_3_rollback_to_stage_2_on_poor_efficiency(self):
        metrics = make_metrics(finished_trials=[400], foraging_efficiency=[0.6])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_2")

    def test_stage_3_no_transition_on_middle_performance(self):
        metrics = make_metrics(finished_trials=[350], foraging_efficiency=[0.67])
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_3")


class TestFinalTransitions(unittest.TestCase):
    def setUp(self):
        self.trainer_state = TRAINER.create_trainer_state(stage=s_final)

    def test_final_to_graduated_on_excellent_performance(self):
        metrics = make_metrics(
            finished_trials=[450] * 5,
            foraging_efficiency=[0.70] * 5,
            session_total=10,
            session_at_current_stage=5,
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "graduated")

    def test_final_rollback_to_stage_3_on_poor_performance(self):
        metrics = make_metrics(
            finished_trials=[250] * 5,
            foraging_efficiency=[0.55] * 5,
            session_total=10,
            session_at_current_stage=5,
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertEqual(updated.stage.name, "stage_3")

    def test_final_no_graduation_without_enough_sessions(self):
        metrics = make_metrics(
            finished_trials=[450] * 5,
            foraging_efficiency=[0.70] * 5,
            session_total=5,  # not enough total sessions
            session_at_current_stage=3,  # not enough at final
        )
        updated = TRAINER.evaluate(self.trainer_state, metrics)
        self.assertNotEqual(updated.stage.name, "graduated")

    def test_graduated_is_absorbing(self):
        trainer_state = TRAINER.create_trainer_state(stage=s_graduated)
        metrics = make_metrics(
            finished_trials=[500] * 5,
            foraging_efficiency=[0.9] * 5,
            session_total=20,
            session_at_current_stage=10,
        )
        updated = TRAINER.evaluate(trainer_state, metrics)
        self.assertEqual(updated.stage.name, "graduated")


if __name__ == "__main__":
    unittest.main()

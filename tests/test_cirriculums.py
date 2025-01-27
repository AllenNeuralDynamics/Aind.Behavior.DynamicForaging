import unittest
from aind_auto_train.curriculum_manager import CurriculumManager
from aind_auto_train.schema.task import DynamicForagingMetrics, TrainingStage, DynamicForagingParas
from aind_auto_train.schema.curriculum import Curriculum as AutoTrainCurriculum

from aind_behavior_dynamic_foraging.CurriculumManager.curriculums.coupled_baiting_2p3 import (
    construct_coupled_baiting_2p3_curriculum,
    s_stage_1_warmup as cb_stage_1_warmup,
    s_stage_1 as cb_stage_1,
    s_stage_2 as cb_stage_2,
    s_stage_3 as cb_stage_3,
    s_final as cb_final,
    s_graduated as cb_graduated
)
from aind_behavior_dynamic_foraging.CurriculumManager.curriculums.uncoupled_baiting_2p3 import (
    construct_uncoupled_baiting_2p3_curriculum,
    s_stage_1_warmup as ucb_stage_1_warmup,
    s_stage_1 as ucb_stage_1,
    s_stage_2 as ucb_stage_2,
    s_stage_3 as ucb_stage_3,
    s_final as ucb_final,
    s_graduated as ucb_graduated
)
from aind_behavior_dynamic_foraging.CurriculumManager.curriculums.uncoupled_no_baiting_2p3p1rwdDelay159 import (
    construct_uncoupled_no_baiting_2p3p1_reward_delay_curriculum,
    s_stage_1_warmup as uc_stage_1_warmup,
    s_stage_1 as uc_stage_1,
    s_stage_2 as uc_stage_2,
    s_stage_3 as uc_stage_3,
    s_stage_4 as uc_stage_4,
    s_final as uc_final,
    s_graduated as uc_graduated
)
from aind_behavior_dynamic_foraging.DataSchemas.task_logic import AindDynamicForagingTaskParameters
from aind_behavior_curriculum.trainer import Trainer, TrainerState
from aind_behavior_curriculum import (
    Stage,
    Curriculum as AINDCurriculum
)
from tests.mock_databases import MockCurriculumManager

class TestCurriculums(unittest.TestCase):
    """ Testing aind-behavior-curriculum against aind-auto-train"""

    curriculum_manager: CurriculumManager or MockCurriculumManager

    def setUp(self) -> None:
        """
        Create curriculum manager
        """

        try:
            self.curriculum_manager = CurriculumManager(
                saved_curriculums_on_s3=dict(
                    bucket='aind-behavior-data',
                    root='foraging_auto_training/saved_curriculums/'
                ),
                saved_curriculums_local='/root/capsule/scratch/tmp/'
            )
        except:  # use resource curriculums if error using s3
            self.curriculum_manager = MockCurriculumManager()
        
    def test_coupled_baiting(self):
        """
        Test coupled baiting task
        """

        coupled_baiting = self.curriculum_manager.get_curriculum(
            curriculum_name='Coupled Baiting',
            curriculum_version='2.3',
            curriculum_schema_version='1.0',
        )
        old_curriculum = coupled_baiting['curriculum']

        new_curriculum = construct_coupled_baiting_2p3_curriculum()

        # --WARMUP--

        # Check warmup stay conditions
        warmup_stay = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=0,
            foraging_efficiency=[0.0],
            finished_trials=[0]
        )
        self.compare_decision(metrics=warmup_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=cb_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1_WARMUP,
                              new_next_stage=cb_stage_1_warmup)

        # test warmup transition to stage 1 conditions with above foraging efficiency
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.8],
            finished_trials=[10]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=cb_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=cb_stage_1)

        # test warmup transition to stage 1 conditions with above finished_trials
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.1],
            finished_trials=[300]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=cb_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=cb_stage_1)

        # test warmup transition to stage 2 conditions
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.8],
            finished_trials=[300]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=cb_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=cb_stage_2)

        # --STAGE 1--
        # test stage 1 stay
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.3],
            finished_trials=[10, 100, 150]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=cb_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=cb_stage_1)

        # test stage 1 stay above foraging efficiency
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.6],
            finished_trials=[10, 100, 150]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=cb_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=cb_stage_1)

        # test stage 1 stay above finished trials
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.3],
            finished_trials=[10, 100, 200]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=cb_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=cb_stage_1)

        # test stage 1 transition
        stage_1_transition = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.6],
            finished_trials=[10, 100, 200]
        )
        self.compare_decision(metrics=stage_1_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=cb_stage_1,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=cb_stage_2)

        # --STAGE 2--
        # test stage 2 stay
        stage_2_stay = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=cb_stage_2,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=cb_stage_2)

        # test stage 2 stay with above foraging efficiency
        stage_2_stay = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=cb_stage_2,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=cb_stage_2)

        # test stage 2 stay with above finished trials
        stage_2_stay = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 300]
        )
        self.compare_decision(metrics=stage_2_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=cb_stage_2,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=cb_stage_2)

        # test stage 2 transition
        stage_2_transition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65],
            finished_trials=[10, 100, 200, 300]
        )
        self.compare_decision(metrics=stage_2_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=cb_stage_2,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=cb_stage_3)

        # test stage 2 de-transition
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .54],
            finished_trials=[10, 100, 200, 199]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=cb_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=cb_stage_1)

        # test stage 2 de-transition below foraging efficiency
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .54],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=cb_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=cb_stage_1)

        # test stage 2 de-transition below finished trials
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 199]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=cb_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=cb_stage_1)

        # --STAGE 3--
        # test stage 3 stay
        stage_3_stay = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .66],
            finished_trials=[10, 100, 200, 300, 350]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=cb_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=cb_stage_3)

        # test stage 3 stay with above foraging efficiency
        stage_3_stay = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .7],
            finished_trials=[10, 100, 200, 300, 350]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=cb_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=cb_stage_3)

        # test stage 3 stay with above finished trials
        stage_3_stay = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .66],
            finished_trials=[10, 100, 200, 300, 400]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=cb_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=cb_stage_3)

        # test stage 3 transition
        stage_3_transition = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .7],
            finished_trials=[10, 100, 200, 300, 400]
        )
        self.compare_decision(metrics=stage_3_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=cb_stage_3,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=cb_final)

        # test stage 3 de-transition
        stage_3_detransition = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64],
            finished_trials=[10, 100, 200, 300, 299]
        )
        self.compare_decision(metrics=stage_3_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=cb_stage_3,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=cb_stage_2)

        # test stage 3 de-transition with below foraging efficiency
        stage_3_detransition = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64],
            finished_trials=[10, 100, 200, 300, 300]
        )
        self.compare_decision(metrics=stage_3_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=cb_stage_3,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=cb_stage_2)

        # test stage 3 de-transition with below finished trials
        stage_3_detransition = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65],
            finished_trials=[10, 100, 200, 300, 299]
        )
        self.compare_decision(metrics=stage_3_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=cb_stage_3,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=cb_stage_2)

        # --FINAL--
        # test final stay
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .60, .60, .60, .60, .60],
            finished_trials=[10, 100, 200, 300, 400, 300, 300, 300, 300]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=cb_final)

        # test final stay with above foraging efficiency
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=cb_final)

        # test final stay with above finished trials
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 500, 500, 500, 500]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=cb_final)

        # test final stay with above session total
        final_stay = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 425, 425, 425, 425, 425]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=cb_final)

        # test final stay with above session at current stage
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=cb_final)

        # test final transition
        final_transition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 500, 500, 500, 500, 500]
        )
        self.compare_decision(metrics=final_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.GRADUATED,
                              new_next_stage=cb_graduated)

        # test final de-transition
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .59, .59, .59, .59, .59],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 299, 299, 299, 299, 299]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=cb_stage_3)

        # test final de-transition below foraging efficiency
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .59, .59, .59, .59, .59],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 500, 500, 500, 500, 500]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=cb_stage_3)

        # test final de-transition below finished trials
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 299, 299, 299, 299, 299]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=cb_final,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=cb_stage_3)

    def test_uncoupled_baiting(self):
        """
        Test uncoupled baiting task
        """

        uncoupled_baiting = self.curriculum_manager.get_curriculum(
            curriculum_name='Uncoupled Baiting',
            curriculum_version='2.3',
            curriculum_schema_version='1.0',
        )
        old_curriculum = uncoupled_baiting['curriculum']

        new_curriculum = construct_uncoupled_baiting_2p3_curriculum()

        # --WARMUP--

        # Check warmup stay conditions
        warmup_stay = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=0,
            foraging_efficiency=[0.0],
            finished_trials=[0]
        )
        self.compare_decision(metrics=warmup_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=ucb_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1_WARMUP,
                              new_next_stage=ucb_stage_1_warmup)

        # test warmup transition to stage 1 conditions above foraging efficiency
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.8],
            finished_trials=[10]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=ucb_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=ucb_stage_1)

        # test warmup transition to stage 1 conditions above finished_trials
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.1],
            finished_trials=[300]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=ucb_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=ucb_stage_1)

        # test warmup transition to stage 2 conditions
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.6],
            finished_trials=[200]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=ucb_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=ucb_stage_2)

        # --STAGE 1--
        # test stage 1 stay
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.3],
            finished_trials=[10, 100, 150]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=ucb_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=ucb_stage_1)

        # test stage 1 stay above foraging efficiency
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.6],
            finished_trials=[10, 100, 150]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=ucb_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=ucb_stage_1)

        # test stage 1 stay above finished trials
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.3],
            finished_trials=[10, 100, 200]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=ucb_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=ucb_stage_1)

        # test stage 1 transition
        stage_1_transition = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.6],
            finished_trials=[10, 100, 200]
        )
        self.compare_decision(metrics=stage_1_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=ucb_stage_1,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=ucb_stage_2)

        # --STAGE 2--
        # test stage 2 stay
        stage_2_stay = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=ucb_stage_2,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=ucb_stage_2)

        # test stage 2 stay with above foraging efficiency
        stage_2_stay = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=ucb_stage_2,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=ucb_stage_2)

        # test stage 2 stay with above finished trials
        stage_2_stay = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 300]
        )
        self.compare_decision(metrics=stage_2_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=ucb_stage_2,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=ucb_stage_2)

        # test stage 2 stay with above session at current stage
        stage_2_stay = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=ucb_stage_2,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=ucb_stage_2)

        # test stage 2 transition
        stage_2_transition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.7, .65],
            finished_trials=[10, 100, 200, 300]
        )
        self.compare_decision(metrics=stage_2_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=ucb_stage_2,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=ucb_stage_3)

        # test stage 2 de-transition
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .54],
            finished_trials=[10, 100, 200, 199]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=ucb_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=ucb_stage_1)

        # test stage 2 de-transition below foraging efficiency
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .54],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=ucb_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=ucb_stage_1)

        # test stage 2 de-transition below finished trials
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 199]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=ucb_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=ucb_stage_1)

        # --STAGE 3--
        # test stage 3 stay
        stage_3_stay = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=0,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .66],
            finished_trials=[10, 100, 200, 300, 350]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=ucb_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=ucb_stage_3)

        # test stage 3 transition
        stage_3_stay = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .7],
            finished_trials=[10, 100, 200, 300, 350]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=ucb_stage_3,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=ucb_final)

        # --FINAL--
        # test final stay
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .60, .60, .60, .60, .60],
            finished_trials=[10, 100, 200, 300, 400, 300, 300, 300, 300]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=ucb_final)

        # test final stay with above foraging efficiency
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=ucb_final)

        # test final stay with above finished trials
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 500, 500, 500, 500]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=ucb_final)

        # test final stay with above session total
        final_stay = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 425, 425, 425, 425, 425]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=ucb_final)

        # test final stay with above session at current stage
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=ucb_final)

        # test final transition
        final_transition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 500, 500, 500, 500, 500]
        )
        self.compare_decision(metrics=final_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.GRADUATED,
                              new_next_stage=ucb_graduated)

        # test final de-transition
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .59, .59, .59, .59, .59],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 299, 299, 299, 299, 299]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=ucb_stage_3)

        # test final de-transition below foraging efficiency
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .59, .59, .59, .59, .59],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 500, 500, 500, 500, 500]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=ucb_stage_3)

        # test final de-transition below finished trials
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 299, 299, 299, 299, 299]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=ucb_final,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=ucb_stage_3)

    def test_uncoupled_no_baiting(self):
        """
        Test uncoupled no baiting task
        """

        uncoupled_baiting = self.curriculum_manager.get_curriculum(
            curriculum_name='Uncoupled Without Baiting',
            curriculum_version='2.3.1rwdDelay159',
            curriculum_schema_version='1.0',
        )
        old_curriculum = uncoupled_baiting['curriculum']

        new_curriculum = construct_uncoupled_no_baiting_2p3p1_reward_delay_curriculum()

        # --WARMUP--

        # Check warmup stay conditions
        warmup_stay = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=0,
            foraging_efficiency=[0.0],
            finished_trials=[0]
        )
        self.compare_decision(metrics=warmup_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=uc_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1_WARMUP,
                              new_next_stage=uc_stage_1_warmup)

        # test warmup transition to stage 1 conditions above foraging efficiency
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.8],
            finished_trials=[10]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=uc_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=uc_stage_1)

        # test warmup transition to stage 1 conditions above finished_trials
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.1],
            finished_trials=[300]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=uc_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=uc_stage_1)

        # test warmup transition to stage 2 conditions
        warmup_transition = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=1,
            foraging_efficiency=[0.6],
            finished_trials=[200]
        )
        self.compare_decision(metrics=warmup_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1_WARMUP,
                              new_stage=uc_stage_1_warmup,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=uc_stage_2)

        # --STAGE 1--
        # test stage 1 stay
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.3],
            finished_trials=[10, 100, 150]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=uc_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=uc_stage_1)

        # test stage 1 stay above foraging efficiency
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.6],
            finished_trials=[10, 100, 150]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=uc_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=uc_stage_1)

        # test stage 1 stay above finished trials
        stage_1_stay = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.3],
            finished_trials=[10, 100, 200]
        )
        self.compare_decision(metrics=stage_1_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=uc_stage_1,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=uc_stage_1)

        # test stage 1 transition
        stage_1_transition = DynamicForagingMetrics(
            session_total=3,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.6],
            finished_trials=[10, 100, 200]
        )
        self.compare_decision(metrics=stage_1_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_1,
                              new_stage=uc_stage_1,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=uc_stage_2)

        # --STAGE 2--
        # test stage 2 stay
        stage_2_stay = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=uc_stage_2,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=uc_stage_2)

        # test stage 2 transition
        stage_2_transition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=3,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65],
            finished_trials=[10, 100, 200, 300, 300]
        )
        self.compare_decision(metrics=stage_2_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=uc_stage_2,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=uc_stage_3)

        # test stage 2 de-transition
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .54],
            finished_trials=[10, 100, 200, 199]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=uc_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=uc_stage_1)

        # test stage 2 de-transition below foraging efficiency
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .54],
            finished_trials=[10, 100, 200, 200]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=uc_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=uc_stage_1)

        # test stage 2 de-transition below finished trials
        stage_2_detransition = DynamicForagingMetrics(
            session_total=4,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .55],
            finished_trials=[10, 100, 200, 199]
        )
        self.compare_decision(metrics=stage_2_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_2,
                              new_stage=uc_stage_2,
                              old_next_stage=TrainingStage.STAGE_1,
                              new_next_stage=uc_stage_1)

        # --STAGE 3--
        # test stage 3 stay
        stage_3_stay = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64],
            finished_trials=[10, 100, 200, 300, 299]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=uc_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=uc_stage_3)

        # test stage 3 stay above foraging efficiency
        stage_3_stay = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65],
            finished_trials=[10, 100, 200, 300, 299]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=uc_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=uc_stage_3)

        # test stage 3 stay above finished trials
        stage_3_stay = DynamicForagingMetrics(
            session_total=5,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64],
            finished_trials=[10, 100, 200, 300, 300]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=uc_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=uc_stage_3)

        # test stage 3 stay above session_at_current_stage
        stage_3_stay = DynamicForagingMetrics(
            session_total=6,
            session_at_current_stage=3,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64, .64],
            finished_trials=[10, 100, 200, 300, 299, 299]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=uc_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=uc_stage_3)

        # test stage 3 stay below session_at_current_stage, finished trials, and foraging efficiency
        stage_3_stay = DynamicForagingMetrics(
            session_total=6,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .49],
            finished_trials=[10, 100, 200, 300, 249]
        )
        self.compare_decision(metrics=stage_3_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=uc_stage_3,
                              old_next_stage=TrainingStage.STAGE_3,
                              new_next_stage=uc_stage_3)

        # test stage 3 transition
        stage_3_transition = DynamicForagingMetrics(
            session_total=6,
            session_at_current_stage=3,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64, .65],
            finished_trials=[10, 100, 200, 300, 299, 300]
        )
        self.compare_decision(metrics=stage_3_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=uc_stage_3,
                              old_next_stage=TrainingStage.STAGE_4,
                              new_next_stage=uc_stage_4)

        # test stage 3 de-transition
        stage_3_detransition = DynamicForagingMetrics(
            session_total=6,
            session_at_current_stage=3,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64, .49],
            finished_trials=[10, 100, 200, 300, 299, 249]
        )
        self.compare_decision(metrics=stage_3_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_3,
                              new_stage=uc_stage_3,
                              old_next_stage=TrainingStage.STAGE_2,
                              new_next_stage=uc_stage_2)

        # --STAGE 4__
        # test stage 4 stay
        stage_4_stay = DynamicForagingMetrics(
            session_total=7,
            session_at_current_stage=1,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64, .65, .65],
            finished_trials=[10, 100, 200, 300, 299, 300, 300]
        )
        self.compare_decision(metrics=stage_4_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_4,
                              new_stage=uc_stage_4,
                              old_next_stage=TrainingStage.STAGE_4,
                              new_next_stage=uc_stage_4)

        # test stage 4 transition
        stage_4_transition = DynamicForagingMetrics(
            session_total=8,
            session_at_current_stage=2,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .64, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 299, 300, 300, 300]
        )
        self.compare_decision(metrics=stage_4_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_4,
                              new_stage=uc_stage_4,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=uc_final)

        # --FINAL--
        # test final stay
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .60, .60, .60, .60, .60],
            finished_trials=[10, 100, 200, 300, 300, 399, 399, 399, 399]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=uc_final)

        # test final stay with above foraging efficiency
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 300, 399, 399, 399, 399]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=uc_final)

        # test final stay with above finished trials
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 400, 400, 400, 400]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=uc_final)

        # test final stay with above session total
        final_stay = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=4,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 425, 425, 425, 425, 425]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=uc_final)

        # test final stay with above session at current stage
        final_stay = DynamicForagingMetrics(
            session_total=9,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425]
        )
        self.compare_decision(metrics=final_stay,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.STAGE_FINAL,
                              new_next_stage=uc_final)

        # test final transition
        final_transition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 500, 500, 500, 500, 500]
        )
        self.compare_decision(metrics=final_transition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.GRADUATED,
                              new_next_stage=uc_graduated)

        # test final de-transition
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .59, .59, .59, .59, .59],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 249, 249, 249, 249, 249]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.STAGE_4,
                              new_next_stage=uc_stage_4)

        # test final de-transition below foraging efficiency
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .59, .59, .59, .59, .59],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 250, 250, 250, 250, 250]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.STAGE_4,
                              new_next_stage=uc_stage_4)

        # test final de-transition below finished trials
        final_detransition = DynamicForagingMetrics(
            session_total=15,
            session_at_current_stage=5,
            foraging_efficiency=[0.1, 0.2, 0.7, .65, .65, .65, .65, .65, .65, .7, .7, .7, .7, .7],
            finished_trials=[10, 100, 200, 300, 400, 425, 425, 425, 425, 425, 249, 249, 249, 249, 249]
        )
        self.compare_decision(metrics=final_detransition,
                              old_curriculum=old_curriculum,
                              new_curriculum=new_curriculum,
                              old_stage=TrainingStage.STAGE_FINAL,
                              new_stage=uc_final,
                              old_next_stage=TrainingStage.STAGE_4,
                              new_next_stage=uc_stage_4)

    def compare_decision(self, metrics: DynamicForagingMetrics,
                         old_curriculum: AutoTrainCurriculum,
                         new_curriculum: AINDCurriculum,
                         old_stage: TrainingStage,
                         new_stage: Stage,
                         old_next_stage: TrainingStage,
                         new_next_stage: Stage) -> None:
        """
        Compare decisions of curriculums based on given metrics
        :param metrics: Session metrics
        :param old_curriculum: auto train curriculum object
        :param new_curriculum: aind curriculum object
        :param old_stage: auto train stage object
        :param new_stage: aind stage object
        :param old_next_stage: auto train stage object of expected decision
        :param new_next_stage: aind stage object of expected decision
        """

        old_decision = old_curriculum.evaluate_transitions(
            current_stage=old_stage,
            metrics=metrics
        )
        new_decision = Trainer(new_curriculum).evaluate(TrainerState(stage=new_stage,
                                                                     curriculum=new_curriculum,
                                                                     is_on_curriculum=True),
                                                        metrics)

        # check that both curriculums reach the same decision
        self.assertEqual(old_decision[1], old_next_stage)
        self.assertEqual(new_decision.stage.name, new_next_stage.name)

        # check that task logic parameters match
        self.compare_task_logic(old_curriculum.parameters[old_decision[1]], new_decision.stage.get_task_parameters())

    def compare_task_logic(self, old_logic: DynamicForagingParas, new_logic: AindDynamicForagingTaskParameters) -> None:
        """
        Compare values of task logic produced by old and new curriculum
        :param old_logic:
        :param new_logic:
        """

        # check warmup parameters
        self.assertEqual(old_logic.warmup, new_logic.warmup)
        self.assertEqual(old_logic.warm_min_trial, new_logic.warm_min_trial)
        self.assertEqual(old_logic.warm_max_choice_ratio_bias, new_logic.warm_max_choice_ratio_bias)
        self.assertEqual(old_logic.warm_min_finish_ratio, new_logic.warm_min_finish_ratio)
        self.assertEqual(old_logic.warm_windowsize, new_logic.warm_windowsize)

        # check reward statistic parameters
        self.assertEqual(old_logic.BaseRewardSum, new_logic.base_reward_sum)
        self.assertEqual(old_logic.RewardFamily, new_logic.reward_family)
        self.assertEqual(old_logic.RewardPairsN, new_logic.reward_pairs_n)

        # check block parameters
        self.assertEqual(old_logic.BlockMin, new_logic.block_min)
        self.assertEqual(old_logic.BlockMax, new_logic.block_max)
        self.assertEqual(old_logic.BlockBeta, new_logic.block_beta)
        self.assertEqual(old_logic.BlockMinReward, new_logic.block_min_reward)

        # check iti parameters
        self.assertEqual(old_logic.ITIMin, new_logic.iti_min)
        self.assertEqual(old_logic.ITIMax, new_logic.iti_max)
        self.assertEqual(old_logic.ITIBeta, new_logic.iti_beta)

        # check delay parameters
        self.assertEqual(old_logic.DelayMin, new_logic.delay_min)
        self.assertEqual(old_logic.DelayMax, new_logic.delay_max)
        self.assertEqual(old_logic.DelayBeta, new_logic.delay_beta)

        # check reward size and reward delay parameters
        self.assertEqual(old_logic.RewardDelay, new_logic.reward_delay)
        self.assertEqual(old_logic.RightValue_volume, new_logic.right_value_volume)
        self.assertEqual(old_logic.LeftValue_volume, new_logic.left_value_volume)

        # check auto water parameters
        self.assertEqual(old_logic.AutoReward, new_logic.auto_reward)
        self.assertEqual(old_logic.AutoWaterType, new_logic.auto_water_type)
        self.assertEqual(old_logic.Unrewarded, new_logic.unrewarded)
        self.assertEqual(old_logic.Ignored, new_logic.ignored)
        self.assertEqual(old_logic.Multiplier, new_logic.multiplier)

        # check auto block parameters
        self.assertEqual(old_logic.AdvancedBlockAuto, new_logic.advanced_block_auto)
        self.assertEqual(old_logic.SwitchThr, new_logic.switch_thr)
        self.assertEqual(old_logic.PointsInARow, new_logic.points_in_a_row)

        # check auto stop parameters
        self.assertEqual(old_logic.MaxTrial, new_logic.max_trial)
        self.assertEqual(old_logic.MaxTime, new_logic.max_time)
        self.assertEqual(old_logic.StopIgnores, round(new_logic.auto_stop_ignore_win *
                                                      new_logic.auto_stop_ignore_ratio_threshold))

        # check miscellaneous parameters
        self.assertEqual(old_logic.ResponseTime, new_logic.response_time)
        self.assertEqual(old_logic.RewardConsumeTime, new_logic.reward_consume_time)
        self.assertEqual(old_logic.UncoupledReward, new_logic.uncoupled_reward)


if __name__ == "__main__":
    unittest.main()

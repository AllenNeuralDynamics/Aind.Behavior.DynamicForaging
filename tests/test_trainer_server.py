import unittest
from aind_behavior_dynamic_foraging import DynamicForagingTrainerServer, DynamicForagingMetrics
from tests.mock_databases import MockSlimsClient, MockDocDBClient
from aind_behavior_dynamic_foraging.CurriculumManager.curriculums.coupled_baiting_2p3 import \
    construct_coupled_baiting_2p3_curriculum


class TestTrainerServer(unittest.TestCase):
    """ Testing TrainerServer model"""

    trainer_server: DynamicForagingTrainerServer

    @classmethod
    def setUp(self):
        """
        Setup trainer server
        """
        self.slims_client = MockSlimsClient()
        self.docdb_client = MockDocDBClient()

        self.trainer_server = DynamicForagingTrainerServer(slims_client=self.slims_client,
                                                           docdb_client=self.docdb_client)

    def test_load_data(self):
        """
        Test that server can correctly load and parse data
        """

        # test load data
        curriculum, trainer_state, metrics = self.trainer_server.load_data('00000001')

        # check curriculum returned
        expected_curriculum = construct_coupled_baiting_2p3_curriculum()
        self.assertEqual(curriculum.name, expected_curriculum.name)
        self.assertEqual(curriculum.graph, expected_curriculum.graph)

        # check metrics returned
        expected_metrics = DynamicForagingMetrics(
            session_total=1,
            session_at_current_stage=0,
            foraging_efficiency=[0.6539447567017806],
            finished_trials=[483]
        )
        self.assertEqual(metrics, expected_metrics)

    def test_write_data(self):
        """
        Test that server can correctly write data
        """

        curriculum, trainer_state, metrics = self.trainer_server.load_data('00000001')

        # test write data
        self.trainer_server.write_data(
            subject_id='00000001',
            curriculum=curriculum,
            trainer_state=trainer_state)

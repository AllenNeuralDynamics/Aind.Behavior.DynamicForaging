import unittest
from aind_behavior_dynamic_foraging import DynamicForagingTrainerServer, DynamicForagingMetrics
from tests.mock_databases import MockSlimsClient, MockDocDBClient
from aind_behavior_dynamic_foraging.CurriculumManager.curriculums.coupled_baiting_2p3 import \
    construct_coupled_baiting_2p3_curriculum

class TestTrainer(unittest.TestCase):
    """ Testing Trainer model"""

    def test_trainer_server(self):
        """
        Test trainer that will connect to Slims and docdb
        """

        slims_client = MockSlimsClient()
        docdb_client = MockDocDBClient()

        trainer_server = DynamicForagingTrainerServer(slims_client=slims_client,
                                                      docdb_client=docdb_client)
        curriculum, trainer_state, metrics = trainer_server.load_data('00000001')

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

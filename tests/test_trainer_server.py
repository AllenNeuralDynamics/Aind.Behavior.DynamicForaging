import unittest
from aind_behavior_dynamic_foraging import DynamicForagingTrainerServer, DynamicForagingMetrics
from tests.mock_databases import MockSlimsClient, MockDocDBClient
from aind_behavior_dynamic_foraging.CurriculumManager.curriculums.coupled_baiting_2p3 import \
    construct_coupled_baiting_2p3_curriculum
from unittest import mock
import os

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

    @mock.patch.dict(os.environ, {"SLIMS_USERNAME": "user",
                                  "SLIMS_PASSWORD": "lemme_in"})
    def test_internal_slims_client(self):
        """
        Test creating of slims client within class
        """

        # mock slims client for success
        with mock.patch('aind_slims_api.SlimsClient', autospec=True) as MockSlimsClient:
            trainer_server = DynamicForagingTrainerServer(docdb_client=self.docdb_client)

        # mock for failed
        with self.assertRaises(Exception) as context:
            DynamicForagingTrainerServer(docdb_client=self.docdb_client)

        expected_exception = 'Exception trying to read from Slims: ' \
                             'Could not fetch entities: <!doctype html><html lang="en"><head><title>HTTP ' \
                             'Status 401 – Unauthorized</title><style type="text/css">body ' \
                             '{font-family:Tahoma,Arial,sans-serif;} h1, h2, h3, b ' \
                             '{color:white;background-color:#525D76;} h1 {font-size:22px;} h2 {font-size:16px;} ' \
                             'h3 {font-size:14px;} p {font-size:12px;} a {color:black;} .line ' \
                             '{height:1px;background-color:#525D76;border:none;}</style></head><body><h1>' \
                             'HTTP Status 401 – Unauthorized</h1></body></html>.\n' \
                             'Please check credentials:\n' \
                             'Username: user\n' \
                             'Password: lemme_in'

        self.assertEqual(expected_exception, str(context.exception))

    def test_internal_docdb_client(self):
        """
        Test creating of docdb client within class
        """

        # mock docdb client for
        with mock.patch('aind_data_access_api.document_db.MetadataDbClient', autospec=True) as MockDocDBClient:
            trainer_server = DynamicForagingTrainerServer(slims_client=self.slims_client)


    def test_load_data(self):
        """
        Test that server can correctly load and parse data
        """

        # test load data
        curriculum, trainer_state, metrics, attachments, session = self.trainer_server.load_data('00000001')

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

        curriculum, trainer_state, metrics, attachments, session = self.trainer_server.load_data('00000001')

        # test write data
        self.trainer_server.write_data(
            subject_id='00000001',
            curriculum=curriculum,
            trainer_state=trainer_state)

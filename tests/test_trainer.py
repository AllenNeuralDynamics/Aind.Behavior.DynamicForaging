import unittest
from aind_behavior_dynamic_foraging import DynamicForagingTrainerServer

class TestTrainer(unittest.TestCase):
    """ Testing Trainer model"""

    def test_trainer_server(self):
        """
        Test trainer that will connect to Slims and docdb
        """

        trainer_server = DynamicForagingTrainerServer()
        curriculum, trainer_state, metrics = trainer_server.load_data('00000001')
from aind_behavior_curriculum import (
    Curriculum,
    Metrics,
    TrainerServer,
    TrainerState,
)
from aind_slims_api import SlimsClient, models
import logging
import os

class DynamicForagingTrainer(TrainerServer):
    def __init__(self, slims_client: SlimsClient = None) -> None:
        """
        Dynamic Foraging Trainer that loads data from Slims and DocDB and writes trainer state to Slims

        :param slims_client: client for Slims. If None, client will be instantiated based on environment variables.
        """
        super().__init__()

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.slims_client = slims_client if slims_client else self._connect_to_slims()

    def _connect_to_slims(self) -> SlimsClient:
        """
        Connect to Slims client

        :returns: Slims Client
        """

        try:
            self.log.info('Attempting to connect to Slims')
            self.slims_client = SlimsClient(username=os.environ['SLIMS_USERNAME'],
                                            password=os.environ['SLIMS_PASSWORD'])
        except KeyError as e:
            raise KeyError('SLIMS_USERNAME and SLIMS_PASSWORD do not exist as '
                           f'environment variables on machine. Please add. {e}')

        try:
            self.slims_client.fetch_model(models.SlimsMouseContent, barcode='00000000')
        except Exception as e:
            if 'Status 401 – Unauthorized' in str(e):    # catch error if username and password are incorrect
                raise Exception(f'Exception trying to read from Slims: {e}.\n'
                                f' Please check credentials:\n'
                                f'Username: {os.environ["SLIMS_USERNAME"]}\n'
                                f'Password: {os.environ["SLIMS_PASSWORD"]}')
            elif 'No record found' not in str(e):    # bypass if mouse doesn't exist
                raise Exception(f'Exception trying to read from Slims: {e}.\n')

        self.log.info('Successfully connected to Slims')

    def load_data(self, subject_id: int) -> tuple[Curriculum, TrainerState, Metrics]:
        """
        Read TrainerState of session from Slims and Metrics from DocDB

        :param subject_id: subject id of mouse to use to query docDB and Slims
        :returns: tuple of Curriculum, TrainerState, and Metrics of session
        """



    def write_data(
        self,
        subject_id: int,
        curriculum: Curriculum,
        trainer_state: TrainerState,
    ) -> None:
        """
        Add to proxy database.
        """
        MICE_CURRICULUMS[subject_id] = curriculum
        MICE_SUBJECT_HISTORY[subject_id].append(trainer_state)

        self.subject_history[subject_id].append(trainer_state)
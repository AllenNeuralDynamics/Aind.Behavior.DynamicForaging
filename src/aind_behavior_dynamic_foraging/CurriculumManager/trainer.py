from aind_behavior_curriculum import (
    Curriculum,
    Metrics,
    TrainerServer,
    TrainerState,
)
from aind_slims_api import SlimsClient, models
import logging
import os
from aind_data_access_api.document_db import MetadataDbClient
from aind_behavior_dynamic_foraging import DynamicForagingMetrics


class DynamicForagingTrainerServer(TrainerServer):
    def __init__(self, slims_client: SlimsClient = None, docdb_client: MetadataDbClient = None) -> None:
        """
        Dynamic Foraging Trainer that loads data from Slims and DocDB and writes trainer state to Slims

        :param slims_client: client for Slims. If None, client will be instantiated based on environment variables.
        """
        super().__init__()

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.slims_client = slims_client if slims_client else self._connect_to_slims()

        self.docdb_client = docdb_client if docdb_client else MetadataDbClient(
            host='api.allenneuraldynamics.org',
            database='metadata_index',
            collection='data_assets'
        )

    def _connect_to_slims(self) -> SlimsClient:
        """
        Connect to Slims client

        :returns: Slims Client
        """

        try:
            self.log.info('Attempting to connect to Slims')
            slims_client = SlimsClient(username=os.environ['SLIMS_USERNAME'],
                                       password=os.environ['SLIMS_PASSWORD'])
        except KeyError as e:
            raise KeyError('SLIMS_USERNAME and SLIMS_PASSWORD do not exist as '
                           f'environment variables on machine. Please add. {e}')

        try:
            slims_client.fetch_model(models.SlimsMouseContent, barcode='00000000')
        except Exception as e:
            if 'Status 401 – Unauthorized' in str(e):  # catch error if username and password are incorrect
                raise Exception(f'Exception trying to read from Slims: {e}.\n'
                                f' Please check credentials:\n'
                                f'Username: {os.environ["SLIMS_USERNAME"]}\n'
                                f'Password: {os.environ["SLIMS_PASSWORD"]}')
            elif 'No record found' not in str(e):  # bypass if mouse doesn't exist
                raise Exception(f'Exception trying to read from Slims: {e}.\n')

        self.log.info('Successfully connected to Slims')

        return slims_client

    def load_data(self, subject_id: str) -> tuple[Curriculum, TrainerState, Metrics]:
        """
        Read TrainerState of session from Slims and Metrics from DocDB

        :param subject_id: subject id of mouse to use to query docDB and Slims
        :returns: tuple of Curriculum, TrainerState, and Metrics of session
        """

        # grab trainer state from slims
        mouse = self.slims_client.fetch_model(models.SlimsMouseContent, barcode=subject_id)
        last_session = self.slims_client.fetch_models(models.behavior_session.SlimsBehaviorSession,
                                                      mouse_pk=mouse.pk, start=0, end=1)
        curriculum_attachments = self.slims_client.fetch_attachments(last_session[0])
        response = self.slims_client.fetch_attachment_content(curriculum_attachments[0]).json()
        # format response for valid TrainerState
        graph = {int(k): [tuple(transition) for transition in v] for k, v in
                 response['curriculum']['graph']['graph'].items()}
        nodes = {int(k): v for k, v in response['curriculum']['graph']['nodes'].items()}
        response['curriculum']['graph'] = {'graph': graph, 'nodes': nodes}

        trainer_state = TrainerState(**response)
        curriculum = trainer_state.curriculum

        # populate metrics
        sessions = self.docdb_client.retrieve_docdb_records(filter_query={
            "name": {"$regex": f"^behavior_{subject_id}"}
        })

        session_total = len(sessions)
        epochs = [session['session']['stimulus_epochs'][0] for session in sessions if session['session'] is not None]
        finished_trials = [epoch['trials_finished'] for epoch in epochs]
        foraging_efficiency = [epoch['output_parameters']['performance']['foraging_efficiency'] for epoch in epochs]

        # query slims for sessions to determine how many sessions at current stage
        slims_session = self.slims_client.fetch_models(models.behavior_session.SlimsBehaviorSession, mouse_pk=mouse.pk)
        current_stage = trainer_state.stage.name
        session_at_current_stage = len([session for session in slims_session if session.task_stage == current_stage])

        metrics = DynamicForagingMetrics(
            foraging_efficiency=foraging_efficiency,
            finished_trials=finished_trials,
            session_total=session_total,
            session_at_current_stage=session_at_current_stage
        )

        return curriculum, trainer_state, metrics

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

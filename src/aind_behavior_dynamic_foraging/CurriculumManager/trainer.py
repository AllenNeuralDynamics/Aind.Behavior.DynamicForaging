from aind_behavior_curriculum import (
    create_curriculum,
    Curriculum,
    Metrics,
    Trainer,
    TrainerState,
)
import aind_slims_api as slims
import logging
import os
import aind_data_access_api.document_db
from aind_behavior_dynamic_foraging import DynamicForagingMetrics, AindDynamicForagingTaskLogic
from pydantic import Field
from datetime import datetime
from typing import Union


class DynamicForagingTrainerState(TrainerState):
    curriculum: Union[
        create_curriculum("CoupledBaiting2p3Curriculum", "2.3.0",
                          [AindDynamicForagingTaskLogic])(),
        create_curriculum("UnCoupledBaiting2p3Curriculum", "2.3.0",
                          [AindDynamicForagingTaskLogic])(),
        create_curriculum("UncoupledNoBaitingRewardDelayCurriculum2p3p1",
                          "2.3.1", [AindDynamicForagingTaskLogic])(),
        create_curriculum("UnCoupledNoBaiting2p3Curriculum",
                          "2.3.0",
                          [AindDynamicForagingTaskLogic])()
    ] = Field()


class DynamicForagingTrainerServer:
    def __init__(self, slims_client: slims.SlimsClient = None,
                 docdb_client: aind_data_access_api.document_db.MetadataDbClient = None) -> None:
        """
        Dynamic Foraging Trainer that loads data from Slims and DocDB and writes trainer state to Slims

        :param slims_client: client for Slims. If None, client will be instantiated based on environment variables.
        """

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.slims_client = slims_client if slims_client else self._connect_to_slims()
        self._check_slims_access()

        self.docdb_client = docdb_client if docdb_client else aind_data_access_api.document_db.MetadataDbClient(
            host='api.allenneuraldynamics.org',
            database='metadata_index',
            collection='data_assets'
        )

    def _connect_to_slims(self) -> slims.SlimsClient:
        """
        Connect to Slims client

        :returns: Slims Client
        """

        try:
            self.log.info('Attempting to connect to Slims')
            slims_client = slims.SlimsClient(username=os.environ['SLIMS_USERNAME'],
                                             password=os.environ['SLIMS_PASSWORD'])
        except KeyError as e:
            raise KeyError('SLIMS_USERNAME and SLIMS_PASSWORD do not exist as '
                           f'environment variables on machine. Please add. {e}')

        self.log.info('Successfully connected to Slims')

        return slims_client

    def _check_slims_access(self) -> None:
        """
        Method to check ability to read from Slims. Trying to catch instances of incorrect credentials
        """

        try:
            self.slims_client.fetch_model(slims.models.SlimsMouseContent, barcode='00000000')
        except Exception as e:
            if 'Status 401 – Unauthorized' in str(e):  # catch error if username and password are incorrect
                raise Exception(f'Exception trying to read from Slims: {e}.\n'
                                f'Please check credentials:\n'
                                f'Username: {os.environ["SLIMS_USERNAME"]}\n'
                                f'Password: {os.environ["SLIMS_PASSWORD"]}')
            elif 'No record found' not in str(e):  # bypass if mouse doesn't exist
                raise Exception(f'Exception trying to read from Slims: {e}.\n')

        self.log.debug('Successfully read from Slims.')

    def load_data(self, subject_id: str) -> tuple[Curriculum or None,
                                                  TrainerState or None,
                                                  Metrics or None,
                                                  [slims.models.SlimsAttachment],
                                                  slims.models.behavior_session.SlimsBehaviorSession or None
    ]:
        """
        Read TrainerState of session from Slims and Metrics from DocDB

        :param subject_id: subject id of mouse to use to query docDB and Slims
        :returns: tuple of Curriculum, TrainerState, Metrics, attachments, and last session in slims
        """

        # grab trainer state from slims
        mouse = self.slims_client.fetch_model(slims.models.SlimsMouseContent, barcode=subject_id)
        slims_sessions = self.slims_client.fetch_models(slims.models.behavior_session.SlimsBehaviorSession,
                                                        mouse_pk=mouse.pk, sort="date")
        if slims_sessions != []:  # no sessions related to mouse
            curriculum_attachments = self.slims_client.fetch_attachments(slims_sessions[-1])
            # get most recently added TrainerState
            response = [self.slims_client.fetch_attachment_content(attach).json() for attach in curriculum_attachments
                        if attach.name == "TrainerState"][0]

            # format response for valid TrainerState
            graph = {int(k): [tuple(transition) for transition in v] for k, v in
                     response['curriculum']['graph']['graph'].items()}
            nodes = {int(k): v for k, v in response['curriculum']['graph']['nodes'].items()}
            response['curriculum']['graph'] = {'graph': graph, 'nodes': nodes}

            trainer_state = DynamicForagingTrainerState(**response)
            trainer_state.stage.task = AindDynamicForagingTaskLogic(**trainer_state.stage.task.model_dump())
            curriculum = trainer_state.curriculum

            # populate metrics
            sessions = self.docdb_client.retrieve_docdb_records(
                filter_query={"name": {"$regex": f"^behavior_{subject_id}(?!.*processed).*"}}
            )
            session_total = len(sessions)
            sessions = [session for session in sessions if session['session'] is not None]  # sort out none types
            sessions.sort(key=lambda session: session['session']['session_start_time'])  # sort based on time
            epochs = [session['session']['stimulus_epochs'][0] for session in sessions]
            finished_trials = [epoch['trials_finished'] for epoch in epochs]
            foraging_efficiency = [epoch['output_parameters']['performance']['foraging_efficiency'] for epoch in epochs]

            current_stage = trainer_state.stage.name

            # add consecutive session on current stage
            session_at_current_stage = 0
            for sess in reversed(slims_sessions):
                if sess.task_stage == current_stage:
                    session_at_current_stage += 1
                else:
                    break
        
            metrics = DynamicForagingMetrics(
                foraging_efficiency=foraging_efficiency,
                finished_trials=finished_trials,
                session_total=session_total,
                session_at_current_stage=session_at_current_stage
            )

        else:
            curriculum = None
            trainer_state = None
            metrics = None
            curriculum_attachments = []
            slims_sessions = [None]

        return curriculum, trainer_state, metrics, curriculum_attachments, slims_sessions[-1]

    def write_data(
            self,
            subject_id: str,
            curriculum: Curriculum,
            trainer_state: TrainerState,
            date: datetime = datetime.now(),
            on_curriculum: bool = True,

    ) -> slims.models.SlimsBehaviorSession:

        """
        Generate and write next SlimsBehaviorSession to slims and add a TrainerState attachment

        :param on_curriculum: if mouse is on curriculum
        :param date: date of acquisitions
        :param subject_id: subject id of mouse to use to query docDB and Slims
        :param curriculum: curriculum for next session
        :param trainer_state: trainer state for next session
        :experimenters: list of experimenters who ran session
        """

        mouse = self.slims_client.fetch_model(slims.models.SlimsMouseContent, barcode=subject_id)

        self.log.info("Writing next session to slims.")
        # create session
        added_session = self.slims_client.add_model(
            slims.models.SlimsBehaviorSession(
                mouse_pk=mouse.pk,
                task_stage=trainer_state.stage.name,
                task=curriculum.name,
                task_schema_version=curriculum.version,
                is_curriculum_suggestion=on_curriculum,
                date=date,
            )
        )

        # add trainer_state as an attachment
        self.slims_client.add_attachment_content(
            record=added_session,
            name="TrainerState",
            content=trainer_state.model_dump_json()
        )

        return added_session

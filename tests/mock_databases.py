from unittest.mock import Mock
from typing import Literal
from aind_auto_train.schema.curriculum import DynamicForagingCurriculum
import os
from pathlib import Path
import json
from aind_slims_api.models import SlimsMouseContent, SlimsAttachment
from aind_slims_api.models.behavior_session import SlimsBehaviorSession
from requests.models import Request

CURRICULUM_NAMES = Literal["Coupled Baiting", "Uncoupled Baiting", "Uncoupled Without Baiting"]
CURRICULUM_VERSION = Literal["2.3", "2.3.1rwdDelay159"]
CURRICULUM_SCHEMA_VERSION = Literal["1.0"]

RESOURCES_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"
COUPLED_BAITING_PATH = RESOURCES_DIR / "Coupled Baiting_curriculum_v2.3_schema_v1.0.json"
UNCOUPLED_BAITING_PATH = RESOURCES_DIR / "Uncoupled Baiting_curriculum_v2.3_schema_v1.0.json"
UNCOUPLED_NO_BAITING_PATH = RESOURCES_DIR / "Uncoupled Without Baiting_curriculum_v2.3.1rwdDelay159_schema_v1.0.json"


class MockCurriculumManager(Mock):
    """Mock class to return saved curriculums"""

    def get_curriculum(self, curriculum_name: CURRICULUM_NAMES,
                       curriculum_version: CURRICULUM_VERSION,
                       curriculum_schema_version: CURRICULUM_SCHEMA_VERSION) -> dict:

        if curriculum_name == "Coupled Baiting":
            path = COUPLED_BAITING_PATH
        elif curriculum_name == "Uncoupled Baiting":
            path = UNCOUPLED_BAITING_PATH
        else:
            path = UNCOUPLED_NO_BAITING_PATH

        with open(path, 'r') as file:
            data = json.load(file)

        return {"curriculum": DynamicForagingCurriculum(**data)}


MOUSE_BARCODE = Literal['00000001']
MOUSE_MODEL_PATH = RESOURCES_DIR / "slims_mouse.json"
BEHAVIOR_SESSION_PATH = RESOURCES_DIR / "slims_behavior_session.json"
ATTACHMENT_PATH = RESOURCES_DIR / "slims_attachment.json"
ATTACHMENT_CONTENT_PATH = RESOURCES_DIR / "slims_attachment_content.json"
class MockSlimsClient(Mock):
    """Mock class to simulate Slims"""

    def fetch_model(self, model: SlimsMouseContent, barcode: MOUSE_BARCODE) -> SlimsMouseContent:
        """
        Mock fetching models of SlimsMouseContent
        """
        with open(MOUSE_MODEL_PATH, 'r') as file:
            data = json.load(file)
        return SlimsMouseContent(**data)

    def fetch_models(self, model: SlimsBehaviorSession, *args, **kwargs) -> list[SlimsBehaviorSession]:
        """
        Mock fetching list of SlimsBehaviorSession
        """

        with open(BEHAVIOR_SESSION_PATH, 'r') as file:
            data = json.load(file)

        return [SlimsBehaviorSession(**data)]

    def fetch_attachments(self, record: SlimsBehaviorSession, *args, **kwargs) -> list[SlimsAttachment]:
        """
        Mock fetching trainer state attachment
        """

        with open(ATTACHMENT_PATH, 'r') as file:
            data = json.load(file)

        return [SlimsAttachment(**data)]

    def fetch_attachment_content(self, attachment: SlimsAttachment, *args, **kwargs) -> dict:
        """
        Mock fetching trainer state attachment
        """

        with open(ATTACHMENT_CONTENT_PATH, 'r') as file:
            data = json.load(file)
        content = Mock()
        content.json = lambda: data
        return content

    def add_model(self, *args, **kwargs):
        """
        Mock adding model to slims
        """

    def add_attachment_content(self, *args, **kwargs):
        """
        Mock adding attachments
        """

DOCDB_SESSION_PATH = RESOURCES_DIR / "docdb_session.json"
class MockDocDBClient(Mock):
    """Mock class to simulate DocDB"""

    def retrieve_docdb_records(self, *args, **kwargs) -> list[dict]:
        """
        Mock retrieving records
        """

        with open(DOCDB_SESSION_PATH, 'r') as file:
            data = json.load(file)

        return [data]
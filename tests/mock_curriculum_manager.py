from unittest.mock import Mock
from typing import Literal
from aind_auto_train.schema.curriculum import DynamicForagingCurriculum
import os
from pathlib import Path
import json

CURRICULUM_NAMES = Literal["Coupled Baiting", "Uncoupled Baiting", "Uncoupled Without Baiting"]
CURRICULUM_VERSION = Literal["2.3","2.3.1rwdDelay159"]
CURRICULUM_SCHEMA_VERSION = Literal["1.0"]

RESOURCES_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"
COUPLED_BAITING_PATH = RESOURCES_DIR / "Coupled Baiting_curriculum_v2.3_schema_v1.0.json"
UNCOUPLED_BAITING_PATH = RESOURCES_DIR / "Uncoupled Baiting_curriculum_v2.3_schema_v1.0.json"
UNCOUPLED_NO_BAITING_PATH = RESOURCES_DIR / "Uncoupled Without Baiting_curriculum_v2.3.1rwdDelay159_schema_v1.0.json"

class MockCurriculumManager(Mock):
    """Mock class to return saved curriculums"""

    def get_curriculum(self, curriculum_name: CURRICULUM_NAMES,
                       curriculum_version:CURRICULUM_VERSION,
                       curriculum_schema_version:CURRICULUM_SCHEMA_VERSION) -> dict:

        if curriculum_name == "Coupled Baiting":
            path = COUPLED_BAITING_PATH
        elif curriculum_name == "Uncoupled Baiting":
            path = UNCOUPLED_BAITING_PATH
        else:
            path = UNCOUPLED_NO_BAITING_PATH

        with open(path, 'r') as file:
            data = json.load(file)

        return {"curriculum":DynamicForagingCurriculum(**data)}



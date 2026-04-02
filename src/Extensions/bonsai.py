import logging
import sys
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from aind_behavior_dynamic_foraging.task_logic import TrialGeneratorSpec  # noqa

if TYPE_CHECKING:
    from aind_behavior_dynamic_foraging.task_logic.trial_generators._base import ITrialGenerator


def resolve_generator(spec: TrialGeneratorSpec | str) -> "ITrialGenerator":
    """Resolves and creates the trial generator instance based on the task logic's trial generator model."""
    if isinstance(spec, str):
        adapter: TypeAdapter[TrialGeneratorSpec] = TypeAdapter(TrialGeneratorSpec)
        spec = adapter.validate_json(spec)
    return spec.create_generator()

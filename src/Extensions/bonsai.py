from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from aind_behavior_dynamic_foraging.task_logic import TrialGeneratorSpec

if TYPE_CHECKING:
    from aind_behavior_dynamic_foraging.task_logic.trial_generators._base import ITrialGenerator

import logging
import logging.config

logging_config = {
    "version": 1, 
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        },
    },
    "root": {
        "handlers": ["console"]
    }
}

logging.config.dictConfig(logging_config)
logging.getLogger(__name__)

def resolve_generator(spec: TrialGeneratorSpec | str) -> "ITrialGenerator":
    """Resolves and creates the trial generator instance based on the task logic's trial generator model."""
    if isinstance(spec, str):
        adapter: TypeAdapter[TrialGeneratorSpec] = TypeAdapter(TrialGeneratorSpec)
        spec = adapter.validate_json(spec)
    return spec.create_generator()

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from aind_behavior_dynamic_foraging.task_logic import TrialGeneratorSpec

if TYPE_CHECKING:
    from aind_behavior_dynamic_foraging.task_logic.trial_generators._base import ITrialGenerator

import logging
import logging.config
from datetime import datetime

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "mode": "a",
            "filename": f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file",]
    }
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

def resolve_generator(spec: TrialGeneratorSpec | str) -> "ITrialGenerator":
    """Resolves and creates the trial generator instance based on the task logic's trial generator model."""
    if isinstance(spec, str):
        adapter: TypeAdapter[TrialGeneratorSpec] = TypeAdapter(TrialGeneratorSpec)
        spec = adapter.validate_json(spec)
    return spec.create_generator()

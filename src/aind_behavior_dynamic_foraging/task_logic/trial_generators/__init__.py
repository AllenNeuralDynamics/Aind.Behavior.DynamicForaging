from typing import TYPE_CHECKING, Annotated, TypeAliasType, Union

from pydantic import Field

from ._base import _ITrialGenerator
from ._dummy_trial_generator import DummyTrialGeneratorModel
from ._dummy_trial_generator_2 import DummyTrialGeneratorModel2

if TYPE_CHECKING:
    TrialGeneratorSpec = Union[(DummyTrialGeneratorModel, DummyTrialGeneratorModel2)]
else:
    TrialGeneratorSpec = TypeAliasType(
        "TrialGeneratorSpec",
        Annotated[
            Union[(DummyTrialGeneratorModel, DummyTrialGeneratorModel2)], Field(discriminator="type", description="Type of trial generator")
        ],
    )


def resolve_generator(spec: TrialGeneratorSpec) -> _ITrialGenerator:
    """Resolves and creates the trial generator instance based on the task logic's trial generator model."""
    return spec.create_generator()

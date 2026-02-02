from typing import TYPE_CHECKING, Annotated, TypeAliasType, Union

from pydantic import Field, TypeAdapter

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


def resolve_generator(spec: TrialGeneratorSpec | str) -> _ITrialGenerator:
    """Resolves and creates the trial generator instance based on the task logic's trial generator model."""
    if isinstance(spec, str):
        adapter: TypeAdapter[TrialGeneratorSpec] = TypeAdapter(TrialGeneratorSpec)
        spec = adapter.validate_json(spec)
    return spec.create_generator()
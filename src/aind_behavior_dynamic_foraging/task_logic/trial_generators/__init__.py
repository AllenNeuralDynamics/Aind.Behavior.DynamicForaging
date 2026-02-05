from typing import TYPE_CHECKING, Annotated, TypeAliasType, Union

from pydantic import Field

from ._base import ITrialGenerator as ITrialGenerator
from .coupled_trial_generator import CoupledTrialGenerator
from .integration_test_trial_generator import IntegrationTestTrialGenerator

if TYPE_CHECKING:
    TrialGeneratorSpec = Union[(CoupledTrialGenerator, IntegrationTestTrialGenerator)]
else:
    TrialGeneratorSpec = TypeAliasType(
        "TrialGeneratorSpec",
        Annotated[
            Union[(CoupledTrialGenerator, IntegrationTestTrialGenerator)],
            Field(discriminator="type", description="Type of trial generator"),
        ],
    )

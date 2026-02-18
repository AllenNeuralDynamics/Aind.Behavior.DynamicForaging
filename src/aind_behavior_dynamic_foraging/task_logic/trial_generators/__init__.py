from typing import TYPE_CHECKING, Annotated, TypeAliasType, Union

from pydantic import Field

from ._base import BaseTrialGeneratorSpecModel as BaseTrialGeneratorSpecModel
from ._base import ITrialGenerator as ITrialGenerator
from .composite_trial_generator import TrialGeneratorCompositeSpec
from .coupled_trial_generator import CoupledTrialGeneratorSpec
from .integration_test_trial_generator import IntegrationTestTrialGeneratorSpec

if TYPE_CHECKING:
    TrialGeneratorSpec = Union[
        (
            CoupledTrialGeneratorSpec,
            IntegrationTestTrialGeneratorSpec,
            TrialGeneratorCompositeSpec["TrialGeneratorSpec"],
        )
    ]
else:
    TrialGeneratorSpec = TypeAliasType(
        "TrialGeneratorSpec",
        Annotated[
            Union[
                (
                    CoupledTrialGeneratorSpec,
                    IntegrationTestTrialGeneratorSpec,
                    TrialGeneratorCompositeSpec["TrialGeneratorSpec"],
                )
            ],
            Field(discriminator="type", description="Type of trial generator"),
        ],
    )

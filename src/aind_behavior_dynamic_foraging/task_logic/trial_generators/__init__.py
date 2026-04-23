from typing import TYPE_CHECKING, Annotated, TypeAliasType, Union

from pydantic import Field

from ._base import BaseTrialGeneratorSpecModel as BaseTrialGeneratorSpecModel
from ._base import ITrialGenerator as ITrialGenerator
from .composite_trial_generator import TrialGeneratorCompositeSpec
from .block_based_trial_generator import BlockBasedTrialGeneratorSpec
from .uncoupled_trial_gnerator import UncoupledTrialGeneratorSpec
from .coupled_trial_generators.base_coupled_trial_generator import BaseCoupledTrialGeneratorSpec
from .coupled_trial_generators.coupled_trial_generator import CoupledTrialGeneratorSpec
from .coupled_trial_generators.coupled_warmup_trial_generator import CoupledWarmupTrialGeneratorSpec
from .integration_test_trial_generator import IntegrationTestTrialGeneratorSpec

if TYPE_CHECKING:
    TrialGeneratorSpec = Union[
        (
            UncoupledTrialGeneratorSpec,
            BlockBasedTrialGeneratorSpec,
            BaseCoupledTrialGeneratorSpec,
            CoupledWarmupTrialGeneratorSpec,
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
                (   UncoupledTrialGeneratorSpec,
                    BlockBasedTrialGeneratorSpec,
                    BaseCoupledTrialGeneratorSpec,
                    CoupledWarmupTrialGeneratorSpec,
                    CoupledTrialGeneratorSpec,
                    IntegrationTestTrialGeneratorSpec,
                    TrialGeneratorCompositeSpec["TrialGeneratorSpec"],
                )
            ],
            Field(discriminator="type", description="Type of trial generator"),
        ],
    )

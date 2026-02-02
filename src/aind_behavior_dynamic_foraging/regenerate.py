from pathlib import Path
from typing import Union

import pydantic
from aind_behavior_services.schema import BonsaiSgenSerializers, convert_pydantic_to_bonsai
from aind_behavior_services.session import Session

import aind_behavior_dynamic_foraging.rig
import aind_behavior_dynamic_foraging.task_logic.trial_generators._dummy_trial_generator

SCHEMA_ROOT = Path("./schema")
EXTENSIONS_ROOT = Path("./src/Extensions/")
NAMESPACE_PREFIX = "AindDynamicForagingDataSchema"


def main():
    models = [
        aind_behavior_dynamic_foraging.task_logic.AindDynamicForagingTaskLogic,
        aind_behavior_dynamic_foraging.rig.AindDynamicForagingRig,
        Session,
        aind_behavior_dynamic_foraging.task_logic.trial_models.Trial,
        aind_behavior_dynamic_foraging.task_logic.trial_models.TrialOutcome,
    ]
    model = pydantic.RootModel[Union[tuple(models)]]

    convert_pydantic_to_bonsai(
        model,
        model_name="aind_behavior_dynamic_foraging",
        root_element="Root",
        cs_namespace=NAMESPACE_PREFIX,
        json_schema_output_dir=SCHEMA_ROOT,
        cs_output_dir=EXTENSIONS_ROOT,
        cs_serializer=[BonsaiSgenSerializers.JSON],
    )


if __name__ == "__main__":
    main()

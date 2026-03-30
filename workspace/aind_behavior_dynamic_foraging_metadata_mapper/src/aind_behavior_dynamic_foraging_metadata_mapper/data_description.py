import os

from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_services.session import Session
from aind_data_schema.components.identifiers import Person
from aind_data_schema.core.data_description import DataDescription, Funding
from aind_data_schema_models.data_name_patterns import DataLevel, Group
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization


def data_description_from_dataset(
    data_directory: os.PathLike,
) -> DataDescription:
    """
    Create acquisition model for completed session.

    Args:
        data_directory (os.PathLike):
            Path to the directory containing the dataset to analyze. This
            directory is expected to include all required behavioral data files.

    Returns:
        DataDescription:
            DataDescription model for session

    Raises:
        FileNotFoundError:
            If the specified data directory or required files do not exist.

        ValueError:
            If the dataset is malformed or missing required fields for
            computing metrics.
    """

    dataset = df_foraging_dataset(data_directory)
    software_events = dataset["Behavior"]["SoftwareEvents"]
    software_events.load_all()

    input_schemas = dataset["Behavior"]["InputSchemas"]
    session = Session.model_validate(input_schemas["Session"].data)

    return DataDescription(
        subject_id=session.subject,
        creation_time=session.date,
        institution=Organization.AIND,
        funding_source=[
            Funding(
                funder=Organization.AI,
            )
        ],
        data_level=DataLevel.RAW,
        investigators=[Person(name=session.experimenter[0])],
        project_name="DynamicForaging",
        modalities=[
            Modality.BEHAVIOR,
            Modality.BEHAVIOR_VIDEOS,
        ],
        group=Group.BEHAVIOR,
    )

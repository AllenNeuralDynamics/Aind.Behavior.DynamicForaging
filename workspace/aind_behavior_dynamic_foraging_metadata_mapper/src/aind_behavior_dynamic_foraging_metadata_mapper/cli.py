import logging
import os
import typing as t
from pathlib import Path

from pydantic import AwareDatetime, Field
from pydantic_settings import BaseSettings, CliApp

logger = logging.getLogger(__name__)


class DataMapperCli(BaseSettings, cli_kebab_case=True):
    data_directory: os.PathLike = Field(description="Path to the session data directory.")
    repo_path: os.PathLike = Field(
        default=Path("."), description="Path to the repository. By default it will use the current directory."
    )
    session_end_time: AwareDatetime | None = Field(
        default=None,
        description="End time of the session in ISO format. If not provided, will use the time the data mapping is run.",
    )
    suffix: t.Optional[str] = Field(default="", description="Suffix to append to the output filenames.")

    def cli_cmd(self):
        """Generate aind-data-schema metadata for the Dynamic Foraging dataset located at the specified path."""
        from .acquisition import acqusition_from_dataset
        from .data_description import data_description_from_dataset
        from .instrument import instrument_from_dataset

        acquisition = acqusition_from_dataset(
            data_directory=Path(self.data_directory),
            repo_path=Path(self.repo_path),
            end_time=self.session_end_time,
        )

        instrument = instrument_from_dataset(data_directory=Path(self.data_directory))
        data_description = data_description_from_dataset(data_directory=Path(self.data_directory))

        acquisition.write_standard_file(output_directory=Path(self.data_directory), filename_suffix=self.suffix)
        instrument.write_standard_file(output_directory=Path(self.data_directory), filename_suffix=self.suffix)
        data_description.write_standard_file(output_directory=Path(self.data_directory), filename_suffix=self.suffix)

        logger.info(
            "Mapping completed! Saved acquisition.json, instrument.json, data_description.json to %s",
            self.data_directory,
        )


def main():
    CliApp.run(DataMapperCli)


if __name__ == "__main__":
    main()

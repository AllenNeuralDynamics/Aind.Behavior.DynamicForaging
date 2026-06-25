import logging
import os
import typing as t
from pathlib import Path

from pydantic import AwareDatetime, Field
from pydantic_settings import BaseSettings, CliApp

logger = logging.getLogger(__name__)


class DataMapperCli(BaseSettings, cli_kebab_case=True):
    data_path: os.PathLike = Field(description="Path to the session data directory.")
    repository_path: os.PathLike = Field(
        default=Path("."), description="Path to the repository. By default it will use the current directory."
    )
    session_end_time: AwareDatetime | None = Field(
        default=None,
        description="End time of the session in ISO format. If not provided, will use the time the data mapping is run.",
    )
    suffix: t.Optional[str] = Field(default="", description="Suffix to append to the output filenames.")

    def cli_cmd(self):
        """Generate aind-data-schema metadata for the Dynamic Foraging dataset located at the specified path."""
        from .acquisition import AindAcquisitionDataMapper
        from .instrument import AindInstrumentDataMapper

        session_mapper = AindAcquisitionDataMapper(
            data_path=Path(self.data_path),
            repository_path=Path(self.repository_path),
            session_end_time=self.session_end_time,
        )
        acquisition = session_mapper.map()

        rig_mapper = AindInstrumentDataMapper(data_path=Path(self.data_path))
        instrument = rig_mapper.map()

        assert session_mapper.mapped is not None
        assert rig_mapper.mapped is not None

        acquisition.write_standard_file(output_directory=Path(self.data_path), filename_suffix=self.suffix)
        instrument.write_standard_file(output_directory=Path(self.data_path), filename_suffix=self.suffix)

        logger.info(
            "Mapping completed! Saved acquisition.json, instrument.json to %s",
            self.repository_path,
        )


def main():
    CliApp.run(DataMapperCli)


if __name__ == "__main__":
    main()

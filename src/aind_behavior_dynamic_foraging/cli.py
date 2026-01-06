import typing as t

from pydantic import Field, RootModel
from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from aind_behavior_dynamic_foraging import __semver__, regenerate


class VersionCli(RootModel):
    root: t.Any

    def cli_cmd(self) -> None:
        print(__semver__)


class DslRegenerateCli(RootModel):
    root: t.Any

    def cli_cmd(self) -> None:
        regenerate.main()


class DynamicForagingCli(BaseSettings, cli_prog_name="dynamic-foraging", cli_kebab_case=True):
    version: CliSubCommand[VersionCli] = Field(
        description="Print the version of the dynamic-foraging package.",
    )
    regenerate: CliSubCommand[DslRegenerateCli] = Field(
        description="Regenerate the dynamic-foraging dsl dependencies.",
    )

    def cli_cmd(self):
        return CliApp().run_subcommand(self)


def main():
    CliApp().run(DynamicForagingCli)


if __name__ == "__main__":
    main()

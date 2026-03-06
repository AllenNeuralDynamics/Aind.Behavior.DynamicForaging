import logging
import typing as ty
from pathlib import Path

import clabe.resource_monitor
import pandas as pd
from aind_behavior_services.rig.aind_manipulator import ManipulatorPosition
from aind_behavior_services.session import Session
from clabe.apps import (
    AindBehaviorServicesBonsaiApp,
)
from clabe.data_transfer.robocopy import RobocopyService, RobocopySettings
from clabe.launcher import Launcher, LauncherCliArgs, experiment
from clabe.pickers import ByAnimalModifier, DefaultBehaviorPicker, DefaultBehaviorPickerSettings
from pydantic_settings import CliApp

from aind_behavior_dynamic_foraging import data_contract
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic

logger = logging.getLogger(__name__)


@experiment()
async def iso_force_experiment(launcher: Launcher) -> None:
    picker = DefaultBehaviorPicker(
        launcher=launcher,
        settings=DefaultBehaviorPickerSettings(
            config_library_dir=r"\\allen\aind\scratch\AindBehavior.db\AindDynamicForaging"
        ),
    )

    session = picker.pick_session(Session)
    task_logic = picker.pick_task(AindDynamicForagingTaskLogic)
    rig = picker.pick_rig(AindDynamicForagingRig)
    ensure_rig_and_computer_name(rig)

    # Post-fetching modifications
    manipulator_modifier = ByAnimalManipulatorModifier(
        subject_db_path=picker.subject_dir / session.subject,
        model_path="manipulator.calibration.initial_position",
        model_name="manipulator_init.json",
        launcher=launcher,
    )
    manipulator_modifier.inject(rig)

    launcher.register_session(session, rig.data_directory)

    clabe.resource_monitor.ResourceMonitor(
        constrains=[
            clabe.resource_monitor.available_storage_constraint_factory(rig.data_directory, 2e11),
        ]
    ).run()

    bonsai_app = AindBehaviorServicesBonsaiApp(
        workflow=Path(r"./src/main.bonsai"),
        temp_directory=launcher.temp_dir,
        rig=rig,
        session=session,
        task=task_logic,
    )
    await bonsai_app.run_async()

    # Update manipulator initial position for next session
    try:
        manipulator_modifier.dump()
    except Exception as e:
        logger.error("Failed to update manipulator initial position: %s", e)

    # Run data qc
    if picker.ui_helper.prompt_yes_no_question("Would you like to generate a qc report?"):
        try:
            import webbrowser

            from contraqctor.qc.reporters import HtmlReporter

            from aind_behavior_dynamic_foraging.data_qc.suite import make_qc_runner

            vr_dataset = data_contract.dataset(launcher.session_directory)
            runner = make_qc_runner(vr_dataset)
            qc_path = launcher.session_directory / "Behavior" / "Logs" / "qc_report.html"
            reporter = HtmlReporter(output_path=qc_path)
            runner.run_all_with_progress(reporter=reporter)
            webbrowser.open(qc_path.as_uri(), new=2)
        except Exception as e:
            logger.error("Failed to run data QC: %s", e)

    # Transfer data
    is_transfer = picker.ui_helper.prompt_yes_no_question("Would you like to transfer data?")
    if not is_transfer:
        logger.info("Data transfer skipped by user.")
        return

    launcher.copy_logs()
    settings = RobocopySettings(destination=r"\\allen\aind\scratch\AindDynamicForaging\data")
    assert launcher.session.session_name is not None, "Session name is None"
    settings.destination = Path(settings.destination) / launcher.session.session_name
    RobocopyService(source=launcher.session_directory, settings=settings).transfer()
    return


def ensure_rig_and_computer_name(rig: AindDynamicForagingRig) -> None:
    """Ensures rig and computer name are set from environment variables if available, otherwise defaults to rig configuration values."""

    import os

    rig_name = os.environ.get("aibs_comp_id", None)
    computer_name = os.environ.get("hostname", None)

    if rig_name is None:
        logger.warning(
            "'aibs_comp_id' environment variable not set. Defaulting to rig name from configuration. %s", rig.rig_name
        )
        rig_name = rig.rig_name
    if computer_name is None:
        computer_name = rig.computer_name
        logger.warning(
            "'hostname' environment variable not set. Defaulting to computer name from configuration. %s",
            rig.computer_name,
        )

    if rig_name != rig.rig_name or computer_name != rig.computer_name:
        logger.warning(
            "Rig name or computer name from environment variables do not match the rig configuration. "
            "Forcing rig name: %s and computer name: %s from environment variables.",
            rig_name,
            computer_name,
        )
        rig.rig_name = rig_name
        rig.computer_name = computer_name


class ByAnimalManipulatorModifier(ByAnimalModifier[AindDynamicForagingRig]):
    """Modifier to set and update manipulator initial position based on animal-specific data."""

    def __init__(
        self, subject_db_path: Path, model_path: str, model_name: str, *, launcher: Launcher, **kwargs
    ) -> None:
        super().__init__(subject_db_path, model_path, model_name, **kwargs)
        self._launcher = launcher

    def _process_before_dump(self) -> ManipulatorPosition:
        _dataset = data_contract.dataset(self._launcher.session_directory)
        manipulator_parking_position: pd.DataFrame = ty.cast(
            pd.DataFrame, _dataset["Behavior"]["SoftwareEvents"]["InitialManipulatorPosition"].read()
        )
        return ManipulatorPosition.model_validate(manipulator_parking_position.iloc[0]["data"])


class ClabeCli(LauncherCliArgs):
    def cli_cmd(self):
        launcher = Launcher(settings=self)
        launcher.run_experiment(iso_force_experiment)
        return None


def main() -> None:
    CliApp().run(ClabeCli)


if __name__ == "__main__":
    main()

import typing as t

import pandas as pd
from contraqctor import contract, qc
from contraqctor.contract.harp import HarpDevice

from ..rig import AindDynamicForagingRig


class DynamicForagingQcSuite(qc.Suite):
    def __init__(self, dataset: contract.Dataset):
        self.dataset = dataset

    def test_end_session_exists(self):
        """Check that the session has an end event."""
        end_session = self.dataset["Behavior"]["SoftwareEvents"]["EndSession"]
        if not end_session.has_data:
            return self.fail_test(
                None, "EndSession event does not exist. Session may be corrupted or not ended properly."
            )

        assert isinstance(end_session.data, pd.DataFrame)
        if end_session.data.empty:
            return self.fail_test(None, "No data in EndSession. Session may be corrupted or not ended properly.")
        else:
            return self.pass_test(None, "EndSession event exists with data.")


def make_qc_runner(dataset: contract.Dataset) -> qc.Runner:
    _runner = qc.Runner()
    dataset.load_all(strict=False)
    exclude: list[contract.DataStream] = []
    rig: AindDynamicForagingRig = dataset["Behavior"]["InputSchemas"]["Rig"].data

    # Exclude commands to Harp boards as these are tested separately
    for cmd in dataset["Behavior"]["HarpCommands"]:
        if cmd.has_error:
            continue
        for stream in cmd:
            if isinstance(stream, contract.harp.HarpRegister):
                exclude.append(stream)

    # Add harp board specific tests
    exclude_streams: list[str] = []
    if not rig.harp_sniff_detector:
        exclude_streams.append("HarpSniffDetector")

    if not rig.harp_environment_sensor:
        exclude_streams.append("HarpEnvironmentSensor")

    if not rig.harp_lickometer_right:
        exclude_streams.append("HarpLickometerRight")

    if not rig.harp_lickometer_left:
        exclude_streams.append("HarpLickometerLeft")

    # Add Harp tests for ALL Harp devices in the dataset
    for stream in (_r := dataset["Behavior"]):
        if isinstance(stream, HarpDevice) and stream.name not in exclude_streams:
            commands = t.cast(HarpDevice, _r["HarpCommands"][stream.name])
            _runner.add_suite(qc.harp.HarpDeviceTestSuite(stream, commands), stream.name)

    # Add Harp Hub tests
    _runner.add_suite(
        qc.harp.HarpHubTestSuite(
            dataset["Behavior"]["HarpClockGenerator"],
            [
                harp_device
                for harp_device in dataset["Behavior"]
                if isinstance(harp_device, HarpDevice) and harp_device.name not in exclude_streams
            ],
        ),
        "HarpHub",
    )

    # Add camera qc
    for camera in dataset["BehaviorVideos"]:
        _runner.add_suite(
            qc.camera.CameraTestSuite(camera, expected_fps=rig.triggered_camera_controller.frame_rate), camera.name
        )

    # Add Csv tests
    csv_streams = [stream for stream in dataset.iter_all() if isinstance(stream, contract.csv.Csv)]
    for stream in csv_streams:
        _runner.add_suite(qc.csv.CsvTestSuite(stream), stream.name)

    # Add the task specific tests
    _runner.add_suite(DynamicForagingQcSuite(dataset), "DynamicForaging")
    return _runner

import os
from pathlib import Path

import pandas as pd
import numpy as np
from aind_behavior_dynamic_foraging.data_contract import dataset as df_dataset
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig


def _calculate_side_volume_ml(
    set_open_time_ms: pd.Series,
    delivery_times: pd.DataFrame,
    slope_g_per_s: float,
    offset_g: float,
) -> float:
    """Estimate delivered volume (mL) for one side from pulse durations and calibration."""

    delivery_times = delivery_times.reset_index(names="Time")
    if delivery_times.empty:
        return 0.0

    pulse_series_ms = pd.to_numeric(set_open_time_ms, errors="coerce").dropna().sort_index()
    if pulse_series_ms.empty:
        return 0.0

    setpoints = pulse_series_ms.rename("open_time_ms").to_frame().reset_index(names="Time")
    delivery_times = delivery_times[["Time"]].sort_values("Time")

    matched = pd.merge_asof(delivery_times, setpoints, on="Time", direction="backward")
    open_times_s = (matched["open_time_ms"].dropna() / 1000.0).to_numpy()
    if len(open_times_s) == 0:
        return 0.0
    delivered_g = np.round((slope_g_per_s * open_times_s) + offset_g, 4)
    # For water, 1 g is approximately 1 mL.
    return float(delivered_g.sum())


def calculate_consumed_water(session_path: str | os.PathLike[str]) -> float:
    """Calculate the delivered water volume for left/right valves and total session consumption.

    Args:
        session_path (str | os.PathLike[str]): Path to the session directory.

    Returns:
        float: Total water delivered in mL for the session.
    """

    dataset = df_dataset(Path(session_path))["Behavior"]

    rig = AindDynamicForagingRig.model_validate(dataset["InputSchemas"]["Rig"].data)
    left_calibration = rig.calibration.water_valve_left
    right_calibration = rig.calibration.water_valve_right

    left_set_open_time_ms = dataset["HarpBehavior"]["PulseSupplyPort0"].load().data
    right_set_open_time_ms = dataset["HarpBehavior"]["PulseSupplyPort1"].load().data
    output_set_stream = dataset["HarpBehavior"]["OutputSet"].load().data
    writes = output_set_stream[output_set_stream["MessageType"] == "WRITE"]

    left_ml = _calculate_side_volume_ml(
        set_open_time_ms=left_set_open_time_ms["PulseSupplyPort0"],
        delivery_times=writes[writes["SupplyPort0"].fillna(False).astype(bool)],
        slope_g_per_s=float(left_calibration.slope),
        offset_g=float(left_calibration.offset),
    )

    right_ml = _calculate_side_volume_ml(
        set_open_time_ms=right_set_open_time_ms["PulseSupplyPort1"],
        delivery_times=writes[writes["SupplyPort1"].fillna(False).astype(bool)],
        slope_g_per_s=float(right_calibration.slope),
        offset_g=float(right_calibration.offset),
    )

    return left_ml + right_ml

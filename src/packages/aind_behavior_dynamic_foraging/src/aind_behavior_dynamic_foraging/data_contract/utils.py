import os
from pathlib import Path

import numpy as np
import pandas as pd

from aind_behavior_dynamic_foraging.data_contract import dataset as df_dataset
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig


def _calculate_side_volume_ml(
    set_open_time_ms: pd.Series,
    delivery_times: pd.DataFrame,
    slope_g_per_s: float,
    offset_g: float,
) -> float:
    """Estimate delivered volume for one side from set open times and valve-open events.

    Args:
        set_open_time_ms (pd.Series): Time-indexed set open-time values in milliseconds.
        delivery_times (pd.DataFrame): Event rows where the side valve was commanded open.
        slope_g_per_s (float): Calibration slope converting open duration (s) to delivered (g).
        offset_g (float): Calibration offset in grams applied per delivered event.

    Returns:
        float: Total delivered volume in mL for the side.
    """

    delivery_times = delivery_times.reset_index(names="Time")[["Time"]].sort_values("Time")
    if delivery_times.empty:
        return 0.0

    # normalize setpoints to numeric values and reshape into a Time-keyed frame.
    setpoints = (
        pd.to_numeric(set_open_time_ms, errors="coerce")
        .dropna()
        .sort_index()
        .rename("set_open_time_ms")
        .to_frame()
        .reset_index(names="Time")
    )
    if setpoints.empty:
        return 0.0

    # Each valve-open event uses the most recent set open-time configured at or before that event.
    matched = pd.merge_asof(delivery_times, setpoints, on="Time", direction="backward")
    open_times_s = (matched["set_open_time_ms"].dropna() / 1000.0).to_numpy()
    if len(open_times_s) == 0:
        return 0.0

    delivered_g = np.round((slope_g_per_s * open_times_s) + offset_g, 4)
    return float(delivered_g.sum())


def calculate_consumed_water(session_path: str | os.PathLike[str]) -> float:
    """Calculate total delivered water volume across left and right valves for a session.

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

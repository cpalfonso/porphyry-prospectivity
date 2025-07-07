from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pygplates
from gplately import PlateReconstruction
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "extract_subduction_history",
]


def extract_subduction_history(
    point_data: pd.DataFrame,
    subduction_data: pd.DataFrame,
    plate_model: Union[PlateReconstruction, pygplates.TopologicalModel],
    time_window: int = 10,
    n_jobs=None,
    columns_to_ignore: Optional[Sequence[str]] = None,
    drop_instantaneous=False,
) -> pd.DataFrame:
    # We will modify point_data, so make a copy
    point_data = pd.DataFrame(point_data)
    # Don't copy subduction_data

    if isinstance(plate_model, PlateReconstruction):
        plate_model = pygplates.TopologicalModel(
            topological_features=plate_model.topology_features,
            rotation_model=plate_model.rotation_model,
        )

    if columns_to_ignore is None:
        columns_to_ignore = []
    columns_to_ignore = [
        "lon",
        "lat",
        "arc_segment_length (degrees)",
        "trench_normal_angle (degrees)",
        "subducting_plate_ID",
        "trench_plate_ID",
        "distance_from_trench_start (km)",
        "age (Ma)",
        *columns_to_ignore
    ]
    subduction_columns = [i for i in subduction_data.columns if i not in columns_to_ignore]

    cols = []
    for dt in range(time_window):
        cols.extend([f"{colname} (t - {dt})" for colname in subduction_columns])
    tmp = pd.DataFrame(
        columns=cols,
        index=point_data.index,
    )

    for initial_time in point_data["age (Ma)"].round().astype(int).unique():
        point_data_time = point_data[point_data["age (Ma)"].round() == initial_time]
        initial_points = [
            pygplates.PointOnSphere(lat, lon)
            for lat, lon in zip(point_data_time["lat"], point_data_time["lon"])
        ]
        time_span = plate_model.reconstruct_geometry(
            initial_points,
            initial_time=initial_time,
            oldest_time=initial_time + time_window,
            deactivate_points=None,
        )

        for dt, current_time in enumerate(range(initial_time, initial_time + time_window)):
            current_points = time_span.get_geometry_points(current_time)
            current_subduction_data = _nearest_subduction_data(
                points=current_points,
                subduction_data=subduction_data,
                time=current_time,
                n_jobs=n_jobs,
            ).drop(columns=columns_to_ignore, errors="ignore")
            for i, (_, subduction_row) in zip(
                point_data_time.index,
                current_subduction_data.iterrows(),
            ):
                for colname in subduction_row.index:
                    new_colname = f"{colname} (t - {dt})"
                    tmp.at[i, new_colname] = subduction_row[colname]

    for colname in subduction_columns:
        history_columns = [
            f"{colname} (t - {dt})"
            for dt in range(time_window)
        ]
        diff = (tmp[f"{colname} (t - 0)"] - tmp[f"{colname} (t - {time_window - 1})"]) / time_window
        mean = tmp[history_columns].mean(axis="columns")

        for which, series in zip(("diff", "mean"), (diff, mean)):
            new_colname = _format_column(colname, which)
            if new_colname not in point_data.columns:
                point_data[new_colname] = np.nan
            for i, val in series.items():
                point_data.at[i, new_colname] = val

    if drop_instantaneous:
        point_data = point_data.drop(
            columns=subduction_columns,
            errors="ignore",
        )
    return point_data


def _nearest_subduction_data(
    points: Union[Sequence[pygplates.PointOnSphere], pygplates.MultiPointOnSphere],
    subduction_data: pd.DataFrame,
    time=None,
    n_jobs=None,
):
    neigh = NearestNeighbors(n_jobs=n_jobs)

    if time is not None:
        subduction_data = subduction_data[subduction_data["age (Ma)"] == time]
    if subduction_data.shape[0] == 0:
        return pd.DataFrame(
            data=np.nan,
            columns=subduction_data.columns,
            index=range(len(points)),
        )
    x_train = pygplates.MultiPointOnSphere(
        np.array(subduction_data[["lat", "lon"]])
    ).to_xyz_array()
    neigh.fit(x_train)

    x_pred = pygplates.MultiPointOnSphere(points).to_xyz_array()
    distances, indices = neigh.kneighbors(
        x_pred,
        n_neighbors=1,
        return_distance=True,
    )
    distances = np.ravel(distances)
    indices = np.ravel(indices)
    return subduction_data.iloc[indices]


def _format_column(
    feature_name: str,
    which: str,
):
    if which not in {"diff", "mean"}:
        raise ValueError(f"Invalid 'which' parameter: {which}")

    split = feature_name.split()
    if len(split) == 1:
        # No units
        if which == "mean":
            return f"{feature_name}_{which}"
        return f"{feature_name}_{which} (/Myr)"

    # Units
    base = " ".join(split[:-1])
    new_base = f"{base}_{which}"
    unit = split[-1]
    new_unit = unit.replace(")", "/Myr)") if which == "diff" else unit
    return f"{new_base} {new_unit}"

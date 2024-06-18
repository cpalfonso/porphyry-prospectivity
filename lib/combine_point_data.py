"""Functions to clean and combine deposit/non-deposit and random
unlabelled datasets.
"""
import os
import warnings
from sys import stderr
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
)
from joblib import Parallel, delayed
from shapely.geometry import Point

from .misc import (
    _FeatureCollectionInput,
    _PathLike,
    _PathOrDataFrame,
    _RotationModelInput,
)


def combine_point_data(
    deposit_data: _PathOrDataFrame,
    unlabelled_data: _PathOrDataFrame,
    study_area_dir: _PathLike,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    topological_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    output_filename: Optional[_PathLike] = None,
    min_time: float = -np.inf,
    max_time: float = np.inf,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Combine deposit and unlabelled datasets.

    Parameters
    ----------
    deposit_data : str or DataFrame
    unlabelled_data : str or DataFrame
    study_area_dir : str
        Directory containing time-dependent study area polygons
        along subduction zones.
    plate_reconstruction : PlateReconstruction, optional
    static_polygons : FeatureCollection, optional
    topological_features : FeatureCollection, optional
    rotation_model : RotationModel, optional
    output_filename : str, optional
        If provided, write the combined data to a CSV file.
    min_time : float, default: -inf
        Ignore any data from before this time.
    max_time : float, default: inf
        Ignore any data from after this time.
    n_jobs : int, default: 1
        Number of processes to use.
    verbose : bool, default: False
        Print log to stderr.

    Returns
    -------
    DataFrame
        The combined dataset.
    """
    if plate_reconstruction is None:
        plate_reconstruction = PlateReconstruction(
            rotation_model=rotation_model,
            topology_features=topological_features,
            static_polygons=static_polygons,
        )

    if verbose:
        print("Preparing labelled data...", file=stderr)
    deposit_data = _prepare_deposit_data(
        deposit_data=deposit_data,
        plate_reconstruction=plate_reconstruction,
        study_area_dir=study_area_dir,
        min_time=min_time,
        max_time=max_time,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    if verbose:
        print("Done.", file=stderr)
        print("Preparing unlabelled data...", file=stderr)

    unlabelled_data = _prepare_unlabelled_data(
        unlabelled_data=unlabelled_data,
        plate_reconstruction=plate_reconstruction,
        min_time=min_time,
        max_time=max_time,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    if verbose:
        print("Done.", file=stderr)

    combined = pd.concat([deposit_data, unlabelled_data], ignore_index=True)
    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.exists(output_dir):
            if verbose:
                print(
                    "Output directory does not exist; creating now: "
                    + output_dir,
                    file=stderr,
                )
            os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(
                "Writing output to file: "
                + os.path.basename(output_filename),
                file=stderr,
            )
        combined.to_csv(output_filename, index=False)
    return combined


def _prepare_deposit_data(
    deposit_data,
    plate_reconstruction,
    study_area_dir,
    min_time=-np.inf,
    max_time=np.inf,
    n_jobs=1,
    verbose=False,
):
    if isinstance(deposit_data, str):
        if verbose:
            print(
                "Loading deposit data from: " + deposit_data,
                file=stderr,
            )
        deposit_data = pd.read_csv(deposit_data)
    else:
        deposit_data = pd.DataFrame(deposit_data)

    deposit_data = deposit_data.drop(
        columns=["index"],
        errors="ignore",
    )

    deposit_data = deposit_data[
        (deposit_data["age (Ma)"] >= min_time)
        & (deposit_data["age (Ma)"] <= max_time)
    ]

    deposit_data = _partition_and_reconstruct(
        deposit_data=deposit_data,
        plate_reconstruction=plate_reconstruction,
    )
    deposit_data = _clean_deposit_data(
        deposit_data=deposit_data,
        polygons_dir=study_area_dir,
        nprocs=n_jobs,
        verbose=verbose,
    )
    deposit_data = _get_overriding_plate_ids(
        data=deposit_data,
        plate_reconstruction=plate_reconstruction,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return deposit_data


def _partition_and_reconstruct(deposit_data, plate_reconstruction):
    """Partition into plates and reconstruct"""
    rotation_model = plate_reconstruction.rotation_model
    static_polygons = plate_reconstruction.static_polygons

    features = []
    for index, row in deposit_data.iterrows():
        lon = float(row["lon"])
        lat = float(row["lat"])
        name = str(index)
        age = float(row["age (Ma)"])

        geom = pygplates.PointOnSphere(lat, lon)
        feature = pygplates.Feature()
        feature.set_geometry(geom)
        feature.set_valid_time(age, 0.0)
        feature.set_name(name)
        features.append(feature)

    deposit_data["plate_id"] = np.nan
    deposit_data = deposit_data.rename(
        columns={
            "lon": "present_lon",
            "lat": "present_lat",
        }
    )
    deposit_data["lon"] = np.nan
    deposit_data["lat"] = np.nan

    partitioned = pygplates.partition_into_plates(
        partitioning_features=static_polygons,
        rotation_model=rotation_model,
        features_to_partition=features,
    )

    reconstructed = []
    times = set([i.get_valid_time()[0] for i in partitioned])
    for time in times:
        to_reconstruct = [
            i for i in partitioned if i.get_valid_time()[0] == time
        ]
        pygplates.reconstruct(
            to_reconstruct,
            rotation_model,
            reconstructed,
            time,
        )

    for i in reconstructed:
        geom = i.get_reconstructed_geometry()
        feature = i.get_feature()
        lat, lon = geom.to_lat_lon()
        index = int(feature.get_name())
        plate_id = int(feature.get_reconstruction_plate_id())

        deposit_data.at[index, "lon"] = lon
        deposit_data.at[index, "lat"] = lat
        deposit_data.at[index, "plate_id"] = plate_id
    return deposit_data


def _clean_deposit_data(deposit_data, polygons_dir, nprocs, verbose=False):
    times = deposit_data["age (Ma)"].unique()

    with Parallel(nprocs, verbose=int(verbose)) as p:
        out = p(
            delayed(_clean_timestep)(
                (deposit_data[deposit_data["age (Ma)"] == time]).copy(),
                polygons_dir,
                time,
            )
            for time in times
        )
    return pd.concat(out, ignore_index=True)


def _clean_timestep(deposit_data, polygons_dir, time):
    polygons_filename = os.path.join(
        polygons_dir, f"study_area_{time:0.0f}Ma.geojson"
    )
    if not os.path.isfile(polygons_filename):
        polygons_filename = os.path.join(
            polygons_dir, f"study_area_{time:0.0f}Ma.shp"
        )
    polygons = gpd.read_file(polygons_filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        union = polygons.unary_union
    valid = []
    for _, row in deposit_data.iterrows():
        p = Point(row["lon"], row["lat"])
        if union.contains(p):
            valid.append(True)
        else:
            valid.append(False)

    return deposit_data[valid]


def _prepare_unlabelled_data(
    unlabelled_data,
    plate_reconstruction,
    min_time=-np.inf,
    max_time=np.inf,
    n_jobs=1,
    verbose=False,
):
    if isinstance(unlabelled_data, str):
        if verbose:
            print(
                "Loading unlabelled data from file: " + unlabelled_data,
                file=stderr,
            )
        unlabelled_data = pd.read_csv(unlabelled_data)
    else:
        unlabelled_data = pd.DataFrame(unlabelled_data)

    unlabelled_data = unlabelled_data[
        (unlabelled_data["age (Ma)"] >= min_time)
        & (unlabelled_data["age (Ma)"] <= max_time)
    ]
    unlabelled_data = _get_overriding_plate_ids(
        data=unlabelled_data,
        plate_reconstruction=plate_reconstruction,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return unlabelled_data


def _get_overriding_plate_ids(
    data,
    plate_reconstruction,
    n_jobs=1,
    verbose=False,
):
    gdf = data.copy()
    geoms = []
    for _, row in data.iterrows():
        p = Point(row["lon"], row["lat"])
        geoms.append(p)
    gdf["geometry"] = geoms
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    times = data["age (Ma)"].unique()
    times_split = np.array_split(times, n_jobs)
    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        results = parallel(
            delayed(_overriding_plate_multiple_timesteps)(
                gdf=gdf[gdf["age (Ma)"].isin(t)],
                plate_reconstruction=plate_reconstruction,
            )
            for t in times_split
        )
    out = []
    for i in results:
        out.extend(i)
    out = pd.concat(out, ignore_index=True)
    columns_to_keep = set(
        list(data.columns.values) + ["overriding_plate_id"]
    )
    columns_to_drop = set(out.columns.values) - columns_to_keep
    out = out.drop(columns=list(columns_to_drop), errors="ignore")
    return out


def _overriding_plate_multiple_timesteps(gdf, plate_reconstruction):
    topological_features = plate_reconstruction.topology_features
    rotation_model = plate_reconstruction.rotation_model

    times = gdf["age (Ma)"].unique()
    out = []
    for time in times:
        out.append(
            _overriding_plate_timestep(
                gdf=gdf,
                topological_features=topological_features,
                rotation_model=rotation_model,
                time=time,
            )
        )
    return out


def _overriding_plate_timestep(
    gdf,
    topological_features,
    rotation_model,
    time,
):
    gdf = gpd.GeoDataFrame(gdf)

    if not isinstance(topological_features, pygplates.FeatureCollection):
        topological_features = pygplates.FeatureCollection(
            pygplates.FeaturesFunctionArgument(
                topological_features
            ).get_features()
        )
    if not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        reconstruction = PlateReconstruction(
            rotation_model=rotation_model,
            topology_features=topological_features,
        )
        gplot = PlotTopologies(reconstruction)
    gplot.time = time
    topologies = gplot.get_all_topologies()
    topologies.crs = "EPSG:4326"
    topologies["feature_type"] = topologies["feature_type"].astype(str)
    plate_types = {
        "gpml:TopologicalClosedPlateBoundary",
        "gpml:OceanicCrust",
        "gpml:TopologicalNetwork",
    }
    topologies = topologies[topologies["feature_type"].isin(plate_types)]

    gdf_time = gdf[gdf["age (Ma)"] == time]
    if gdf_time.crs is None:
        gdf_time.crs = topologies.crs
    joined = gdf_time.sjoin(topologies, how="inner")
    joined = joined.rename(
        columns={"reconstruction_plate_ID": "overriding_plate_id"}
    )
    return joined

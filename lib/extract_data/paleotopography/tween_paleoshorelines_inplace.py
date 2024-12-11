import os
import re
import warnings
from itertools import product

import numpy as np
import pygplates
from joblib import Parallel, delayed

from . import points_spatial_tree
from . import polygon_processing as pp

DIRNAME = os.path.abspath(os.path.dirname(__file__))
# import paleogeography as pg
# from . import paleogeography as pg
from .load_paleogeography import load_paleogeography

_MODELDIR = "/Users/chris/OneDrive - The University of Sydney (Staff)/Dropbox/Work/EarthBytePlateMotionModel-ARCHIVE/Global_Model_WD_Internal_Release_2019_v2_Clennett_NE_Pacific"

DEFAULT_BASE_DIR = os.path.join(DIRNAME, "..", "Matthews_polygons", "final")
DEFAULT_TWEEN_DIR = os.path.join(DIRNAME, "..", "tween_feature_collections")
DEFAULT_ROTATION_FILENAMES = [
    os.path.join(_MODELDIR, "CombinedFiles", "CombinedRotations.rot")
]
DEFAULT_PRESENTDAY_FILENAME = os.path.join(DIRNAME, "..", "present_day_paleogeography.gmt")
DEFAULT_RESOLUTION = 0.5

AREA_THRESHOLD = 0.001
TIME_STEP = 1
PG_MODE_LIST = ["land", "mountains"]


def tween_paleoshorelines_inplace(
    tween_dir=DEFAULT_TWEEN_DIR,
    basedir=DEFAULT_BASE_DIR,
    rotation_filenames=DEFAULT_ROTATION_FILENAMES,
    presentday_filename=DEFAULT_PRESENTDAY_FILENAME,
    times=None,
    resolution=DEFAULT_RESOLUTION,
    nprocs=1,
    verbose=False,
):
    if not os.path.isdir(tween_dir):
        os.makedirs(tween_dir, exist_ok=True)

    # rotation_model = pygplates.RotationModel(rotation_filenames)

    if times is None:
        srcdir = os.path.join(basedir, "..", "src")
        tmp = [i for i in os.listdir(srcdir) if os.path.isdir(os.path.join(srcdir, i))]
        time_list = [float(re.findall(r"\d+Ma+", tm)[-1][:-2]) for tm in tmp]
        time_list.sort()
    else:
        time_list = sorted(times)

    p = Parallel(nprocs, verbose=20 if verbose else 0)
    p(
        delayed(run)(
            pg_mode,
            t1,
            t2,
            basedir,
            rotation_filenames,
            resolution,
            tween_dir,
        )
        for pg_mode, (t1, t2) in product(
            PG_MODE_LIST,
            list(
                zip(
                    time_list[:-1],
                    time_list[1:],
                )
            ),
        )
    )

    rotation_model = pygplates.RotationModel(rotation_filenames)
    pg_mode_list = ["land", "mountains"]
    t1 = 0
    t2 = min(time_list)
    for pg_mode in pg_mode_list:
        if pg_mode == "mountains":
            env_list = ["m"]
        else:
            env_list = ["lm", "m", "i"]
        pg_features = load_paleogeography(presentday_filename, env_list, env_field="Layer")
        for f in pg_features:
            f.set_valid_time(9999, -9999)
        cf = pp.merge_polygons(pg_features, rotation_model, time=t1, sampling=0.25)
        sieve_polygons_t1 = pp.polygon_area_threshold(cf, AREA_THRESHOLD)

        pg_features = load_paleogeography(basedir, env_list, time=t2)
        for f in pg_features:
            f.set_valid_time(9999, -9999)
        cf = pp.merge_polygons(pg_features, rotation_model, time=t1, sampling=0.25)
        sieve_polygons_t2 = pp.polygon_area_threshold(cf, AREA_THRESHOLD)

        env_list = ["lm", "m", "i", "sm"]
        pg_features = load_paleogeography(basedir, env_list, time=t2)
        for f in pg_features:
            f.set_valid_time(9999, -9999)

        lons,lats = np.meshgrid(np.arange(-180,180+resolution,resolution),np.arange(-90,90+resolution,resolution))
        points = [pygplates.PointOnSphere(lat, lon) for lat, lon in zip(lats.flatten(),lons.flatten())]
        spatial_tree_of_uniform_recon_points = points_spatial_tree.PointsSpatialTree(points)

        if pg_mode == "mountains":
            get_vertical_change_multipoints(
                pg_features,
                t1,
                t2,
                sieve_polygons_t1,
                sieve_polygons_t2,
                points,
                spatial_tree_of_uniform_recon_points,
                rotation_model,
                output_dir=tween_dir,
            )
        else:
            interpolate_paleoshoreline_for_stage(
                pg_features,
                t1,
                t2,
                sieve_polygons_t1,
                sieve_polygons_t2,
                TIME_STEP,
                points,
                spatial_tree_of_uniform_recon_points,
                rotation_model,
                output_dir=tween_dir,
            )


def run(
    pg_mode,
    t1,
    t2,
    basedir,
    rotation_filenames,
    resolution,
    tween_dir,
):
    rotation_model = pygplates.RotationModel(rotation_filenames)

    if pg_mode == 'mountains':
        env_list = ['m']
    else:
        env_list = ['lm','m','i']

    pg_features = load_paleogeography(basedir, env_list, time=t1)
    cf = pp.merge_polygons(pg_features, rotation_model, time=t1, sampling=0.25)
    sieve_polygons_t1 = pp.polygon_area_threshold(cf, AREA_THRESHOLD)

    pg_features = load_paleogeography(basedir, env_list, time=t2)
    for f in pg_features:
        f.set_valid_time(9999, -9999)
    cf = pp.merge_polygons(pg_features, rotation_model, time=t1, sampling=0.25)
    sieve_polygons_t2 = pp.polygon_area_threshold(cf, AREA_THRESHOLD)

    env_list = ["lm", "m", "i", "sm"]
    pg_features = load_paleogeography(basedir, env_list, time=t1)

    lons,lats = np.meshgrid(np.arange(-180,180+resolution,resolution),np.arange(-90,90+resolution,resolution))
    points = [pygplates.PointOnSphere(lat, lon) for lat, lon in zip(lats.flatten(),lons.flatten())]
    spatial_tree_of_uniform_recon_points = points_spatial_tree.PointsSpatialTree(points)

    if pg_mode == "mountains":
        get_vertical_change_multipoints(
            pg_features,
            t1,
            t2,
            sieve_polygons_t1,
            sieve_polygons_t2,
            points,
            spatial_tree_of_uniform_recon_points,
            rotation_model,
            output_dir=tween_dir,
        )
    else:
        interpolate_paleoshoreline_for_stage(
            pg_features,
            t1,
            t2,
            sieve_polygons_t1,
            sieve_polygons_t2,
            TIME_STEP,
            points,
            spatial_tree_of_uniform_recon_points,
            rotation_model,
            output_dir=tween_dir,
        )


def get_masked_multipoint(
    coords, masking_array, plate_partitioner, valid_time=None
):
    # Inputs:
    # a list of coordinates (typically a regular grid of Lat/Long points),
    # an array of indices into that list of coordinates,
    # a set of polygons to use for cookie-cutting
    # a valid time to assign to the output features
    # Returns:
    # a multipoint feature that

    multipoint_feature = pygplates.Feature()

    multipoint_feature.set_geometry(
        pygplates.MultiPointOnSphere(
            zip(
                np.array(coords[0])[masking_array],
                np.array(coords[1])[masking_array],
            )
        )
    )
    (pg_points_masked, dummy) = plate_partitioner.partition_features(
        multipoint_feature,
        partition_return=pygplates.PartitionReturn.separate_partitioned_and_unpartitioned,
        properties_to_copy=[
            pygplates.PartitionProperty.reconstruction_plate_id,
            pygplates.PartitionProperty.valid_time_period,
            pygplates.PropertyName.gml_name,
        ],
    )

    if valid_time is not None:
        for feature in pg_points_masked:
            feature.set_valid_time(valid_time[0], valid_time[1])

    return pg_points_masked


def get_change_masks(
    t1,
    points,
    spatial_tree_of_uniform_recon_points,
    psl_t1,
    psl_t2,
    rotation_model,
):

    distance_to_land_t1, distance_to_psl_t1 = pp.run_grid_pnp(
        t1, points, spatial_tree_of_uniform_recon_points, psl_t1, rotation_model
    )

    distance_to_land_t2, distance_to_psl_t2 = pp.run_grid_pnp(
        t1, points, spatial_tree_of_uniform_recon_points, psl_t2, rotation_model
    )

    # this mask is true where one (and only one) of the indicators is not zero
    # --> delineates points that change environment between time steps
    # msk = np.logical_xor(distance_to_land_t1,distance_to_land_t2)

    regression_msk = np.logical_and(
        distance_to_land_t1 == 0, distance_to_land_t2 > 0
    )

    transgression_msk = np.logical_and(
        distance_to_land_t1 > 0, distance_to_land_t2 == 0
    )

    always_land_msk = np.logical_and(
        distance_to_land_t1 == 0, distance_to_land_t2 == 0
    )

    return (
        distance_to_land_t1,
        distance_to_psl_t1,
        distance_to_land_t2,
        distance_to_psl_t2,
        regression_msk,
        transgression_msk,
        always_land_msk,
    )


# function to run point in/near polygon test for two successive time slices
def get_change_mask_multipoints(
    pg_features,
    t1,
    t2,
    psl_t1,
    psl_t2,
    points,
    spatial_tree_of_uniform_recon_points,
    rotation_model,
    output_dir=DEFAULT_TWEEN_DIR,
):

    print("Working on interpolation from %0.2f Ma to %0.2f Ma ....." % (t1, t2))

    plate_partitioner = pygplates.PlatePartitioner(
        pg_features, rotation_model, reconstruction_time=t1
    )

    (
        distance_to_land_t1,
        distance_to_psl_t1,
        distance_to_land_t2,
        distance_to_psl_t2,
        regression_msk,
        transgression_msk,
        always_land_msk,
    ) = get_change_masks(
        t1,
        points,
        spatial_tree_of_uniform_recon_points,
        psl_t1,
        psl_t2,
        rotation_model,
    )

    coords = zip(*[point.to_lat_lon() for point in points])

    pg_points_regression = get_masked_multipoint(
        coords, regression_msk, plate_partitioner, valid_time=[t2, t1 + 0.01]
    )
    pg_points_transgression = get_masked_multipoint(
        coords, transgression_msk, plate_partitioner, valid_time=[t2, t1 + 0.01]
    )
    pg_points_always_land = get_masked_multipoint(
        coords, always_land_msk, plate_partitioner, valid_time=[t2, t1]
    )

    pygplates.FeatureCollection(pg_points_regression).write(
        os.path.join(
            output_dir, "mountain_regression_%0.2fMa_%0.2fMa.gpmlz" % (t1, t2)
        )
    )
    pygplates.FeatureCollection(pg_points_transgression).write(
        os.path.join(
            output_dir,
            "mountain_transgression_%0.2fMa_%0.2fMa.gpmlz" % (t1, t2),
        )
    )
    pygplates.FeatureCollection(pg_points_always_land).write(
        os.path.join(
            output_dir, "mountain_stable_%0.2fMa_%0.2fMa.gpmlz" % (t1, t2)
        )
    )


def get_vertical_change_multipoints(
    pg_features,
    t1,
    t2,
    psl_t1,
    psl_t2,
    points,
    spatial_tree_of_uniform_recon_points,
    rotation_model,
    output_dir=DEFAULT_TWEEN_DIR,
):
    # NOT WORKING DUE TO LACK OF SUPPORT FOR SCALAR COVERAGES

    print("Working on interpolation from %0.2f Ma to %0.2f Ma ....." % (t1, t2))

    plate_partitioner = pygplates.PlatePartitioner(
        pg_features, rotation_model, reconstruction_time=t1
    )

    (
        distance_to_land_t1,
        distance_to_psl_t1,
        distance_to_land_t2,
        distance_to_psl_t2,
        regression_msk,
        transgression_msk,
        always_land_msk,
    ) = get_change_masks(
        t1,
        points,
        spatial_tree_of_uniform_recon_points,
        psl_t1,
        psl_t2,
        rotation_model,
    )

    coords = list(zip(*[point.to_lat_lon() for point in points]))

    pg_points_regression = get_masked_multipoint(
        coords, regression_msk, plate_partitioner, valid_time=[t2, t1]
    )
    pg_points_transgression = get_masked_multipoint(
        coords, transgression_msk, plate_partitioner, valid_time=[t2, t1]
    )
    pg_points_always_land = get_masked_multipoint(
        coords, always_land_msk, plate_partitioner, valid_time=[t2, t1]
    )
    pygplates.FeatureCollection(pg_points_regression).write(
        os.path.join(
            output_dir, "mountain_regression_%0.2fMa_%0.2fMa.gpmlz" % (t1, t2)
        )
    )
    pygplates.FeatureCollection(pg_points_transgression).write(
        os.path.join(
            output_dir,
            "mountain_transgression_%0.2fMa_%0.2fMa.gpmlz" % (t1, t2),
        )
    )
    pygplates.FeatureCollection(pg_points_always_land).write(
        os.path.join(
            output_dir, "mountain_stable_%0.2fMa_%0.2fMa.gpmlz" % (t1, t2)
        )
    )


def interpolate_paleoshoreline_for_stage(
    pg_features,
    t1,
    t2,
    psl_t1,
    psl_t2,
    time_step,
    points,
    spatial_tree_of_uniform_recon_points,
    rotation_model,
    output_dir=DEFAULT_TWEEN_DIR,
):
    """Function to run point in/near polygon test for two successive time slices."""

    print("Working on interpolation from %0.2f Ma to %0.2f Ma ....." % (t1, t2))

    plate_partitioner = pygplates.PlatePartitioner(
        pg_features, rotation_model, reconstruction_time=t1
    )

    (
        distance_to_land_t1,
        distance_to_psl_t1,
        distance_to_land_t2,
        distance_to_psl_t2,
        regression_msk,
        transgression_msk,
        always_land_msk,
    ) = get_change_masks(
        t1,
        points,
        spatial_tree_of_uniform_recon_points,
        psl_t1,
        psl_t2,
        rotation_model,
    )

    # normalised distance derivation
    # for each point, divide the distance to shoreline at t0 by the total distance to both shorelines
    # --> if the point is halfway between the shorelines, value will be 0.5
    #     if the point is closer to the t1 shoreline, the value will be less than 0.5
    #     all values will be between 0 and 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psl_dist_norm = np.divide(
            distance_to_psl_t1, (distance_to_psl_t1 + distance_to_psl_t2)
        )

    t_diff = t2 - t1

    # before looping over the time steps, create an empty list to put the features in
    pg_points_land_list = []
    pg_points_marine_list = []

    # don't need to do t2 itself, since this will be first step in next iteration
    for reconstruction_time in np.arange(t1, t2, time_step):

        if reconstruction_time == t1:
            land_points = np.where(distance_to_land_t1 == 0)[0]

        else:
            # normalised time, in range 0 to 1 between start and end of stage
            t_norm = (reconstruction_time - t1) / t_diff

            is_transgressing_land_msk = np.less_equal(psl_dist_norm, t_norm)

            is_regressing_land_msk = np.greater_equal(psl_dist_norm, t_norm)

            land_points = np.where(
                np.logical_or(
                    np.logical_or(
                        np.logical_and(is_regressing_land_msk, regression_msk),
                        np.logical_and(
                            is_transgressing_land_msk, transgression_msk
                        ),
                    ),
                    always_land_msk,
                )
            )[0]

        marine_mask = np.ones(distance_to_land_t1.shape, dtype=bool)
        marine_mask[land_points] = False
        marine_points = np.where(marine_mask)

        coords = list(zip(*[point.to_lat_lon() for point in points]))

        pg_points_land = get_masked_multipoint(
            coords,
            land_points,
            plate_partitioner,
            valid_time=[
                reconstruction_time + (time_step / 2.0),
                reconstruction_time - (time_step / 2.0) + 0.01,
            ],
        )

        pg_points_marine = get_masked_multipoint(
            coords,
            marine_points,
            plate_partitioner,
            valid_time=[
                reconstruction_time + (time_step / 2.0),
                reconstruction_time - (time_step / 2.0) + 0.01,
            ],
        )

        # append the point features for this time to the overall list
        pg_points_land_list += pg_points_land
        pg_points_marine_list += pg_points_marine

    pygplates.FeatureCollection(pg_points_land_list).write(
        os.path.join(
            output_dir, "tweentest_land_%0.2fMa_%0.2fMa.gpmlz" % (t1, t2)
        )
    )
    pygplates.FeatureCollection(pg_points_marine_list).write(
        os.path.join(
            output_dir, "tweentest_ocean_%0.2fMa_%0.2fMa.gpmlz" % (t1, t2)
        )
    )

import os
from sys import stderr

import numpy as np
import xarray as xr

DEFAULT_REFERENCE_THICKNESS = 32.0e3  # m
DEFAULT_MIN_ELEVATION = -200.0  # m


def calculate_crustal_thickness(
    time,
    input_dir,
    output_dir,
    reference_thickness=DEFAULT_REFERENCE_THICKNESS,
    min_elevation=DEFAULT_MIN_ELEVATION,
    verbose=False,
):
    time = int(time)
    input_filenames = os.listdir(input_dir)
    for input_filename in input_filenames:
        if input_filename.startswith("paleotopo_") and input_filename.endswith(
            "d_{}.00Ma.nc".format(time)
        ):
            break
    else:
        raise FileNotFoundError(
            "Could not find input file for time {} Ma".format(time)
            + " in input directory: {}".format(input_dir)
        )
    input_filename = os.path.join(input_dir, input_filename)

    output_filename = os.path.join(
        output_dir, "crustal_thickness_{}Ma.nc".format(time)
    )

    with xr.open_dataset(input_filename) as dset:
        z = np.array(dset["z"])
        moho = topo2moho(z, reference_thickness)
        thickness = z - moho
        thickness[z < min_elevation] = np.nan
        out = dset.copy(data={"z": thickness})

    if verbose:
        print(
            "\t- Writing output file: " + os.path.basename(output_filename),
            file=stderr,
        )
    out.to_netcdf(
        output_filename,
        encoding={
            "z": {
                "zlib": True,
                "dtype": "float32",
            },
        },
    )


def topo2moho(elevation, ref_depth=32000.0, rhoM=3300.0, rhoC=2700.0):
    """Calculate (air-loaded) isostatic crustal thickness.

    Parameters
    ----------
    elevation : array_like or float
        Elevation (m above sea level) of the topographic surface.
    ref_depth : float, default: 32000
        Reference crustal thickness at `elevation = 0` (m).
    rhoM : float, default: 3300
        Mantle density (kg/m3).
    rhoC : float, default: 2700
        Crust density (kg/m3).
    Returns
    -------
    moho_depth : array_like or float
        The elevation (m) of the Moho; crustal thickness can be calculated as
        `elevation - moho_depth`.
    """
    base_surface = (elevation * rhoC) / (rhoM - rhoC)
    moho_depth = -base_surface - ref_depth
    return moho_depth

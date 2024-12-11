"""Generate crustal CO2 grids from seafloor age and bottom-water temperature
models. Output grids are in units of megatons per square metre (Mt/m^2).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d, NearestNDInterpolator

_DIRPATH = Path(__file__).absolute().parent

DEFAULT_CO2_FILEPATH = Path(_DIRPATH, "age_bwt_co2_model_bilinear_log.nc")
DEFAULT_CO2_FILENAME = str(DEFAULT_CO2_FILEPATH)

DEFAULT_BWT_FILEPATH = Path(_DIRPATH, "age_deep-ocean-temp_Scotese2021.txt")
DEFAULT_BWT_FILENAME = str(DEFAULT_BWT_FILEPATH)


def calculate_crustal_co2(
    times,
    seafloor_age_dir,
    output_dir,
    n_jobs=1,
    bwt_filename=DEFAULT_BWT_FILENAME,
    co2_filename=DEFAULT_CO2_FILENAME,
):
    seafloor_age_dir = Path(seafloor_age_dir)
    if not seafloor_age_dir.is_dir():
        raise FileNotFoundError(
            f"Seafloor age directory not found: {seafloor_age_dir}"
        )

    times = np.array(times)
    times_grid = np.column_stack([times] * 20)
    dt_grid = np.row_stack([np.arange(20)] * times.size)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get bottom water temperature
    df_bwt = pd.read_csv(
        bwt_filename,
        header=0,
        names=("age", "bwt"),
        sep=r"\s+",
        skiprows=1,
    )
    # Average BWT over next 20Myr after given time
    bwts = np.interp(
        times_grid - dt_grid,
        df_bwt["age"],
        df_bwt["bwt"],
    ).mean(axis=-1)

    # Create interpolator (seafloor age, BWT) -> CO2 content
    co2_interp = create_co2_interpolator(co2_filename)

    if n_jobs == 1:
        for time, bwt in zip(times, bwts):
            create_co2_grid(
                bwt=bwt,
                co2_interpolator=co2_interp,
                seafloor_age=Path(
                    seafloor_age_dir,
                    f"seafloor_age_{time:0.0f}Ma.nc",
                ),
                output_filename=Path(
                    output_dir,
                    f"crustal_co2_{time:0.0f}Ma.nc",
                ),
            )
    else:
        import joblib

        with joblib.Parallel(n_jobs) as parallel:
            parallel(
                joblib.delayed(create_co2_grid)(
                    bwt=bwt,
                    co2_interpolator=co2_interp,
                    seafloor_age=Path(
                        seafloor_age_dir,
                        f"seafloor_age_{time:0.0f}Ma.nc",
                    ),
                    output_filename=Path(
                        output_dir,
                        f"crustal_co2_{time:0.0f}Ma.nc",
                    ),
                )
                for time, bwt in zip(times, bwts)
            )


def create_co2_grid(
    bwt,
    co2_interpolator,
    seafloor_age,
    output_filename=None,
):
    if not isinstance(seafloor_age, xr.Dataset):
        seafloor_age = xr.load_dataset(seafloor_age)
    if "seafloor_age" in seafloor_age.data_vars:
        varname = "seafloor_age"
    else:
        varname = "z"
    age_arr = np.array(seafloor_age[varname])

    non_nans = ~np.isnan(age_arr)
    age_shape = np.shape(age_arr)
    age_arr = age_arr[non_nans]

    co2_grid = np.full(age_shape, np.nan)
    # CO2 in wt %
    co2_values = co2_interpolator(
        age=age_arr,
        bwt=np.full_like(age_arr, bwt),
    )
    co2_grid[non_nans] = co2_values
    co2_grid = np.reshape(co2_grid, age_shape)

    # Convert to t/m^2
    co2_grid *= 0.01  # wt % to wt fraction
    # co2_grid *= 2900.0  # kg/m^3 (multiply by density)
    co2_grid *= 2266.0  # kg/m^3 (multiply by density - corrected value)
    co2_grid *= 450.0  # kg/m^2 (multiply by thickness of carbon-rich layer)
    carbon_grid = co2_grid * (12.0107 / 44.0095)  # kg/m^2 (convert CO2 to equivalent C)
    carbon_grid *= 1.0e-3  # t/m^2

    out = seafloor_age[[varname]].copy(
        deep=True,
        data={varname: carbon_grid},
    ).rename_vars({varname: "z"})
    if output_filename is None:
        return out
    out.to_netcdf(
        path=output_filename,
        encoding={
            i: {"zlib": True}
            for i in out.data_vars
        }
    )
    return None


def create_bwt_interpolator(
    filename=DEFAULT_BWT_FILENAME,
    kind="linear",
    **kwargs
):
    header = kwargs.pop("header", 0)
    names = kwargs.pop("names", ("age", "bwt"))
    sep = kwargs.pop("sep", r"\s+")
    skiprows = kwargs.pop("skiprows", 1)

    df = pd.read_csv(
        filename,
        header=header,
        names=names,
        sep=sep,
        skiprows=skiprows,
        **kwargs,
    )
    age = np.array(df[names[0]])
    bwt = np.array(df[names[1]])

    indices = np.argsort(age)
    age = age[indices]
    bwt = bwt[indices]

    interp = interp1d(
        x=age,
        y=bwt,
        kind=kind,
        assume_sorted=True,
        bounds_error=False,
        fill_value=(bwt[0], bwt[-1]),
    )
    return interp


def create_co2_interpolator(filename=DEFAULT_CO2_FILENAME):
    dset = xr.load_dataset(filename)
    ages = np.array(dset["x"])
    bwts = np.array(dset["y"])
    co2s = np.array(dset["z"])

    ages, bwts = np.meshgrid(ages, bwts)
    ages = np.ravel(ages)
    bwts = np.ravel(bwts)

    interp = NearestNDInterpolator(
        x=np.column_stack((bwts, ages)),
        y=np.ravel(co2s),
    )

    def f(age, bwt):
        age = np.clip(age, np.nanmin(ages), np.nanmax(ages))
        bwt = np.clip(bwt, np.nanmin(bwts), np.nanmax(bwts))
        return interp(np.column_stack((bwt, age)))

    return f

import os
import warnings
from multiprocessing import cpu_count
from typing import (
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from gplately import (
        PlateReconstruction,
        PlotTopologies,
        Raster,
    )
    from gplately.plot import (
        SubductionTeeth,
        shapelify_feature_lines,
    )
    from gplately.plot import _meridian_from_ax
    from gplately.tools import lonlat2xyz
from joblib import (
    Parallel,
    delayed,
)
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import (
    LogFormatterSciNotation,
    LogLocator,
)
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
)

from ._extract_erodep import (
    extract_lat_lon,
    _erorate_timestep,
)
from ..misc import (
    _PathLike,
    _PathOrDataFrame,
    load_data,
    reconstruct_by_topologies,
)
from ..visualisation import (
    SCATTER_KWARGS as SCATTER_KW,
    _add_deposits,
)

__all__ = [
    "TRANSFORM",
    "FIGSIZE",
    "FONTSIZE",
    "TITLESIZE",
    "TICKSIZE",
    "COASTLINES_KW",
    "IMSHOW_KW",
    "EROSION_KW",
    "LIKELIHOOD_KW",
    "TOPOLOGIES_KW",
    "RIDGES_KW",
    "TEETH_KW",
    "SAVEFIG_KW",
    "SCATTER_KW",
    "plot_combined_maps",
    "plot_combined",
    "plot_erosion_maps",
    "plot_erosion",
    "plot_erosion_rate_maps",
    "plot_erosion_rate",
    "plot_likelihood_maps",
    "plot_likelihood",
    "_prepare_map",
    "_add_deposits",
]

TRANSFORM = ccrs.Geodetic()

FIGSIZE = (12, 7.5)
FONTSIZE = 18
TITLESIZE = FONTSIZE * 1.8
TICKSIZE = FONTSIZE * 0.6

COASTLINES_KW = {
    "facecolor": "lightgrey",
    "edgecolor": "lightgrey",
    "tessellate_degrees": 0.1,
    "zorder": 0,
}
IMSHOW_KW = {
    "cmap": colormaps["inferno"],
    "zorder": COASTLINES_KW["zorder"] + 1,
}
EROSION_KW = {
    **IMSHOW_KW,
    "norm": LogNorm(100, 10000),
}
ERORATE_KW = {
    "cmap": "RdBu",
    "zorder": IMSHOW_KW["zorder"],
    "vmin": -300,
    "vmax": 300,
}
LIKELIHOOD_KW = {
    **IMSHOW_KW,
    "cmap": colormaps["inferno_r"],
    "norm": LogNorm(1.0e-5, 5.0e-4),
}
COMBINED_KW = {
    **IMSHOW_KW,
    "cmap": "RdYlBu_r",
    "vmin": 0,
    "vmax": 100,
}
COMBINED_KW = {
    "probability": {
        **IMSHOW_KW,
        "cmap": "RdYlBu_r",
        "vmin": 0,
        "vmax": 100,
    },
    "likelihood": {
        **IMSHOW_KW,
        "cmap": "RdYlBu_r",
        # "norm": LogNorm(1.0e-6, 1.0),
        "vmin": 0,
        "vmax": 1,
    }
}
TOPOLOGIES_KW = {
    "color": "black",
    "linewidth": 0.8,
    "tessellate_degrees": COASTLINES_KW["tessellate_degrees"],
    "zorder": IMSHOW_KW["zorder"] + 1,
}
RIDGES_KW = {
    **TOPOLOGIES_KW,
    "color": "red",
    "zorder": TOPOLOGIES_KW["zorder"] + 1,
}
TEETH_KW = {
    "color": TOPOLOGIES_KW["color"],
    "zorder": TOPOLOGIES_KW["zorder"],
    "size": 8,
    "spacing": 0.13,
    "aspect": 0.7,
}
SAVEFIG_KW = {
    "dpi": 300,
    "bbox_inches": "tight",
}
SCATTER_KW = {
    **SCATTER_KW,
    "markerfacecolor": "white",
    "zorder": TEETH_KW["zorder"] + 1,
}


def plot_erosion_maps(
    times: Sequence[float],
    input_dir: _PathLike,
    output_dir: str,
    gplot: Optional[PlotTopologies] = None,
    topology_filenames: Optional[Union[str, Sequence[str]]] = None,
    rotation_filenames: Optional[Union[str, Sequence[str]]] = None,
    coastline_filenames: Optional[Union[str, Sequence[str]]] = None,
    projection: ccrs.Projection = ccrs.Mollweide(),
    n_jobs: int = 1,
    verbose: int = 1,
    output_template: str = r"image_{:0.0f}Ma.png",
    deposits: Optional[_PathOrDataFrame] = None,
):
    if gplot is None:
        if (
            topology_filenames is None
            or rotation_filenames is None
            or coastline_filenames is None
        ):
            raise TypeError(
                "Either `gplot` or all of "
                "`topology_filenames`, `rotation_filenames`, "
                "and `coastline_filenames` must be provided."
            )
        reconstruction = PlateReconstruction(
            rotation_filenames,
            topology_filenames,
        )
        gplot = PlotTopologies(reconstruction, coastlines=coastline_filenames)

    if deposits is not None:
        deposits = load_data(deposits, verbose=verbose > 0)
        deposits = deposits[deposits["label"] == "positive"]
        deposits = reconstruct_by_topologies(
            data=deposits,
            plate_reconstruction=gplot.plate_reconstruction,
            times=times,
            verbose=verbose > 0,
        )

    if n_jobs == 1:
        for time in times:
            output_filename = os.path.join(
                output_dir,
                output_template.format(time),
            )
            plot_erosion(
                time=time,
                gplot=gplot,
                input_dir=input_dir,
                projection=projection,
                output_filename=output_filename,
                deposits=(
                    None if deposits is None
                    else deposits[[
                        f"lon_{time:0.0f}",
                        f"lat_{time:0.0f}",
                        "lon",
                        "lat",
                        "label",
                        "age (Ma)",
                    ]]
                ),
            )
    else:
        if n_jobs == 0:
            raise ValueError("`n_jobs` must not be zero")
        if n_jobs < 0:
            n_jobs = cpu_count() + n_jobs + 1
        times_split = np.array_split(times, n_jobs)
        with Parallel(n_jobs, verbose=verbose) as parallel:
            parallel(
                delayed(_plot_erosion_subset)(
                    times=t,
                    gplot=gplot,
                    topology_filenames=topology_filenames,
                    rotation_filenames=rotation_filenames,
                    coastline_filenames=coastline_filenames,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    projection=projection,
                    output_template=output_template,
                    deposits=(
                        None if deposits is None
                        else deposits[[
                            *[f"lon_{i:0.0f}" for i in t],
                            *[f"lat_{i:0.0f}" for i in t],
                            "lon",
                            "lat",
                            "label",
                            "age (Ma)",
                        ]]
                    ),
                )
                for t in times_split
            )


def _plot_erosion_subset(
    times,
    gplot,
    topology_filenames,
    rotation_filenames,
    coastline_filenames,
    input_dir,
    output_dir,
    projection = ccrs.Mollweide(),
    output_template=r"image_{:0.0f}Ma.png",
    deposits: Optional[_PathOrDataFrame] = None,
):
    if gplot is None:
        reconstruction = PlateReconstruction(
            rotation_filenames,
            topology_filenames,
            static_polygons=coastline_filenames,
        )
        gplot = PlotTopologies(
            reconstruction,
            coastlines=coastline_filenames,
        )

    if deposits is not None and (not isinstance(deposits, pd.DataFrame)):
        deposits = pd.read_csv(deposits)
    for time in times:
        output_filename = os.path.join(
            output_dir,
            output_template.format(time),
        )
        plot_erosion(
            time=time,
            gplot=gplot,
            input_dir=input_dir,
            projection=projection,
            output_filename=output_filename,
            deposits=deposits,
        )


def plot_erosion(
    time: float,
    gplot: PlotTopologies,
    input_dir: _PathLike,
    projection: ccrs.Projection = ccrs.Mollweide(),
    output_filename: Optional[str] = None,
    deposits: Optional[_PathOrDataFrame] = None,
) -> Optional[Figure]:
    gplot.time = time
    input_filename = os.path.join(
        input_dir,
        f"erosion_grid_{time:0.0f}Ma.nc",
    )
    erodep = Raster(input_filename)

    fig, ax, cax = _prepare_map(gplot=gplot, projection=projection, time=time)
    im = erodep.imshow(
        ax=ax,
        **EROSION_KW,
    )

    if deposits is not None:
        deposits = load_data(deposits)
        if (
            f"lon_{time:0.0f}" not in deposits.columns
            or f"lat_{time:0.0f}" not in deposits.columns
        ):
            deposits = reconstruct_by_topologies(
                data=deposits,
                plate_reconstruction=gplot.plate_reconstruction,
                times=np.arange(time, deposits["age (Ma)"].round().max() + 1),
            )
        _add_deposits(
            ax=ax,
            deposits=deposits,
            time=time,
            **SCATTER_KW,
        )

    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )
    locator = LogLocator(subs=(1, 2, 3, 4, 5))
    formatter = LogFormatterSciNotation(
        labelOnlyBase=False,
        minor_thresholds=(2, 0.3),
    )
    cbar.ax.xaxis.set_major_locator(locator)
    cbar.ax.xaxis.set_major_formatter(formatter)

    cbar.ax.tick_params(labelsize=TICKSIZE)
    cbar.ax.set_xlabel("Cumulative erosion (m)", fontsize=FONTSIZE)

    if output_filename is not None:
        fig.savefig(output_filename, **SAVEFIG_KW)
        plt.close(fig)
        fig = None
    return fig


def plot_erosion_rate_maps(
    times: Sequence[float],
    input_dir: _PathLike,
    output_dir: str,
    gplot: Optional[PlotTopologies] = None,
    topology_filenames: Optional[Union[str, Sequence[str]]] = None,
    rotation_filenames: Optional[Union[str, Sequence[str]]] = None,
    coastline_filenames: Optional[Union[str, Sequence[str]]] = None,
    projection: ccrs.Projection = ccrs.Mollweide(),
    n_jobs: int = 1,
    verbose: int = 1,
    output_template: str = r"image_{:0.0f}Ma.png",
    deposits: Optional[_PathOrDataFrame] = None,
):
    if gplot is None:
        if (
            topology_filenames is None
            or rotation_filenames is None
            or coastline_filenames is None
        ):
            raise TypeError(
                "Either `gplot` or all of "
                "`topology_filenames`, `rotation_filenames`, "
                "and `coastline_filenames` must be provided."
            )
        reconstruction = PlateReconstruction(
            rotation_filenames,
            topology_filenames,
        )
        gplot = PlotTopologies(reconstruction, coastlines=coastline_filenames)

    if deposits is not None:
        deposits = load_data(deposits, verbose=verbose > 0)
        deposits = deposits[deposits["label"] == "positive"]
        deposits = reconstruct_by_topologies(
            data=deposits,
            plate_reconstruction=gplot.plate_reconstruction,
            times=times,
            verbose=verbose > 0,
        )

    if n_jobs == 1:
        for time in times:
            output_filename = os.path.join(
                output_dir,
                output_template.format(time),
            )
            plot_erosion_rate(
                time=time,
                gplot=gplot,
                input_dir=input_dir,
                projection=projection,
                output_filename=output_filename,
                deposits=(
                    None if deposits is None
                    else deposits[[
                        f"lon_{time:0.0f}",
                        f"lat_{time:0.0f}",
                        "lon",
                        "lat",
                        "label",
                        "age (Ma)",
                    ]]
                ),
            )
    else:
        if n_jobs == 0:
            raise ValueError("`n_jobs` must not be zero")
        if n_jobs < 0:
            n_jobs = cpu_count() + n_jobs + 1
        times_split = np.array_split(times, n_jobs)
        with Parallel(n_jobs, verbose=verbose) as parallel:
            parallel(
                delayed(_plot_erosion_rate_subset)(
                    times=t,
                    gplot=gplot,
                    topology_filenames=topology_filenames,
                    rotation_filenames=rotation_filenames,
                    coastline_filenames=coastline_filenames,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    projection=projection,
                    output_template=output_template,
                    deposits=(
                        None if deposits is None
                        else deposits[[
                            *[f"lon_{i:0.0f}" for i in t],
                            *[f"lat_{i:0.0f}" for i in t],
                            "lon",
                            "lat",
                            "label",
                            "age (Ma)",
                        ]]
                    ),
                )
                for t in times_split
            )


def _plot_erosion_rate_subset(
    times,
    gplot,
    topology_filenames,
    rotation_filenames,
    coastline_filenames,
    input_dir,
    output_dir,
    projection = ccrs.Mollweide(),
    output_template=r"image_{:0.0f}Ma.png",
    deposits: Optional[_PathOrDataFrame] = None,
):
    if gplot is None:
        reconstruction = PlateReconstruction(
            rotation_filenames,
            topology_filenames,
            static_polygons=coastline_filenames,
        )
        gplot = PlotTopologies(
            reconstruction,
            coastlines=coastline_filenames,
        )

    if deposits is not None and (not isinstance(deposits, pd.DataFrame)):
        deposits = pd.read_csv(deposits)
    for time in times:
        output_filename = os.path.join(
            output_dir,
            output_template.format(time),
        )
        plot_erosion_rate(
            time=time,
            gplot=gplot,
            input_dir=input_dir,
            projection=projection,
            output_filename=output_filename,
            deposits=deposits,
        )


def plot_erosion_rate(
    time: float,
    gplot: PlotTopologies,
    input_dir: _PathLike,
    projection: ccrs.Projection = ccrs.Mollweide(),
    output_filename: Optional[str] = None,
    deposits: Optional[_PathOrDataFrame] = None,
) -> Optional[Figure]:
    gplot.time = time
    lats, lons = extract_lat_lon(input_dir)
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    erorate = Raster(
        _erorate_timestep(time, input_dir),
        extent=extent,
        time=0,
        plate_reconstruction=gplot.plate_reconstruction,
    )
    erorate = erorate.reconstruct(time)

    fig, ax, cax = _prepare_map(gplot=gplot, projection=projection, time=time)
    im = erorate.imshow(
        ax=ax,
        **ERORATE_KW,
    )

    if deposits is not None:
        deposits = load_data(deposits)
        if (
            f"lon_{time:0.0f}" not in deposits.columns
            or f"lat_{time:0.0f}" not in deposits.columns
        ):
            deposits = reconstruct_by_topologies(
                data=deposits,
                plate_reconstruction=gplot.plate_reconstruction,
                times=np.arange(time, deposits["age (Ma)"].round().max() + 1),
            )
        _add_deposits(
            ax=ax,
            deposits=deposits,
            time=time,
            **SCATTER_KW,
        )

    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )
    cbar.ax.tick_params(labelsize=TICKSIZE)
    cbar.ax.set_xlabel(r"Erosion rate ($\mathrm{m \; {Myr}^{-1}}$)", fontsize=FONTSIZE)

    if output_filename is not None:
        fig.savefig(output_filename, **SAVEFIG_KW)
        plt.close(fig)
        fig = None
    return fig


def plot_likelihood_maps(
    times: Sequence[float],
    input_dir: _PathLike,
    output_dir: str,
    gplot: Optional[PlotTopologies] = None,
    topology_filenames: Optional[Union[str, Sequence[str]]] = None,
    rotation_filenames: Optional[Union[str, Sequence[str]]] = None,
    coastline_filenames: Optional[Union[str, Sequence[str]]] = None,
    projection: ccrs.Projection = ccrs.Mollweide(),
    n_jobs: int = 1,
    verbose: int = 10,
    output_template: str = r"image_{:0.0f}Ma.png",
    deposits: Optional[_PathOrDataFrame] = None,
):
    if gplot is None:
        if (
            topology_filenames is None
            or rotation_filenames is None
            or coastline_filenames is None
        ):
            raise TypeError(
                "Either `gplot` or all of "
                "`topology_filenames`, `rotation_filenames`, "
                "and `coastline_filenames` must be provided."
            )
        reconstruction = PlateReconstruction(
            rotation_filenames,
            topology_filenames,
        )
        gplot = PlotTopologies(reconstruction, coastlines=coastline_filenames)

    if deposits is not None:
        deposits = load_data(deposits, verbose=verbose > 0)
        deposits = deposits[deposits["label"] == "positive"]
        deposits = reconstruct_by_topologies(
            data=deposits,
            plate_reconstruction=gplot.plate_reconstruction,
            times=times,
            verbose=verbose > 0,
        )

    if n_jobs == 1:
        for time in times:
            output_filename = os.path.join(
                output_dir,
                output_template.format(time),
            )
            plot_likelihood(
                time=time,
                gplot=gplot,
                input_dir=input_dir,
                projection=projection,
                output_filename=output_filename,
                deposits=(
                    None if deposits is None
                    else deposits[[
                        f"lon_{time:0.0f}",
                        f"lat_{time:0.0f}",
                        "lon",
                        "lat",
                        "label",
                        "age (Ma)",
                    ]]
                ),
            )
    else:
        if n_jobs == 0:
            raise ValueError("`n_jobs` must not be zero")
        if n_jobs < 0:
            n_jobs = cpu_count() + n_jobs + 1
        times_split = np.array_split(times, n_jobs)
        with Parallel(n_jobs, verbose=verbose) as parallel:
            parallel(
                delayed(_plot_likelihood_subset)(
                    times=t,
                    gplot=gplot,
                    topology_filenames=topology_filenames,
                    rotation_filenames=rotation_filenames,
                    coastline_filenames=coastline_filenames,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    projection=projection,
                    output_template=output_template,
                    deposits=(
                        None if deposits is None
                        else deposits[[
                            *[f"lon_{i:0.0f}" for i in t],
                            *[f"lat_{i:0.0f}" for i in t],
                            "lon",
                            "lat",
                            "label",
                            "age (Ma)",
                        ]]
                    ),
                )
                for t in times_split
            )


def _plot_likelihood_subset(
    times,
    gplot,
    topology_filenames,
    rotation_filenames,
    coastline_filenames,
    input_dir,
    output_dir,
    projection = ccrs.Mollweide(),
    output_template=r"image_{:0.0f}Ma.png",
    deposits: Optional[_PathOrDataFrame] = None,
):
    if gplot is None:
        reconstruction = PlateReconstruction(
            rotation_filenames,
            topology_filenames,
            static_polygons=coastline_filenames,
        )
        gplot = PlotTopologies(
            reconstruction,
            coastlines=coastline_filenames,
        )

    if deposits is not None and (not isinstance(deposits, pd.DataFrame)):
        deposits = pd.read_csv(deposits)
    for time in times:
        output_filename = os.path.join(
            output_dir,
            output_template.format(time),
        )
        plot_likelihood(
            time=time,
            gplot=gplot,
            input_dir=input_dir,
            projection=projection,
            output_filename=output_filename,
            deposits=deposits,
        )


def plot_likelihood(
    time: float,
    gplot: PlotTopologies,
    input_dir: _PathLike,
    projection: ccrs.Projection = ccrs.Mollweide(),
    output_filename: Optional[str] = None,
    deposits: Optional[_PathOrDataFrame] = None,
) -> Optional[Figure]:
    gplot.time = time
    input_filename = os.path.join(
        input_dir,
        f"preservation_likelihood_grid_{time:0.0f}Ma.nc",
    )
    likelihood = Raster(input_filename)

    fig, ax, cax = _prepare_map(gplot=gplot, projection=projection, time=time)

    im = likelihood.imshow(
        ax=ax,
        **LIKELIHOOD_KW,
    )

    if deposits is not None:
        deposits = load_data(deposits)
        if (
            f"lon_{time:0.0f}" not in deposits.columns
            or f"lat_{time:0.0f}" not in deposits.columns
        ):
            deposits = reconstruct_by_topologies(
                data=deposits,
                plate_reconstruction=gplot.plate_reconstruction,
                times=np.arange(time, deposits["age (Ma)"].round().max() + 1),
            )
        _add_deposits(
            ax=ax,
            deposits=deposits,
            time=time,
            **SCATTER_KW,
        )

    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )
    locator = LogLocator(subs=(1, 2, 3, 4, 5))
    formatter = LogFormatterSciNotation(
        labelOnlyBase=False,
        minor_thresholds=(2, 0.3),
    )
    cbar.ax.xaxis.set_major_locator(locator)
    cbar.ax.xaxis.set_major_formatter(formatter)

    cbar.ax.tick_params(labelsize=TICKSIZE)
    cbar.ax.set_xlabel("Preservation likelihood", fontsize=FONTSIZE)

    if output_filename is not None:
        fig.savefig(output_filename, **SAVEFIG_KW)
        plt.close(fig)
        fig = None
    return fig


def plot_combined_maps(
    times: Sequence[float],
    prospectivity_dir: _PathLike,
    preservation_dir: _PathLike,
    output_dir: str,
    gplot: Optional[PlotTopologies] = None,
    topology_filenames: Optional[Union[str, Sequence[str]]] = None,
    rotation_filenames: Optional[Union[str, Sequence[str]]] = None,
    coastline_filenames: Optional[Union[str, Sequence[str]]] = None,
    projection: ccrs.Projection = ccrs.Mollweide(),
    n_jobs: int = 1,
    verbose: int = 10,
    output_template: str = r"image_{:0.0f}Ma.png",
    deposits: Optional[_PathOrDataFrame] = None,
    method: Literal["probability", "likelihood"] = "probability",
    transformer: Optional[BaseEstimator] = None,
):
    if gplot is None:
        if (
            topology_filenames is None
            or rotation_filenames is None
            or coastline_filenames is None
        ):
            raise TypeError(
                "Either `gplot` or all of "
                "`topology_filenames`, `rotation_filenames`, "
                "and `coastline_filenames` must be provided."
            )
        reconstruction = PlateReconstruction(
            rotation_filenames,
            topology_filenames,
        )
        gplot = PlotTopologies(reconstruction, coastlines=coastline_filenames)

    if deposits is not None:
        deposits = load_data(deposits, verbose=verbose > 0)
        deposits = deposits[deposits["label"] == "positive"]
        deposits = reconstruct_by_topologies(
            data=deposits,
            plate_reconstruction=gplot.plate_reconstruction,
            times=times,
            verbose=verbose > 0,
        )

    if n_jobs == 1:
        for time in times:
            output_filename = os.path.join(
                output_dir,
                output_template.format(time),
            )
            plot_combined(
                time=time,
                gplot=gplot,
                prospectivity_dir=prospectivity_dir,
                preservation_dir=preservation_dir,
                projection=projection,
                output_filename=output_filename,
                deposits=(
                    None if deposits is None
                    else deposits[[
                        f"lon_{time:0.0f}",
                        f"lat_{time:0.0f}",
                        "lon",
                        "lat",
                        "label",
                        "age (Ma)",
                    ]]
                ),
                method=method,
                transformer=transformer,
            )
    else:
        if n_jobs == 0:
            raise ValueError("`n_jobs` must not be zero")
        if n_jobs < 0:
            n_jobs = cpu_count() + n_jobs + 1
        times_split = np.array_split(times, n_jobs)
        with Parallel(n_jobs, verbose=verbose) as parallel:
            parallel(
                delayed(_plot_combined_subset)(
                    times=t,
                    gplot=gplot,
                    topology_filenames=topology_filenames,
                    rotation_filenames=rotation_filenames,
                    coastline_filenames=coastline_filenames,
                    prospectivity_dir=prospectivity_dir,
                    preservation_dir=preservation_dir,
                    output_dir=output_dir,
                    projection=projection,
                    output_template=output_template,
                    deposits=(
                        None if deposits is None
                        else deposits[[
                            *[f"lon_{i:0.0f}" for i in t],
                            *[f"lat_{i:0.0f}" for i in t],
                            "lon",
                            "lat",
                            "label",
                            "age (Ma)",
                        ]]
                    ),
                    method=method,
                    transformer=transformer,
                )
                for t in times_split
            )


def _plot_combined_subset(
    times,
    gplot,
    topology_filenames,
    rotation_filenames,
    coastline_filenames,
    prospectivity_dir,
    preservation_dir,
    output_dir,
    projection=ccrs.Mollweide(),
    output_template=r"image_{:0.0f}Ma.png",
    deposits: Optional[_PathOrDataFrame] = None,
    method="probability",
    transformer=None,
):
    if gplot is None:
        reconstruction = PlateReconstruction(
            rotation_filenames,
            topology_filenames,
            static_polygons=coastline_filenames,
        )
        gplot = PlotTopologies(
            reconstruction,
            coastlines=coastline_filenames,
        )

    if deposits is not None and (not isinstance(deposits, pd.DataFrame)):
        deposits = pd.read_csv(deposits)
    for time in times:
        output_filename = os.path.join(
            output_dir,
            output_template.format(time),
        )
        plot_combined(
            time=time,
            gplot=gplot,
            prospectivity_dir=prospectivity_dir,
            preservation_dir=preservation_dir,
            projection=projection,
            output_filename=output_filename,
            deposits=deposits,
            method=method,
            transformer=transformer,
        )


def plot_combined(
    time,
    gplot,
    prospectivity_dir,
    preservation_dir,
    method="probability",
    projection=ccrs.Mollweide(),
    output_filename=None,
    deposits=None,
    transformer=None,
):
    valid_methods = {"probability", "likelihood"}
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method ({method})"
            + f"; must be one of {valid_methods}"
        )

    gplot.time = time
    prospectivity_filename = os.path.join(
        prospectivity_dir,
        f"probability_grid_{time:0.0f}Ma.nc",
    )

    if method == "probability":
        basename = f"preservation_probability_grid_{time:0.0f}Ma.nc"
    else:  # method == "likelihood"
        basename = f"preservation_likelihood_grid_{time:0.0f}Ma.nc"
    preservation_filename = os.path.join(
        preservation_dir,
        basename
    )
    prospectivity = Raster(prospectivity_filename)
    preservation = Raster(preservation_filename)
    if transformer is not None:
        x = np.array(preservation.data).ravel()
        valid_mask = ~np.logical_or(x == 0.0, np.isnan(x))
        if np.sum(valid_mask) == 0:
            data_tmp = np.full(np.size(preservation.data), np.nan)
        else:
            x_valid = x[valid_mask].reshape((-1, 1))
            try:
                x_valid = transformer.transform(x_valid)
            except NotFittedError:
                x_valid = transformer.fit_transform(x_valid)
            data_tmp = np.full(np.size(preservation.data), np.nan)
            data_tmp[valid_mask] = np.ravel(x_valid)
        preservation.data = data_tmp.reshape(np.shape(preservation.data))

    combined = Raster(prospectivity * preservation)
    if method == "probability":
        combined *= 1.0e2  # convert to %

    fig, ax, cax = _prepare_map(gplot=gplot, projection=projection, time=time)

    im = combined.imshow(
        ax=ax,
        **(COMBINED_KW[method]),
    )

    if deposits is not None:
        deposits = load_data(deposits)
        if (
            f"lon_{time:0.0f}" not in deposits.columns
            or f"lat_{time:0.0f}" not in deposits.columns
        ):
            deposits = reconstruct_by_topologies(
                data=deposits,
                plate_reconstruction=gplot.plate_reconstruction,
                times=np.arange(time, deposits["age (Ma)"].round().max() + 1),
            )
        _add_deposits(
            ax=ax,
            deposits=deposits,
            time=time,
            **SCATTER_KW,
        )

    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )

    cbar.ax.tick_params(labelsize=TICKSIZE)
    xlabel = (
        "Deposit formation and preservation probability (%)"
        if method == "probability"
        else r"Deposit formation probability $\times$ log preservation likelihood"
    )
    cbar.ax.set_xlabel(
        xlabel,
        fontsize=FONTSIZE,
    )

    if output_filename is not None:
        fig.savefig(output_filename, **SAVEFIG_KW)
        plt.close(fig)
        fig = None
    return fig


def _prepare_map(
    gplot: PlotTopologies,
    projection: ccrs.Projection = ccrs.Mollweide(),
    time=None,
    figsize=FIGSIZE,
    colorbar_orientation="horizontal",
) -> Tuple[Figure, Axes, Axes]:
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": projection},
    )

    gplot.plot_coastlines(
        ax=ax,
        **COASTLINES_KW,
    )
    gplot.plot_all_topological_sections(
        ax=ax,
        **TOPOLOGIES_KW,
    )
    gplot.plot_ridges_and_transforms(
        ax=ax,
        **RIDGES_KW,
    )
    # gplot.plot_subduction_teeth(
    #     ax=ax,
    #     **TEETH_KW,
    # )

    ax.set_global()

    SubductionTeeth(
        ax=ax,
        left=shapelify_feature_lines(
            gplot.trench_left,
            tessellate_degrees=0.1,
            central_meridian=_meridian_from_ax(ax),
        ),
        right=shapelify_feature_lines(
            gplot.trench_right,
            tessellate_degrees=0.1,
            central_meridian=_meridian_from_ax(ax),
        ),
        **TEETH_KW,
    )

    if time is not None:
        ax.set_title(f"{time:0.0f} Ma", fontsize=TITLESIZE)
    ax_bbox = ax.get_position()
    if colorbar_orientation == "horizontal":
        cax = fig.add_axes(
            [
                ax_bbox.x0,
                ax_bbox.y0 * 0.5,
                ax_bbox.width,
                (ax_bbox.height) * 0.1,
            ]
        )
    elif colorbar_orientation == "vertical":
        cax = fig.add_axes(
            [
                ax_bbox.x0 * 1.2,
                ax_bbox.y0,
                ax_bbox.width * 0.1,
                ax_bbox.height,
            ]
        )
    else:
        raise ValueError(f"Invalid colorbar_orientation: {colorbar_orientation}")

    return fig, ax, cax


def _ticks_from_norm(norm: LogNorm, as_str=False):
    vmin = norm.vmin
    vmax = norm.vmax
    ticks = []
    for i in np.arange(np.log10(vmin), np.log10(vmax)):
        ticks.extend(np.arange(1, 6) * 10 ** i)
    ticks.append(vmax)
    if as_str:
        if isinstance(as_str, str):
            ticks = [as_str.format(i) for i in ticks]
        else:
            ticks = [f"{i:0.0f}" for i in ticks]
    return ticks

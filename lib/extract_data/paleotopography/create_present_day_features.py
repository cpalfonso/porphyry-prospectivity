import os
import sys

import numpy as np
import pygplates
import xarray as xr
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours

DIRNAME = os.path.abspath(os.path.dirname(__file__))
from . import paleogeography as pg

DEFAULT_TOPOGRAPHY_FILENAME = os.path.join(DIRNAME, "..", "topo15_3600x1800.nc")
DEFAULT_CLASSES_FILENAME = os.path.join(DIRNAME, "..", "present_day_topo_as_classes.nc")
DEFAULT_FEATURES_FILENAME = os.path.join(DIRNAME, "..", "present_day_paleogeography.gmt")


def create_present_day_features(
    topography_filename=DEFAULT_TOPOGRAPHY_FILENAME,
    classes_filename=DEFAULT_CLASSES_FILENAME,
    features_filename=DEFAULT_FEATURES_FILENAME,
):
    # load global topography
    gridX,gridY,gridZ = pg.load_netcdf(topography_filename)
    #gridX,gridY,gridZ = pg.load_netcdf('/Users/Simon/Data/SedThickness/GEBCO1m/test.nc')
    #gridX,gridY,gridZ = pg.load_netcdf('/Users/Simon/Data/GMTdata/hawaii2017/earth_relief_10m.grd')

    # gdal complains if latitudes are outside 90 degrees, so remove top and bottom rows
    gridZ = gridZ[1:-1,:]
    gridY = gridY[1:-1]

    meshX, meshY = np.meshgrid(gridX, gridY)

    # make a grid of zeros, then replace values by incrementally increasinf integers for each higher level of topography
    tmp = np.zeros(gridZ.shape)
    tmp[gridZ>-500] = 1
    tmp[gridZ>0] = 2
    tmp[gridZ>1000] = 3

    # write the integer category grid to a file
    ds = xr.DataArray(tmp,
                    coords=[('lat',gridY),('lon',gridX)])
    # ds = xr.DataArray(tmp,
    #                   coords=[('y',gridY),('x',gridX)])
    ds.to_netcdf(classes_filename, format='NETCDF3_CLASSIC')

    # use skimage.measure.find_contours
    levels = [0.5, 1.5, 2.5]
    layers = ["sm", "lm", "m"]
    features = []
    for level, layer in zip(levels, layers):
        contours = find_contours(
            tmp,
            level,
        )
        for contour in contours:
            # coords = np.array(contour)
            xcoords = map_coordinates(meshX, contour.T, order=1, mode="grid-wrap", prefilter=False).reshape((-1, 1))
            ycoords = map_coordinates(meshY, contour.T, order=1, mode="grid-wrap", prefilter=False).reshape((-1, 1))
            coords = np.hstack((xcoords, ycoords))
            feature = pygplates.Feature()
            geom = pygplates.PolygonOnSphere(np.fliplr(coords))
            feature.set_geometry(geom)
            feature.set_shapefile_attribute("Layer", layer)
            features.append(feature)
    pg_features = pygplates.FeatureCollection(features)
    pg_features.write(features_filename)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        create_present_day_features(*sys.argv[1:])
    else:
        create_present_day_features()

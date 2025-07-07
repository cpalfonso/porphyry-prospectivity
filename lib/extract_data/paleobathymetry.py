import os
import warnings

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygplates
import rasterio
import rasterio.features
import rioxarray
import shapely
import xarray as xr
from datetime import datetime
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from gplately.ptt.utils import call_system_command


def shift_antimeridian(x, y, z=None):
    """ Modify longitudes if they go above 180.
    Note that this ignores longitudes that are below -180."""
    if x <= 180.0:
        new_x = x
    else:
        new_x = x - 360
    return tuple(filter(None, [new_x, y, z]))


def shift_antimeridian_post_crosscheck(x, y, z=None):
    """ Shift longitudes (+180 to -180) after checking if we cross the world
    This is because polygons in the eastern hemisphere need 180°, while
    polygons in the western hemisphere should use -180° """
    if x >= 180:
        new_x = x - 360
    else:
        new_x = x
    return tuple(filter(None, [new_x, y, z]))


def shift_neg_antimeridian(x, y, z=None):
    """ Modify longitudes if they go below -180.
    Note that this ignores longitudes that are above +180"""
    if x >= -180.0:
        new_x = x
    else:
        new_x = x + 360
    return tuple(filter(None, [new_x, y, z]))


def shift_neg_antimeridian_post_crosscheck(x, y, z=None):
    """ Shift longitudes (-180 to +180) after checking if we cross the world
    This is because polygons in the eastern hemisphere need 180°, while
    polygons in the western hemisphere should use -180° """
    if x <= -180:
        new_x = x + 360
    else:
        new_x = x
    return tuple(filter(None, [new_x, y, z]))


def check_crossing(lon1: float, lon2: float, validate: bool = False, dlon_threshold: float = 180.0):
    """
    Inspired by: https://towardsdatascience.com/around-the-world-in-80-lines-crossing-the-antimeridian-with-python-and-shapely-c87c9b6e1513

    Assuming a minimum travel distance between two provided longitude coordinates,
    checks if the 180th meridian (antimeridian) is crossed.
    """

    if validate and any([abs(x) > 180.0 for x in [lon1, lon2]]):
        raise ValueError("longitudes must be in degrees [-180.0, 180.0]")

    out = abs(lon2 - lon1) > dlon_threshold

    # return 100 if longitudes cross antimeridian. Otherwise return 0.

    if out is True:
        return 100
    else:
        return 0


def modify_longitudes_gdf(gdf_LIPs):
    """ Check if geometries cross the antimeridian, and modify if they do.

    This is because shapely doesn't really know that -180 and 180 is the same longitude,
    and so will allow longitudinal values outside of -180 to 180.
    If we want the polygon to be included in the final raster, we need all our geometeries to be 
    within -180° and 180° longitude. 
    This function will first split all polygons at 180°, check if it actually split anything
    (will return the same polygon if not), and then shift longitudes to be within -180 to 180 if 
    a polygon was split. It then repeats the process at -180°.

    Returns a new geodataframe, with minimal columns (age, height, name). More can be added if required.

    """
    geometries = []
    names = []
    height = []
    ages = []
    plateids = []

    # loop through each row
    for i in range(len(gdf_LIPs)):

        geom = gdf_LIPs.geometry.loc[i]  # get LIP geometry

        # Split geometry if it crosses 180°
        antimeridian = shapely.geometry.LineString([(180, -90), (180, 90)])
        split_geometry = shapely.ops.split(geom, antimeridian)

        # if the polygon has been split, modify longitude
        if np.shape(split_geometry.geoms)[0] != 1:

            for row in split_geometry.geoms:

                # fix longitudes so they are in -180 to 180 format
                row_transformed = shapely.ops.transform(shift_antimeridian, row)

                # need to additionally change +180 to -180 for polygons in western hemisphere
                # First check if adjacent longitudes in polygons span more than 180 at all.
                # If they do, assume that 180 should be -180.
                crossing = []
                for coord_index, (lon, lat) in enumerate(row_transformed.exterior.coords[:]):
                    if coord_index > 0:
                        lon_prev = row_transformed.exterior.coords[:][coord_index - 1][0]

                        # check distance between both
                        cross_res = check_crossing(lon, lon_prev, validate=False)
                        crossing.append(cross_res)
                # if 100 (True) is in the crossing list, means that polygon is in western hemisphere
                if max(crossing) == 100:
                    # fix longitudes again JUST for geometries that require it
                    row_transformed = shapely.ops.transform(shift_antimeridian_post_crosscheck, row_transformed)

                # append results to list
                geometries.append(row_transformed)
                height.append(gdf_LIPs['HEIGHT'].loc[i])
                names.append(gdf_LIPs['NAME'].loc[i])
                ages.append(gdf_LIPs['FROMAGE'].loc[i])
                plateids.append(gdf_LIPs['PLATEID1'].loc[i])

        else:
            # if the geometry hasn't been split yet, check if it crosses -180 instead
            antimeridian_neg = shapely.geometry.LineString([(-180, -90), (-180, 90)])
            split_geometry_neg = shapely.ops.split(split_geometry.geoms[0], antimeridian_neg)

            # if the polygon has been split, modify longitude
            if np.shape(split_geometry_neg.geoms)[0] != 1:
                # print(len(np.shape(aa)))
                for row in split_geometry_neg.geoms:

                    # fix longitudes so they are in -180 to 180 format
                    row_transformed = shapely.ops.transform(shift_neg_antimeridian, row)

                    # need to additionally change -180 to 180 for polygons in eastern hemisphere
                    # check if longitudes in polygons span more than 180 at all.
                    # If they do, assume that -180 should be +180.
                    crossing = []
                    for coord_index, (lon, lat) in enumerate(row_transformed.exterior.coords[:]):
                        if coord_index > 0:
                            # get previous coordinate
                            lon_prev = row_transformed.exterior.coords[:][coord_index - 1][0]
                            # check distance between both. If 100 (True), means that coordinates need to be fixed
                            cross_res = check_crossing(lon, lon_prev, validate=False)
                            crossing.append(cross_res)
                    # if 100 is in the crossing list, means that polygon is in western hemisphere
                    if max(crossing) == 100:
                        # fix longitudes again if required
                        row_transformed = shapely.ops.transform(shift_neg_antimeridian_post_crosscheck, row_transformed)

                    # append to list
                    geometries.append(row_transformed)
                    height.append(gdf_LIPs['HEIGHT'].loc[i])
                    names.append(gdf_LIPs['NAME'].loc[i])
                    ages.append(gdf_LIPs['FROMAGE'].loc[i])
                    plateids.append(gdf_LIPs['PLATEID1'].loc[i])

            else:
                # geometry wasn't split, append to list
                geometries.append(split_geometry_neg.geoms[0])
                height.append(gdf_LIPs['HEIGHT'].loc[i])
                names.append(gdf_LIPs['NAME'].loc[i])
                ages.append(gdf_LIPs['FROMAGE'].loc[i])
                plateids.append(gdf_LIPs['PLATEID1'].loc[i])

    # make geodataframe
    gdf_LIPs_out = gpd.GeoDataFrame()
    gdf_LIPs_out['NAME'] = None
    gdf_LIPs_out['geometry'] = None
    gdf_LIPs_out = gdf_LIPs_out.set_crs(epsg=4326)

    # add things to it
    gdf_LIPs_out['NAME'] = names
    gdf_LIPs_out['HEIGHT'] = height
    gdf_LIPs_out['geometry'] = geometries
    gdf_LIPs_out['FROMAGE'] = ages
    gdf_LIPs_out['PLATEID1'] = plateids

    # DOES THIS SORTING MAKE SENSE? I don't know...
    gdf_LIPs_out.sort_values(by=['HEIGHT', 'NAME'], inplace=True, ignore_index=True, ascending=[True, True])
    

    return gdf_LIPs_out


def create_mask_from_gdf(
    gdf,
    grid_spacing,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    mask_value=1,
    fill_value=0,
):
    
    # # set region
    # output_shape = (1801, 3601)
    # output_transform = rasterio.Affine(0.1, 0.0, -180.05, 0.0, 0.1, -90.05)  # reference point for lats/lons
    # lats = np.arange(-90, 90.1, 0.1)
    # lons = np.arange(-180., 180.1, 0.1)

    lat_shape = int((180 / grid_spacing) + 1)
    lon_shape = int((360 / grid_spacing) + 1)


    output_shape = (lat_shape, lon_shape)
    # output_shape = (1801, 3601)
    output_transform = rasterio.Affine(
        grid_spacing, 0.0, lon_min - (grid_spacing/2), 0.0, grid_spacing, lat_min -(grid_spacing/2))
    # NOTE ref point is bottom left corner)
    lats = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
    lons = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)
    
    
    mask = rasterio.features.rasterize([(x.geometry, mask_value) 
                                        for i, x in gdf.iterrows()],
                                       out_shape=output_shape,
                                       transform=output_transform,
                                       fill=fill_value,
                                       all_touched=True)

    da_mask = xr.DataArray(mask, coords=[lats, lons], dims=('lat', 'lon'), name='z')

    # convert to dataset and add some attributes
    ds_mask = da_mask.to_dataset()
    ds_mask['z'].attrs = {
        'actual_range': np.array([np.nanmin(ds_mask.z), np.nanmax(ds_mask.z)], dtype=np.float32)}
    ds_mask['lat'].attrs = {
        'long_name': "latitude", 'standard_name': "latitude", 'units': "degrees_north",
        'actual_range': np.array([np.nanmin(ds_mask.lat), np.nanmax(ds_mask.lat)], dtype=np.float32)}
    ds_mask['lon'].attrs = {
        'long_name': "longitude", 'standard_name': "longitude", 'units': "degrees_east", 
        'actual_range': np.array([np.nanmin(ds_mask.lon), np.nanmax(ds_mask.lon)], dtype=np.float32)}


    # global attributes
    ds_mask.attrs['history'] = "created %s" % (datetime.now().strftime('%Y-%m-%d %H:%M'))

    # round lats and lons, to ensure we can safely add them
    # because xarray is annoying sometimes
    ds_mask['lat'] = np.round(ds_mask.lat.values, 1)
    ds_mask['lon'] = np.round(ds_mask.lon.values, 1)
    
    # saving logic. Not used here
    
    # path_output_grids = 'LIP_masks'
    # if not os.path.exists('%s' % (path_output_grids)):
    #     print("... Creating " + str(path_output_grids) + " now")
    #     os.makedirs('%s' % path_output_grids)

    # # compress so it doesn't take a ridiculous amount of space!
    # comp = dict(zlib=True, complevel=7)
    # encoding = {var: comp for var in ds_lip_raster.data_vars}

    # ds_lip_mask.to_netcdf('%s/LIP_mask_%sMa.nc' %
    #                       (path_output_grids, age), encoding=encoding)
    
    return ds_mask


def create_geodataframe_from_pygplates_reconstructed_feature(pygplates_recon_geom, reconstruction_time):
    """ This is a general function to convert reconstructed LIP features
    from pygplates into a GeoDataFrame. This helps avoid plotting artefacts.
    Note that the input geometry must be a polygon.
    
    Input: 
        - pygplates.ReconstructedFeatureGeometry (i.e., output of pygplates.reconstruct)
        - recontruction time - this is just for safekeeping in the geodataframe!
    Output: 
        - gpd.GeoDataFrame of the feature"""

    # create new and empty geodataframe
    recon_gdf = gpd.GeoDataFrame()
    recon_gdf['NAME'] = None
    recon_gdf['PLATEID1'] = None
    recon_gdf['FROMAGE'] = None
    recon_gdf['TOAGE'] = None
    recon_gdf['geometry'] = None
    recon_gdf['reconstruction_time'] = None
    recon_gdf = recon_gdf.set_crs(epsg=4326)

    date_line_wrapper = pygplates.DateLineWrapper()

    names                = []
    plateid1s            = []
    fromages             = []
    toages               = []
    geometrys            = []
    reconstruction_times = []
    con_elevs = []

    for i, seg in enumerate(pygplates_recon_geom):
        if isinstance(seg, pygplates.ReconstructedFeatureGeometry):
            wrapped_polygons = date_line_wrapper.wrap(seg.get_reconstructed_geometry())
            for poly in wrapped_polygons:
                ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
                ring[:,1] = np.clip(ring[:, 1], -89.9, 89.9) # anything approaching the poles creates artefacts

                name = seg.get_feature().get_name()
                plateid = seg.get_feature().get_reconstruction_plate_id()
                from_age, to_age = seg.get_feature().get_valid_time()
                
                con_elev = seg.get_feature().get_shapefile_attribute('CON_ELEV')
                
                # append things
                names.append(name)
                plateid1s.append(plateid)
                fromages.append(from_age)
                toages.append(to_age)
                geometrys.append(shapely.geometry.Polygon(ring))
                reconstruction_times.append(reconstruction_time)
                con_elevs.append(con_elev)
        elif isinstance(seg, pygplates.Feature):
            wrapped_polygons = date_line_wrapper.wrap(seg.get_reconstructed_geometry())
            for poly in wrapped_polygons:
                ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
                ring[:,1] = np.clip(ring[:, 1], -89.9, 89.9) # anything approaching the poles creates artefacts

                name = seg.get_feature().get_name()
                plateid = seg.get_feature().get_reconstruction_plate_id()
                from_age, to_age = seg.get_feature().get_valid_time()
                
                con_elev = seg.get_feature().get_shapefile_attribute('CON_ELEV')
                
                # append things
                names.append(name)
                plateid1s.append(plateid)
                fromages.append(from_age)
                toages.append(to_age)
                geometrys.append(shapely.geometry.Polygon(ring))
                reconstruction_times.append(reconstruction_time)
                con_elevs.append(con_elev)

    # write to geodataframe
    recon_gdf['NAME'] = names
    recon_gdf['PLATEID1'] = plateid1s
    recon_gdf['FROMAGE'] = fromages
    recon_gdf['TOAGE'] = toages
    recon_gdf['reconstruction_time'] = reconstruction_times
    recon_gdf['CON_ELEV'] = con_elevs
    recon_gdf['geometry'] = geometrys

    return recon_gdf


def rotate_and_grid_contoured_shapefile(
    time,
    input_file,
    rotation_filenames,
    path_output_grids,
    grid_spacing,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
):
    # import rotation file here, otherwise Python will complain during multiprocessing.
    rotation_model = pygplates.RotationModel(rotation_filenames)
    print('...... Working on %s Ma' % time)

    reconstructed_feature = []
    pygplates.reconstruct(input_file, rotation_model, reconstructed_feature, float(time), export_wrap_to_dateline=True)

    gdf_reconstructed_contours = create_geodataframe_from_pygplates_reconstructed_feature(reconstructed_feature, time)
    
    """
    Convert LIP shapefile (as geodataframe) into a netcdf using rasterio/rasterize
    Defaults to 0.1° and -180/180/-90/90.
    Then pass the raster to xarray, so we can save the netcdf out nicely.

    """

    if gdf_reconstructed_contours.empty is True:
        print('... No features at %s Ma' % time)
    else:
        lat_shape = int((180 / grid_spacing) + 1)
        lon_shape = int((360 / grid_spacing) + 1)


        output_shape = (lat_shape, lon_shape)
        # output_shape = (1801, 3601)
        output_transform = rasterio.Affine(
            grid_spacing, 0.0, lon_min - (grid_spacing/2), 0.0, grid_spacing, lat_min -(grid_spacing/2))
        # NOTE ref point is bottom left corner)
        lats = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
        lons = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)

        z_column = 'CON_ELEV'

        # convert geometries to raster based on z column
        out_raster = rasterio.features.rasterize([(x.geometry, x[z_column]) for i, x in gdf_reconstructed_contours.iterrows()],
                                                 out_shape=output_shape, transform=output_transform,
                                                 fill=np.nan, all_touched=True)

        # convert to xarray, since that makes saving it out nicely easier
        da_raster = xr.DataArray(out_raster, coords=[lats, lons], dims=('lat', 'lon'), name='z')

        # convert to dataset and add some attributes
        ds_raster = da_raster.to_dataset()

        # round lats and lons, to ensure we can safely add them
        # because xarray is annoying sometimes
        ds_raster['lat'] = np.round(ds_raster.lat.values, 1)
        ds_raster['lon'] = np.round(ds_raster.lon.values, 1)

        ds_raster['z'].attrs = {
            'long_name': 'sediment thickness', 'units': 'm',
            'actual_range': np.array([np.nanmin(ds_raster.z), np.nanmax(ds_raster.z)], dtype=np.float32)}
        ds_raster['lat'].attrs = {
            'long_name': "latitude", 'standard_name': "latitude", 'units': "degrees_north",
            'actual_range': np.array([np.nanmin(ds_raster.lat), np.nanmax(ds_raster.lat)], dtype=np.float32)}
        ds_raster['lon'].attrs = {
            'long_name': "longitude", 'standard_name': "longitude", 'units': "degrees_east", 
            'actual_range': np.array([np.nanmin(ds_raster.lon), np.nanmax(ds_raster.lon)], dtype=np.float32)}
        

        # global attributes
        ds_raster.attrs['title'] = "Sediment thickness from LIP emplacement for %s Ma" % (time)
        ds_raster.attrs['history'] = "created %s" % (datetime.now().strftime('%Y-%m-%d %H:%M'))

        # compress so it doesn't take a ridiculous amount of space!
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds_raster.data_vars}

        ds_raster.to_netcdf('%s/reconstructedSeds_%sMa.nc' %
                                (path_output_grids, time), encoding=encoding)
        print('... saved netcdf for %s Ma' % time)


def rotate_LIP_shapefile(LIPs_0Ma, rotation_model, path_rotated_polygons, time):
    print('...... Working on %s Ma, creating %s/reconLIPs_%sMa.shp' %
          (time, path_rotated_polygons, time))

    pygplates.reconstruct(LIPs_0Ma, rotation_model, '%s/reconLIPs_%sMa.shp' % (path_rotated_polygons, time),
                          float(time), export_wrap_to_dateline=True)


def convert_shapefile_to_raster(
    gdf_LIPs,
    z_column,
    path_output_grids,
    time,
    grid_spacing,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
):
    """
    Convert LIP shapefile (as geodataframe) into a netcdf using rasterio/rasterize
    Defaults to 0.1° and -180/180/-90/90.
    Then pass the raster to xarray, so we can save the netcdf out nicely.

    """

    # see if basement_depth has previously been imported (probably not), and
    # make the output grid the same resolution/region.
    # Otherwise, default to 0.1° -180/180/-90/90

        # print('...... Defaulting to 0.1° global grid')
        # output_shape = (1801, 3601)
        # output_transform = rasterio.Affine(
        #     0.1, 0.0, -180.05, 0.0, 0.1, -90.05)
        # # NOTE that lats are in descending order (ref point is top left corner)
        # # NOTE 2 : line above isn't true anymore
        # lats = np.arange(-90, 90.1, 0.1)
        # lons = np.arange(-180., 180.1, 0.1)

    lat_shape = int((180 / grid_spacing) + 1)
    lon_shape = int((360 / grid_spacing) + 1)


    output_shape = (lat_shape, lon_shape)
    # output_shape = (1801, 3601)
    output_transform = rasterio.Affine(
        grid_spacing, 0.0, lon_min - (grid_spacing/2), 0.0, grid_spacing, lat_min -(grid_spacing/2))
    # NOTE ref point is bottom left corner)
    lats = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
    lons = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)


    print('... creating LIP raster for %s Ma' % time)
    # convert geometries to raster based on HEIGHT column
    lip_raster = rasterio.features.rasterize([(x.geometry, x[z_column]) for i, x in gdf_LIPs.iterrows()],
                                             out_shape=output_shape,
                                             transform=output_transform,
                                             fill=np.nan,
                                             all_touched=True)

    # convert to xarray, since that makes saving it out nicely easier
    da_lip_raster = xr.DataArray(
        lip_raster, coords=[lats, lons], dims=('lat', 'lon'), name='z')

    # convert to dataset and add some attributes
    ds_lip_raster = da_lip_raster.to_dataset()
    ds_lip_raster['z'].attrs = {
        'long_name': 'sediment thickness', 'units': 'm',
        'actual_range': np.array([np.nanmin(ds_lip_raster.z), np.nanmax(ds_lip_raster.z)], dtype=np.float32)}
    ds_lip_raster['lat'].attrs = {
        'long_name': "latitude", 'standard_name': "latitude", 'units': "degrees_north",
        'actual_range': np.array([np.nanmin(ds_lip_raster.lat), np.nanmax(ds_lip_raster.lat)], dtype=np.float32)}
    ds_lip_raster['lon'].attrs = {
        'long_name': "longitude", 'standard_name': "longitude", 'units': "degrees_east", 
        'actual_range': np.array([np.nanmin(ds_lip_raster.lon), np.nanmax(ds_lip_raster.lon)], dtype=np.float32)}

    # round lats and lons, to ensure we can safely add them
    # because xarray is annoying sometimes
    ds_lip_raster['lat'] = np.round(ds_lip_raster.lat.values, 1)
    ds_lip_raster['lon'] = np.round(ds_lip_raster.lon.values, 1)

    # global attributes
    ds_lip_raster.attrs['title'] = "Sediment thickness from LIP emplacement for %s Ma" % (
        time)
    ds_lip_raster.attrs['history'] = "created %s" % (
        datetime.now().strftime('%Y-%m-%d %H:%M'))

    # compress so it doesn't take a ridiculous amount of space!
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_lip_raster.data_vars}

    ds_lip_raster.to_netcdf('%s/reconstructedSeds_%sMa.nc' %
                            (path_output_grids, time), encoding=encoding)
    print('... saved netcdf for %s Ma' % time)


def mp_wrapper_for_shapefile_to_raster(
    time,
    LIP_contour_dir_merged,
    path_output_grids,
):
    sed_filename = '%s/reconLIPs_%sMa.shp' % (LIP_contour_dir_merged, time)
    # print(sed_filename)
    if os.path.isfile(sed_filename):
        gdf_sed = gpd.read_file(sed_filename)

        convert_shapefile_to_raster(gdf_sed, 'CON_ELEV', path_output_grids, time)
    else:
        print('... no LIP shapefile for %s Ma' % time)
        pass


def get_LIPmask_and_newly_emplaced_sed_contours_at_time(
    LIP_path,
    sed_path,
    rotation_filenames,
    time,
    LIP_mask_dir,
    sed_thickness_contour_interval,
    LIP_contour_dir,
):
    print('...... working on %s Ma' % time)
    rotation_model = pygplates.RotationModel(rotation_filenames)
    # import LIPs
    # check if the file exists first
    if os.path.isfile(LIP_path):
        # print('... LIP_path is: %s' % LIP_path)
        gdf_LIPs = gpd.read_file(LIP_path)
        
        # get LIP outline of ALL LIPs that are active at this time
        gdf_LIPs_outline = gdf_LIPs[gdf_LIPs['NAME'].str.contains('outline')]
        gdf_LIPs_outline = gdf_LIPs_outline.reset_index(drop=True)
        gdf_LIPs_outline = gdf_LIPs_outline.drop(('index'), axis=1)
        gdf_LIPs_outline_fixed = modify_longitudes_gdf(gdf_LIPs_outline)
        
        # create LIP mask
        ds_lip_mask = create_mask_from_gdf(gdf_LIPs_outline_fixed, mask_value=np.nan, fill_value=0)
        
        # compress so it doesn't take a ridiculous amount of space!
        comp = dict(zlib=True, complevel=7)
        encoding = {var: comp for var in ds_lip_mask.data_vars}
        
        ds_lip_mask.to_netcdf('%s/LIPmask_%sMa.nc' % (LIP_mask_dir, time), encoding=encoding)
        
        # ------- LIPs that were emplaced at this time only
        # Get LIPs that were emplaced at the age of interest only
        gdf_LIPs_outline_emplaced_only = gdf_LIPs_outline[np.round(gdf_LIPs_outline['FROMAGE'],0) == time]
        # # fix index and geometry
        gdf_LIPs_outline_emplaced_only = gdf_LIPs_outline_emplaced_only.reset_index(drop=True)
        gdf_LIPs_outline_emplaced_only['geometry'] = gdf_LIPs_outline_emplaced_only.buffer(0)
        gdf_LIPs_outline_emplaced_only_fixed = modify_longitudes_gdf(gdf_LIPs_outline_emplaced_only)
        
        emplaced_on_land = gpd.GeoDataFrame()
        emplaced_on_land['geometry'] = None
        
        if gdf_LIPs_outline_emplaced_only_fixed.empty:
            print('No LIPs emplaced at this time (%s Ma)' % time)
            # gdf_contoured_LIPs = gpd.GeoDataFrame()
            pass
        
        else:
            
            # --- sed grid stuff
            rio_sedthickgrd = rioxarray.open_rasterio(sed_path, masked=True)
            rio_sedthickgrd = rio_sedthickgrd.sel(band=1)
            # get rid of uneeded coordinates (makes life easier later)
            rio_sedthickgrd = rio_sedthickgrd.drop(['band', 'spatial_ref'])
            rio_sedthickgrd.rio.write_crs("epsg:4326", inplace=True)  # set crs

            # get sediment thickness of LIP regions
            gdf_LIPs_outline_emplaced_only_fixed_clipped = rio_sedthickgrd.rio.clip(gdf_LIPs_outline_emplaced_only_fixed.geometry, invert=False)
            # print(gdf_LIPs_outline_emplaced_only_fixed)
            # ds_gdf_LIPs_outline_emplaced_only_fixed_clipped = gdf_LIPs_outline_emplaced_only_fixed_clipped.to_dataset()
            # ds_gdf_LIPs_outline_emplaced_only_fixed_clipped.to_netcdf('aaa.nc')
            # -- convert to contours
            # calculate min and max values for contours
            min_level = None
            if min_level is None:
                min_value = gdf_LIPs_outline_emplaced_only_fixed_clipped.min()
                min_level = sed_thickness_contour_interval * np.floor((min_value) / sed_thickness_contour_interval)
        
            max_value = gdf_LIPs_outline_emplaced_only_fixed_clipped.max()

            # Due to range issues, a level is added
            max_level = sed_thickness_contour_interval * (1 + np.ceil(max_value / sed_thickness_contour_interval))

            try:
                cont_levels = np.arange(min_level, max_level, sed_thickness_contour_interval)
                # print(cont_levels, '%s Ma' % time)
                cs = plt.contourf(gdf_LIPs_outline_emplaced_only_fixed_clipped.x, gdf_LIPs_outline_emplaced_only_fixed_clipped.y, gdf_LIPs_outline_emplaced_only_fixed_clipped, cont_levels)
                # the above line causes issues with multiprocessing. Could try something from here instead? https://stackoverflow.com/questions/41487642/how-to-export-contours-created-in-scikit-image-find-contours-to-shapefile-or-geo

                print('... found contours for %s Ma' % time)
                # ----
                # Get contours for each LIP into a geodataframe
                # Using code from here: https://chrishavlin.com/tag/shapely/

                contour_gdf = gpd.GeoDataFrame()
                contour_gdf['CON_ELEV'] = None
                contour_gdf['geometry'] = None
                contour_gdf = contour_gdf.set_crs(epsg=4326)

                # loop over collections (and polygons in each collection), store in list
                polys = []
                elev = []

                # create lookup table so we can get the contour level out
                lvl_lookup = dict(zip(cs.collections, cs.levels))
                for col in cs.collections:
                    z = lvl_lookup[col]  # the value of this level
                    for contour_path in col.get_paths():
                        # create the polygon for this level
                        for ncp, cp in enumerate(contour_path.to_polygons()):
                            lons = cp[:, 0]
                            lats = cp[:, 1]
                            new_shape = shapely.geometry.Polygon(
                                [(i[0], i[1]) for i in zip(lons, lats)])
                            if ncp == 0:
                                poly = new_shape  # first shape
                            else:
                                poly = poly.difference(new_shape)  # Remove the holes

                            polys.append(poly)
                            elev.append(z)

                contour_gdf['geometry'] = polys
                contour_gdf['CON_ELEV'] = elev

                # shouldn't be needed, but keeping anyway - make sure contours are inside outline
                clipped_LIPs_contour = []
                for i, row in gdf_LIPs_outline_emplaced_only_fixed.iterrows():
                    single = gpd.GeoDataFrame(row.to_frame().T, crs=gdf_LIPs_outline_emplaced_only_fixed.crs)
                    clipped_LIP_contour = gpd.clip(contour_gdf, single)
                    clipped_LIPs_contour.append(clipped_LIP_contour)
                clipped_LIPs_gdf = gpd.GeoDataFrame(
                    pd.concat(clipped_LIPs_contour), crs=gdf_LIPs_outline_emplaced_only_fixed.crs)

                gdf_contoured_LIPs = gpd.overlay(gdf_LIPs_outline_emplaced_only_fixed, clipped_LIPs_gdf, how='intersection')
                gdf_contoured_LIPs = gdf_contoured_LIPs.drop('HEIGHT', axis=1)
                if gdf_contoured_LIPs.empty is True:
                    pass
                else:
                    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
                        gdf_contoured_LIPs.to_file('%s/gdf_contoured_LIPs_%sMa.shp' % (LIP_contour_dir, time))
                    pygplates.reverse_reconstruct('%s/gdf_contoured_LIPs_%sMa.shp' % (LIP_contour_dir, time), rotation_model, float(time))
                
            except ValueError:
                print('LIP %s may not have emplaced in ocean' % gdf_LIPs_outline_emplaced_only_fixed.NAME)
                # emplaced_on_land = emplaced_on_land.append(gdf_LIPs_outline_emplaced_only_fixed)

                emplaced_on_land = pd.concat([emplaced_on_land, gdf_LIPs_outline_emplaced_only_fixed], ignore_index=True)
                
                gdf_contoured_LIPs = None
                with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
                    emplaced_on_land.to_file('%s/emplaced_on_land_%s.shp' % (LIP_contour_dir, time))
    else:
        print('... No LIP shapefile for %s Ma' % time)
        pass


# def mp_wrapper_for_LIP_mask_and_sed_contours_at_time(time):
# 	LIPs_rotated_buffered = '%s/rotated_polygons_buffered/LIPs_%s.0Ma.shp' % (LIP_outdir, time)
# 	sedthickgrd = '%s/%s%s%s' % (sediment_thickness_grid_dir, sediment_thickness_grid_filename, time, sediment_thickness_grid_filename_ext)

# 	get_LIPmask_and_newly_emplaced_sed_contours_at_time(LIPs_rotated_buffered, sedthickgrd, rotation_filenames, time)


def calculate_paleobathymetry_old(
    time,
    sediment_thickness_filename,
    output_dir,
    depth2basedir,
    age_depth_model,
    output_file_age_zero_padding,
    paleobathygrid_output_ext,
    LIP_output_dir,
    LIP_feature_type,
    paleobathymetry_main_output_dir,
    sediment_thickness_model_name,
    output_paleobathydir,
    model_name,
    LIP_age_depth_model,
    part4_add_Pacific_synthetic_seamounts_to_paleobathy,
    create_final_geotiffs,
):
    print('Calculating paleobathymetry for %s Ma' % time)

    ## ----
    sedthickgrd = '%s%s.0.nc' % (sediment_thickness_filename, int(time))

    LIP_maskgrd = '%s/LIP_masks/LIPmask_%sMa.nc' % (output_dir, time)
    sedsfromLIPtimegrd = '%s/sediments_from_LIP_emplacement_grids/reconstructedSeds_%sMa.nc'  % (output_dir, time)
    depth2basegrid = '%s/depth2base_%s_%s%s' % (depth2basedir, age_depth_model, str(time).zfill(output_file_age_zero_padding), paleobathygrid_output_ext)

    LIPheightgrd = '%s/grids/reconstructed_%s_%sMa.nc' % (LIP_output_dir, LIP_feature_type, time)

    # --- new files this script will create
    sedthick_noLIPsedsgrd = '%s/sediments_maskedLIPs/sediments_noLIPS_%sMa.nc'  % (output_dir, time)
    sedthick_adjforLIPsgrd = '%s/sediments_grds_adjusted_for_LIPs/sediments_adj_for_LIPs_%sMa.nc' % (output_dir, time)

    isostatic_correction_grd = '%s/isostatic_correction/isocorr_%sMa.nc' % (output_dir, time)

    depth_to_basement_with_seds_dir = '%s/Paleobathy_%s_%s' % (paleobathymetry_main_output_dir, age_depth_model, sediment_thickness_model_name)
    # if not os.path.exists(depth_to_basement_with_seds_dir):
    os.makedirs(depth_to_basement_with_seds_dir, exist_ok=True)

    depth_to_basement_with_seds_grd = '%s/paleobathymetry_%sMa.nc' % (depth_to_basement_with_seds_dir, time)

    # ---- paleobathymetry with LIPs!
    paleobathygrd = '%s/paleobathymetry_%sMa.nc' % (output_paleobathydir, time)

    if os.path.isfile(LIP_maskgrd):
        # LIP_mask = xr.open_dataset(LIP_maskgrd)

        # mask out LIPs from sediment thickness grid, convert LIP NaNs to 0, and then remask with the original mask
        # sedthick_noLIPseds = (sedthick + LIP_mask).fillna(0) + (sedthick / sedthick)
        # change to use GMT, because xarray was doing weird things again.
        # call_system_command(('gmt', 'grdmath', '%s' % sedthickgrd, '%s' % LIP_maskgrd, 'OR', '0', 'DENAN', 'STO@lipsmasked',
        #              '%s' % sedthickgrd, '%s' % sedthickgrd, 'DIV', 'STO@origmask', 'RCL@lipsmasked', 'RCL@origmask', 'OR', '-Vn',
        #              '=', '%s' % sedthick_noLIPsedsgrd))

        LIP_mask = xr.open_dataset(LIP_maskgrd)
        sedthick = xr.open_dataset(sedthickgrd)

        LIP_mask['lon'] = np.round(LIP_mask.lon.values, 2)
        LIP_mask['lat'] = np.round(LIP_mask.lat.values, 2)

        sedthick['lon'] = np.round(sedthick.lon.values, 2)
        sedthick['lat'] = np.round(sedthick.lat.values, 2)

        sedthick_noLIPseds = (sedthick + LIP_mask).fillna(0) + (sedthick / sedthick)
        
        sedthick_noLIPseds['z'].attrs = {
            'actual_range': np.array([np.nanmin(sedthick_noLIPseds.z), np.nanmax(sedthick_noLIPseds.z)], dtype=np.float32)}
        sedthick_noLIPseds['lat'].attrs = {
            'long_name': "latitude", 'standard_name': "latitude", 'units': "degrees_north", 'actual_range': np.array([np.nanmin(sedthick_noLIPseds.lat), np.nanmax(sedthick_noLIPseds.lat)], dtype=np.float32)}
        sedthick_noLIPseds['lon'].attrs = {
            'long_name': "longitude", 'standard_name': "longitude", 'units': "degrees_east", 'actual_range': np.array([np.nanmin(sedthick_noLIPseds.lon), np.nanmax(sedthick_noLIPseds.lon)], dtype=np.float32)}

        # save with some compression to save some space
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in sedthick_noLIPseds.data_vars}

        sedthick_noLIPseds.to_netcdf('%s' % (sedthick_noLIPsedsgrd), mode='w', compute=True, encoding=encoding)


        # makes sedthick_noLIPsedsgrd
    else: 
        print('Note: no %s exist at %s Ma' % (LIP_feature_type, time))
        sedthick_noLIPsedsgrd = sedthickgrd

    if os.path.isfile(sedsfromLIPtimegrd):
        # check if there are any adjustments needed for sediment thickness at this time
        # ADD sediments from when the LIP emplaced
        # sedsfromLIPtime = xr.open_dataset(sedsfromLIPtimegrd)

        # add reconstructed sediment thickness
        # sedthick_adjforLIPs = sedthick_noLIPseds + sedsfromLIPtime.fillna(0)

        ## create new sediment thickness grid that has been adjusted for LIPs
        call_system_command(('gmt', 'grdmath', '%s' % sedsfromLIPtimegrd, '0', 'DENAN',
            '%s' % sedthick_noLIPsedsgrd, 'ADD', '-Vn', '=', '%s' % sedthick_adjforLIPsgrd))

    else:
        # print('no sediments to adjust for at %s Ma' % time)
        sedthick_adjforLIPsgrd = sedthick_noLIPsedsgrd
        # print('Using: %s' % sedthick_adjforLIPsgrd)
    # calculate isostatic correction
    # iso_corr = ((0.42422 * sedthick_adjforLIPs/1000) - (0.010395 * ((sedthick_adjforLIPs/1000)*(sedthick_adjforLIPs/1000))))*1000
    # ---- First compute isostatic correction --------------
    # echo "  Computing Sykes et al. isostatic correction for sediment thickness (correction is in m (positive))"
    # gmt grdmath ${verbose} $sedthickgrd 1000 DIV 0.43422 MUL = zzz1.grd
    call_system_command(('gmt', 'grdmath', '%s' % sedthick_adjforLIPsgrd, '1000', 'DIV', '0.43422', 'MUL', '-Vn', '=', 'zzz1_%s.grd' % time))
    # gmt grdmath ${verbose} $sedthickgrd 1000 DIV 2 POW 0.010395 MUL = zzz2.grd
    call_system_command(('gmt', 'grdmath', '%s' % sedthick_adjforLIPsgrd, '1000', 'DIV', '2', 'POW', '0.010395', 'MUL', '-Vn', '=', 'zzz2_%s.grd' % time))
    # gmt grdmath ${verbose} zzz1.grd zzz2.grd SUB 1000 MUL = zzz3.grd
    call_system_command(('gmt', 'grdmath', 'zzz1_%s.grd' % time,'zzz2_%s.grd' % time, 'SUB', '1000', 'MUL', '-Vn', '=', 'zzz3_%s.grd' % time))
    # gmt grdmath ${verbose} zzz3.grd 0 GE zzz3.grd MUL = isocorr_${age}.grd
    call_system_command(('gmt', 'grdmath', 'zzz3_%s.grd' % time, '0', 'GE', 'zzz3_%s.grd' % time, 'MUL', '-Vn', '=', '%s' % isostatic_correction_grd))

    call_system_command(('rm', 'zzz1_%s.grd' % time, 'zzz2_%s.grd' % time, 'zzz3_%s.grd' % time))
    # --- add to depth-to-basement
    # depth2base = xr.open_dataset(depth2basegrid)
    # depth2base_and_seds = depth2base + sedthick_adjforLIPs - iso_corr
    #   gmt grdmath ${verbose} $depth2basegrd $sedthickgrd ADD isocorr_${age}.grd SUB = $palaeobathgrid
    call_system_command(('gmt', 'grdmath', '%s' % depth2basegrid, '%s' % sedthick_adjforLIPsgrd, 'ADD', 
        '%s' % isostatic_correction_grd, 'SUB', '-Vn', '=', '%s' % depth_to_basement_with_seds_grd))
    call_system_command(('gmt', 'grdedit', '%s' % depth_to_basement_with_seds_grd, '-G%s' % depth_to_basement_with_seds_grd, '-fg', '-Vn',
            '-D+t"Depth-to-basement + D17 sediments, adjusted for LIP emplacement, and including isostatic correction at %s Ma. Created %s"+ddepth[m]+r"Plate model: %s; depth-to-basement: %s; sediment thickness: %s"' % (time, 
                datetime.now().strftime('%Y-%m-%d %H:%M'), model_name, age_depth_model, sediment_thickness_model_name)))

    # ----- check if LIPs exist at this time, and add if they do
    if os.path.isfile(LIPheightgrd):
        # LIPheight = xr.open_dataset(LIPheightgrd)
        # check if the variable name is HEIGHT or z.
        # Change to z otherwise, since xarray needs variables to have the same name
        
        # var_name = list(LIPheight.keys())[0]
        # if  var_name != 'z':
        #     print('need to change variable name')
        #     LIPheight['z'] = LIPheight[var_name]
        #     LIPheight = LIPheight.drop(var_name)

        call_system_command(('gmt', 'grdmath', '%s' % LIPheightgrd, '0', 'DENAN', '%s' % depth_to_basement_with_seds_grd, 'ADD', '-Vn', '=', '%s' % paleobathygrd))

        # add better info to the netcdf itself
        # call_system_command(('gmt', 'grdedit', '%s' % paleobathygrd, '-G%s' % paleobathygrd, '-fg', '-Vn',
        #     '-D+t"Paleobathymetry at %s Ma. Created %s"+ddepth[m]+r"Plate model: %s; depth-to-basement: %s; sediment thickness: %s; %s using %s. Sediment thickness has been adjusted for LIPs"' % (time, 
        #         datetime.now().strftime('%Y-%m-%d %H:%M'), model_name, age_depth_model, sediment_thickness_model_name, 
        #         LIP_feature_type, LIP_age_depth_model)))

    else:
        # no LIPs to add. Grid is the same as the previous step.
        call_system_command(('cp', '%s' % depth_to_basement_with_seds_grd, '%s' % paleobathygrd))
        call_system_command(('gmt', 'grdedit', '%s' % paleobathygrd, '-G%s' % paleobathygrd, '-fg', '-Vn',
            '-D+t"Paleobathymetry at %s Ma. Created %s"+ddepth[m]+r"Plate model: %s; depth-to-basement: %s; sediment thickness: %s; %s using %s. Sediment thickness has been adjusted for LIPs. NOTE: no LIPs were actually added at this time"' % (time, 
                datetime.now().strftime('%Y-%m-%d %H:%M'), model_name, age_depth_model, sediment_thickness_model_name, 
                LIP_feature_type, LIP_age_depth_model)))

    if part4_add_Pacific_synthetic_seamounts_to_paleobathy.lower() in ['true', '1', 't', 'y', 'yes']:
        # potentially create geotiffs later
        pass
    else:
        # not adding synthetic seamounts, create geotiffs now
        if create_final_geotiffs.lower() in ['true', '1', 't', 'y', 'yes']:
            paleobathy_grid_basename = 'paleobathymetry_'
            paleobathy_grid_filename = os.path.join(output_paleobathydir, '{0}{1}Ma.nc'.format(paleobathy_grid_basename, time))

            tiff_directory_tiff = '%s/geotiff' % paleobathymetry_main_output_dir

            tiff_filename_tiff = os.path.join(tiff_directory_tiff, '{0}{1}Ma.tiff'.format(paleobathy_grid_basename, time))

            # add some compression when making the geotiff
            call_system_command(['gdal_translate', 
                'NETCDF:"%s":z' % paleobathy_grid_filename, '%s' % tiff_filename_tiff, '-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=3', '-co', 'TILED=YES'])
            
    print('... Finished for %s Ma' % time)


def add_Pacific_synthetic_seamounts(
    time,
    output_paleobathydir,
    pacific_synthetic_seamounts_output_dir,
    output_paleobathydir_synseamounts,
    model_name,
    age_depth_model,
    sediment_thickness_model_name,
    LIP_feature_type,
    LIP_age_depth_model,
    create_final_geotiffs,
    paleobathymetry_main_output_dir,
):
    print('Adding synthetic_seamounts to paleobathymetry for %s Ma' % time)
    paleobathygrd = '%s/paleobathymetry_%sMa.nc' % (output_paleobathydir, time)

    synseamountgrd = '%s/synthetic_seamounts_%s.nc' % (pacific_synthetic_seamounts_output_dir, time)

    paleobathy_with_seamounts_grd = '%s/paleobathymetry_%sMa.nc' % (output_paleobathydir_synseamounts, time)

    if os.path.isfile(synseamountgrd):
        print('... Adding synthetic seamounts')
        call_system_command(('gmt', 'grdmath', '%s' % synseamountgrd, '0', 'DENAN', '%s' % paleobathygrd, 'ADD', '-Vn', '=', '%s' % paleobathy_with_seamounts_grd))
        # add better info to the netcdf itself
        call_system_command(('gmt', 'grdedit', '%s' % paleobathy_with_seamounts_grd, '-G%s' % paleobathy_with_seamounts_grd, '-fg', '-Vn',
            '-D+t"Paleobathymetry at %s Ma. Created %s"+ddepth[m]+r"Plate model: %s; depth-to-basement: %s; sediment thickness: %s; %s using %s; Includes synthetic seamounts"' % (time,
                datetime.now().strftime('%Y-%m-%d %H:%M'), model_name, age_depth_model, sediment_thickness_model_name, LIP_feature_type, LIP_age_depth_model)))
    else:
        print('... No synthetic seamounts exist at %s Ma' % time)
        call_system_command(('cp', '%s' % paleobathygrd, '%s' % paleobathy_with_seamounts_grd))
        call_system_command(('gmt', 'grdedit', '%s' % paleobathy_with_seamounts_grd, '-G%s' % paleobathy_with_seamounts_grd, '-fg', '-Vn',
            '-D+t"Paleobathymetry at %s Ma. Created %s"+ddepth[m]+r"Plate model: %s; depth-to-basement: %s; sediment thickness: %s; %s using %s; NOTE: no synthetic seamounts were actually added at this time"' % (time, 
                datetime.now().strftime('%Y-%m-%d %H:%M'), model_name, age_depth_model, sediment_thickness_model_name, LIP_feature_type, LIP_age_depth_model)))
    print('... Finished making netcdf for %s Ma' % time)

    if create_final_geotiffs.lower() in ['true', '1', 't', 'y', 'yes']:

        paleobathy_with_synseamounts_grid_basename = 'paleobathymetry_'
        paleobathy_with_synseamounts_grid_filename = os.path.join(output_paleobathydir_synseamounts, '{0}{1}Ma.nc'.format(paleobathy_with_synseamounts_grid_basename, time))

        tiff_directory_tiff = '%s/geotiff' % paleobathymetry_main_output_dir

        tiff_filename_tiff = os.path.join(tiff_directory_tiff, '{0}{1}Ma.tiff'.format(paleobathy_with_synseamounts_grid_basename, time))

        # add some compression when making the geotiff
        call_system_command(['gdal_translate', 
            'NETCDF:"%s":z' % paleobathy_with_synseamounts_grid_filename, '%s' % tiff_filename_tiff, '-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=3', '-co', 'TILED=YES'])


def calculate_paleobathymetry(
    sedthick_filename,
    basement_depth_filename,
    lip_height_filename,
    output_filename=None,
):
    sedthick = xr.load_dataset(sedthick_filename)
    sedthick = sedthick.rename_dims({"x": "lon", "y": "lat"})
    sedthick = sedthick.rename_vars({"x": "lon", "y": "lat"})
    sedthick = sedthick.astype("float64")
    sedthick = sedthick["z"].data

    basement_depth = xr.load_dataset(basement_depth_filename)
    basement_depth = basement_depth["z"].data

    lip_height = xr.load_dataset(lip_height_filename)
    out_template = lip_height
    lip_height = lip_height.fillna(0.0)
    lip_height = lip_height["z"].data

    # Isostatic correction
    tmp1 = (sedthick / 1000.0) * 0.43422
    tmp2 = ((sedthick / 1000.0) ** 2) * 0.010395
    tmp3 = (tmp1 - tmp2) * 1000.0
    isocorr = (tmp3 >= 0.0) * tmp3
    depth2base_and_seds = basement_depth + sedthick - isocorr

    paleobath = out_template.copy(data={"z": depth2base_and_seds + lip_height})
    # paleobath = lip_height + depth2base_and_seds
    if output_filename is not None:
        paleobath.to_netcdf(
            output_filename,
            encoding={
                i: {"zlib": True}
                for i in paleobath.data_vars
            },
        )
        return None
    return paleobath

"""Taken from paleobathymetry-workflow/traditional_workflow/3a_LIP_reconstruction_workflow/LIP_reconstruction_workflow.py"""
from datetime import datetime
import os

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygplates
import rasterio
import rasterio.features
import scipy.stats as spstats
import shapely
import xarray as xr


def area_weighted_mean(ds):
    weights = np.cos(np.deg2rad(ds.y))
    weights.name = "weights"
    ds_weighted = ds.weighted(weights)
    weighted_mean = ds_weighted.mean(("x", "y"))
    return weighted_mean

def calculate_depth_contours(
    basement_depth,
    gdf_LIP_shapes,
    contour_interval,
    LIP_depth_rounding,
    LIP_output_dir,
    clip_contour_to_outline,
    LIP_model_name,
    path_contoured_LIPs,
    feature_type,
    gdf_LIP_large=None,
):

    # Get depth of LIPs only
    clipped_LIPs = basement_depth.rio.clip(gdf_LIP_shapes.geometry, gdf_LIP_shapes.crs, invert=False)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 1. Create depth contours

    # calculate min and max values for contours
    min_level = None
    if min_level is None:
        min_value = clipped_LIPs.min()
        min_level = contour_interval * np.floor((min_value) / contour_interval)

    max_value = clipped_LIPs.max()

    # Due to range issues, a level is added
    max_level = contour_interval * (1 + np.ceil(max_value / contour_interval))

    cont_levels = np.arange(min_level, max_level, contour_interval)

    print('...... Calculating LIP contours')


    contour_gdf = gpd.GeoDataFrame()
    contour_gdf['CON_ELEV'] = None
    contour_gdf['geometry'] = None
    contour_gdf = contour_gdf.set_crs(epsg=4326)
    polys = []
    elev = []
    for cont_level in cont_levels:
        cs = plt.contour(clipped_LIPs.x, clipped_LIPs.y, clipped_LIPs, [cont_level])
        for contour_path in cs.get_paths():
            # create the polygon for this level
            for ncp, cp in enumerate(contour_path.to_polygons()):
                lons = cp[:, 0]
                lats = cp[:, 1]
                new_shape = shapely.geometry.Polygon(
                    [(i[0], i[1]) for i in zip(lons, lats)])
                new_shape = new_shape.buffer(0.1)  # try to avoid invalid topologies

                # 20240603: for some reason I'm now missing all my LIPs. Commenting out above to try this instead.
                poly = new_shape
                polys.append(poly)
                elev.append(cont_level)

    # ## -------- With matplotlib. 
    # ##  create contours
    # cs = plt.contourf(clipped_LIPs.x, clipped_LIPs.y, clipped_LIPs, cont_levels)
    # # ----
    # # Get contours for each LIP into a geodataframe
    # # Using code from here: https://chrishavlin.com/tag/shapely/

    # contour_gdf = gpd.GeoDataFrame()
    # contour_gdf['CON_ELEV'] = None
    # contour_gdf['geometry'] = None
    # contour_gdf = contour_gdf.set_crs(epsg=4326)

    # # loop over collections (and polygons in each collection), store in list
    # polys = []
    # elev = []

    # # create lookup table so we can get the contour level out
    # lvl_lookup = dict(zip(cs.collections, cs.levels))
    # for col in cs.collections:
    #     z = lvl_lookup[col]  # the value of this level
    #     for contour_path in col.get_paths():
    #         # create the polygon for this level
    #         for ncp, cp in enumerate(contour_path.to_polygons()):
    #             lons = cp[:, 0]
    #             lats = cp[:, 1]
    #             new_shape = shapely.geometry.Polygon(
    #                 [(i[0], i[1]) for i in zip(lons, lats)])

    #             # 20240603: for some reason I'm now missing all my LIPs. Commenting out above to try this instead.
    #             poly = new_shape

    #             polys.append(poly)
    #             elev.append(z)

    contour_gdf['geometry'] = polys
    contour_gdf['CON_ELEV'] = elev

    # --------
    # --- Get depths of surrounding oceanic crust
    # i.e., 'large LIP'
    if gdf_LIP_large is not None:
        # get polygons of just the large LIP (remove the actual LIP)
        gdf_LIP_large_only = gdf_LIP_large.overlay(gdf_LIP_shapes, how='difference')

        # get mean depth of ocean crust for the large-LIP polygons
        large_depths_means = []

        # Loop through and get ocean floor stats for surrounding crust
        for index in range(len(gdf_LIP_large_only)):

            # get area surrounding LIP ONLY
            large_depths = basement_depth.rio.clip(
                gdf_LIP_large_only.loc[[index]].geometry, gdf_LIP_large_only.crs, invert=False)
            # get mean depth of surrounding ocean crust
            large_depths_mean = LIP_depth_rounding * np.round((area_weighted_mean(large_depths)) / LIP_depth_rounding)
            large_depths_means.append(float(large_depths_mean.values))

        # add arrays to the original large LIP gdf
        gdf_LIP_large['LRG_ELEV'] = large_depths_means

        # ------------
        # get LIPs that are in common with the large LIP file
        # combine large and actual LIPS
        common_LIPs_tmp1 = gdf_LIP_large.sjoin(
            gdf_LIP_shapes, how='right', predicate='intersects', lsuffix='large', rsuffix='actual')
        # remove anything that is a NaN!
        common_LIPs_tmp2 = common_LIPs_tmp1[common_LIPs_tmp1['NAME_large'].notna()]
        # only process things where the FROMAGE in the large and actual LIP is the same
        common_LIPs = common_LIPs_tmp2[common_LIPs_tmp2['FROMAGE_large'] == common_LIPs_tmp2['FROMAGE_actual']]

        common_LIPs.reset_index(inplace=True)
        common_LIPs = common_LIPs.drop(['index_large', 'index'], axis=1)
    else:
        # give everything a unique name, just in case we have duplicate rows with the same name
        gdf_LIP_shapes['NAME'] = gdf_LIP_shapes['NAME'].astype(str) + '_' + gdf_LIP_shapes.index.astype(str)
        common_LIPs = gdf_LIP_shapes
    
    common_LIPs['Area'] = common_LIPs.area
    # can't resolve things below a certain area (too small for the output resolution). Drop and re-index
    common_LIPs = common_LIPs[common_LIPs.Area > 0.02]
    common_LIPs.reset_index(inplace=True)
    common_LIPs = common_LIPs.drop(['index'], axis=1)

    # for trying to avoid invalid geometry errors
    common_LIPs['geometry'] = common_LIPs.buffer(0)
 
    # +++++++++++++++++++++++

    if clip_contour_to_outline.lower() in ['true', '1', 't', 'y', 'yes']:
    # Clip/cut contours to LIP outlines by iterating through.
    # otherwise they might go outside the original polygon!
    # this is INCREDIBLY slow for seamounts...
        print('...... Clipping LIP contours to original LIP outline. Note: this can take a few mins, please be patient!')

        # 26/11/22: Don't think we nede to loop through anymore. Left code here just in case

        # clipped_LIPs_contour = []
        # for i, row in common_LIPs.iterrows():
        #     print(i)
        #     single = gpd.GeoDataFrame(row.to_frame().T, crs=common_LIPs.crs)
        #     clipped_LIP_contour = gpd.clip(contour_gdf, single)
        #     clipped_LIPs_contour.append(clipped_LIP_contour)
        # clipped_LIPs_gdf = gpd.GeoDataFrame(
        #     pd.concat(clipped_LIPs_contour), crs=common_LIPs.crs)

        clipped_LIPs_gdf = gpd.clip(contour_gdf, common_LIPs)

        if gdf_LIP_large is None:
            common_LIPs['NAME_large'] = common_LIPs['NAME'].astype(str) + '_polygon'
            common_LIPs['NAME_actual'] = common_LIPs['NAME']

        contoured_LIPs = gpd.overlay(common_LIPs, clipped_LIPs_gdf, how='intersection')
    # Add contours
    else:
        if gdf_LIP_large is None:
            common_LIPs['NAME_large'] = common_LIPs['NAME'].astype(str) + '_polygon'
            common_LIPs['NAME_actual'] = common_LIPs['NAME']

        print('...... Adding LIP contours. Note: they might go outside the original LIP outline!')
        contoured_LIPs = contour_gdf.sjoin(common_LIPs, how='inner')

    print('...... Combining contours and LIPs')

    # ------ Calculating alternative base LIP depth
    # want the mean depth of the LIP outline
    common_LIPs_buff_smaller = common_LIPs.copy()
    common_LIPs_buff_smaller['geometry'] = common_LIPs_buff_smaller.buffer(-0.5)

    common_LIPs_buff_larger = common_LIPs.copy()
    common_LIPs_buff_larger['geometry'] = common_LIPs_buff_larger.buffer(0.5)

    # get thin polygon of LIP outline
    diff = common_LIPs_buff_larger.overlay(common_LIPs_buff_smaller, how='difference')

    LIP_depths_means = []

    for index in range(len(diff)):
        # get depths of LIP outline only
        LIP_depths = basement_depth.rio.clip(diff.loc[[index]].geometry, diff.crs, invert=False)
        
        LIP_depths_mean = LIP_depth_rounding * np.round((area_weighted_mean(LIP_depths)) / LIP_depth_rounding)
        LIP_depths_means.append(float(np.round(LIP_depths_mean.values)))

    # add arrays to the original large LIP gdf
    common_LIPs['LIP_outline_means'] = LIP_depths_means

    # group and get the min elevation for the LIP and rename as the LIP outline
    grouped_large = contoured_LIPs.groupby('NAME_large')['CON_ELEV']

    # add the min elev of the LIP outline to gdf, and rename in new 'NAME' column
    # common_LIPs is the outline only
    print(common_LIPs)

    # create the LIP_depth_contour file first
    f = open('%s/%s/LIP_depth_contours.txt' % (LIP_output_dir, LIP_model_name), 'w')

    for i in range(len(common_LIPs)):
        for j in grouped_large:
            if common_LIPs.NAME_large.loc[i] == j[0]:
                # print out some stats...

                if gdf_LIP_large is None:
                    # print("%s \t deepest contour: %s, \t outline depth: %s" % 
                    #     (j[0], float(min(j[1])), common_LIPs.loc[i, 'LIP_outline_means']))

                    # Append to the text file
                    with open('%s/%s/LIP_depth_contours.txt' % (LIP_output_dir, LIP_model_name), 'a') as f:
                        f.write("%s \t deepest contour: %s, \t outline depth: %s \n" % (j[0], float(min(j[1])), common_LIPs.loc[i, 'LIP_outline_means']))

                else:
                    # print("%s \t deepest contour: %s, \t outline depth: %s, \t lrg_elev: %s" % 
                    #     (j[0], float(min(j[1])), common_LIPs.loc[i, 'LIP_outline_means'], common_LIPs.loc[i, 'LRG_ELEV']))

                    with open('%s/%s/LIP_depth_contours.txt' % (LIP_output_dir, LIP_model_name), 'a') as f:
                        f.write("%s \t deepest contour: %s, \t outline depth: %s, \t lrg_elev: %s \n" % 
                            (j[0], float(min(j[1])), common_LIPs.loc[i, 'LIP_outline_means'], common_LIPs.loc[i, 'LRG_ELEV']))

                # ---- set the elevation of the outline
                # common_LIPs.loc[i, 'ELEV'] = float(min(j[1]))  # set the elevation contour of outline to equal min elev of the LIP
                
                # set the elevation contour to be the elevation of the rounded LIP outline.
                # this is because we can have deeper elevation contours inside this, 
                # so we would be overestimating LIP height by just using the deepest contour.
                common_LIPs.loc[i, 'ELEV'] = common_LIPs.loc[i, 'LIP_outline_means']
                
                ## rename 'polygon' to 'outline' in LIP name - so we have a nice new row of just the outline
                name = common_LIPs.NAME_large.loc[i]
                new_name = name.replace('polygon', 'outline')
                common_LIPs.loc[i, 'NAME_large'] = new_name
            else:
                pass

    # append_outline to the bottom
    # contoured_LIPs = contoured_LIPs.append(common_LIPs)
    contoured_LIPs = pd.concat([contoured_LIPs, common_LIPs], ignore_index=True)
    contoured_LIPs.reset_index(inplace=True)         # reset the index
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print('...... Calculating LIP heights')
    # copy deepest contour (from the outline) to all rows

    contoured_LIPs['ELEV_min'] = contoured_LIPs.groupby('NAME_actual')['ELEV'].transform('mean')

    def _calculate_height_from_elev(row):
        if pd.isna(row.CON_ELEV) is False:
            liph = row.CON_ELEV - row.ELEV_min
        else:
            # --- outlines won't have a value for CON_ELEV
            liph = 0
        return liph

    def _calculate_height_from_lrg_elev(row):
        if pd.isna(row.CON_ELEV) is False:
            # print('Valid contours in', contoured_LIPs.NAME_large.loc[i])
                
            # check if the min contour is deeper than the large elev. 
            # if the min contour is deeper than the large elev, result is < 0
            # if the large elev contour is deeper, result is > 0
            if row.ELEV_min - row.LRG_ELEV > 0:
                # use the deeper contour (LRG contour)
                liph = row.CON_ELEV - row.LRG_ELEV
            elif row.ELEV_min - row.LRG_ELEV < 0:
                # large elevation is too shallow, do everthing relative to elev_min
                liph = row.CON_ELEV - row.ELEV_min
        else:
            if row.ELEV - row.LRG_ELEV < 0:
                liph = 0
            if row.ELEV - row.LRG_ELEV >= 0:
                liph = row.ELEV - row.LRG_ELEV
        return liph

    # calculate LIP heights
    if 'LRG_ELEV' in contoured_LIPs.columns:
        print("...... Calculating LIP height based on surrounding ocean floor ('large LIP') vs deepest contour inside LIP")

        contoured_LIPs['LIPH'] = contoured_LIPs.apply(_calculate_height_from_lrg_elev, axis=1)

        for i in range(len(contoured_LIPs)): 
            # check that toage matches. They should all be to the future (-999)
            if contoured_LIPs.TOAGE_actual.loc[i] == contoured_LIPs.TOAGE_large.loc[i]:
                contoured_LIPs.loc[i, 'TOAGE'] = float(
                    contoured_LIPs.TOAGE_actual.loc[i])
            else:
                contoured_LIPs.loc[i, 'TOAGE'] = int(-999.)

            # check fromage
            if contoured_LIPs.FROMAGE_actual.loc[i] == contoured_LIPs.FROMAGE_large.loc[i]:
                contoured_LIPs.loc[i, 'FROMAGE'] = float(
                    contoured_LIPs.FROMAGE_actual.loc[i])
                # print('works!')
            else:
                # print('we have an issue')
                contoured_LIPs.loc[i, 'FROMAGE'] = 'FIX'

            # check PLATEID1
            if contoured_LIPs.PLATEID1_large.loc[i] == contoured_LIPs.PLATEID1_actual.loc[i]:
                contoured_LIPs.loc[i, 'PLATEID1'] = int(
                    contoured_LIPs.PLATEID1_actual.loc[i])
                # print('works!')
            else:
                # print('we have an issue')
                if contoured_LIPs.NAME_large.loc[i] == 'iceland_eurasia_polygon':
                    # print('large poly is %s, while actual is %s' % (contoured_LIPs.PLATEID1_large.loc[i], contoured_LIPs.PLATEID1_actual.loc[i]))
                    contoured_LIPs.loc[i, 'PLATEID1'] = int(
                        contoured_LIPs.PLATEID1_actual.loc[i])
                elif contoured_LIPs.NAME_large.loc[i] == 'iceland_eurasia_outline':
                    # print('large poly is %s, while actual is %s' % (contoured_LIPs.PLATEID1_large.loc[i], contoured_LIPs.PLATEID1_actual.loc[i]))
                    contoured_LIPs.loc[i, 'PLATEID1'] = int(
                        contoured_LIPs.PLATEID1_actual.loc[i])
                else:
                    # print('%s: large poly is %s, while actual is %s' % (contoured_LIPs.NAME_large.loc[i], contoured_LIPs.PLATEID1_large.loc[i], contoured_LIPs.PLATEID1_actual.loc[i]))
                    contoured_LIPs.loc[i, 'PLATEID1'] = 'FIX'

    else:
        print("...... No polygons for surrounding ocean floor ('large LIP') specified, using deepest contour inside LIP to calculate LIPH")
        contoured_LIPs['LIPH'] = contoured_LIPs.apply(_calculate_height_from_elev, axis=1)

    # if CON_ELEV is NaN, add a value
    contoured_LIPs['CON_ELEV'] = np.where(contoured_LIPs.CON_ELEV.isnull(), contoured_LIPs.ELEV, contoured_LIPs.CON_ELEV)
    
    # remove 'polygon' in LIP name
    contoured_LIPs['NAME'] = contoured_LIPs.NAME_large.str.replace('_polygon', '')

    # for i in range(len(contoured_LIPs)):
    #     print(i)
    #     # --- LIP heights
    #     # Check that surrounding crust is actually deeper - if not, use
    #     # the deepest (ELEV_min) LIP contour. 
    #     # Here, ELEV_min is based on the mean depth of the LIP outline.

    #     if 'LRG_ELEV' in contoured_LIPs.columns:
    #         print("...... Calculating LIP height based on surrounding ocean floor ('large LIP') vs deepest contour inside LIP")

    #         if pd.isna(contoured_LIPs.CON_ELEV.loc[i]) is False:
           
    #             # print('Valid contours in', contoured_LIPs.NAME_large.loc[i])
                
    #             # check if the min contour is deeper than the large elev. 
    #             # if the min contour is deeper than the large elev, result is < 0
    #             # if the large elev contour is deeper, result is > 0
    #             if contoured_LIPs.ELEV_min.loc[i] - contoured_LIPs.LRG_ELEV.loc[i] > 0:
    #                 # use the deeper contour (LRG contour)
    #                 contoured_LIPs.loc[i, 'LIPH'] = contoured_LIPs.CON_ELEV.loc[i] - contoured_LIPs.LRG_ELEV.loc[i]

    #             elif contoured_LIPs.ELEV_min.loc[i] - contoured_LIPs.LRG_ELEV.loc[i] < 0:
    #                 # large elevation is too shallow, do everthing relative to elev_min
    #                 contoured_LIPs.loc[i, 'LIPH'] = contoured_LIPs.CON_ELEV.loc[i] - contoured_LIPs.ELEV_min.loc[i]

    #         # --- outlines won't have a value for CON_ELEV
    #         if pd.isna(contoured_LIPs.CON_ELEV.loc[i]) is True:
    #             print('no contours in ', contoured_LIPs.NAME_large.loc[i])
    #             # calculate LIPH for outline.
    #             contoured_LIPs.loc[i, 'CON_ELEV'] = contoured_LIPs.ELEV.loc[i]
    #             if contoured_LIPs.ELEV.loc[i] - contoured_LIPs.LRG_ELEV.loc[i] < 0:
    #                 contoured_LIPs.loc[i, 'LIPH'] = 0
                    
    #             # otherwise use difference between ELEV and LRG for LIPH
    #             if contoured_LIPs.ELEV.loc[i] - contoured_LIPs.LRG_ELEV.loc[i] >= 0:
    #                 contoured_LIPs.loc[i, 'LIPH'] = contoured_LIPs.ELEV.loc[i] - contoured_LIPs.LRG_ELEV.loc[i]



    #         # check that toage matches. They should all be to the future (-999)
    #         if contoured_LIPs.TOAGE_actual.loc[i] == contoured_LIPs.TOAGE_large.loc[i]:
    #             contoured_LIPs.loc[i, 'TOAGE'] = float(
    #                 contoured_LIPs.TOAGE_actual.loc[i])
    #         else:
    #             contoured_LIPs.loc[i, 'TOAGE'] = int(-999.)

    #         # check fromage
    #         if contoured_LIPs.FROMAGE_actual.loc[i] == contoured_LIPs.FROMAGE_large.loc[i]:
    #             contoured_LIPs.loc[i, 'FROMAGE'] = float(
    #                 contoured_LIPs.FROMAGE_actual.loc[i])
    #             # print('works!')
    #         else:
    #             # print('we have an issue')
    #             contoured_LIPs.loc[i, 'FROMAGE'] = 'FIX'

    #         # check PLATEID1
    #         if contoured_LIPs.PLATEID1_large.loc[i] == contoured_LIPs.PLATEID1_actual.loc[i]:
    #             contoured_LIPs.loc[i, 'PLATEID1'] = int(
    #                 contoured_LIPs.PLATEID1_actual.loc[i])
    #             # print('works!')
    #         else:
    #             # print('we have an issue')
    #             if contoured_LIPs.NAME_large.loc[i] == 'iceland_eurasia_polygon':
    #                 # print('large poly is %s, while actual is %s' % (contoured_LIPs.PLATEID1_large.loc[i], contoured_LIPs.PLATEID1_actual.loc[i]))
    #                 contoured_LIPs.loc[i, 'PLATEID1'] = int(
    #                     contoured_LIPs.PLATEID1_actual.loc[i])
    #             elif contoured_LIPs.NAME_large.loc[i] == 'iceland_eurasia_outline':
    #                 # print('large poly is %s, while actual is %s' % (contoured_LIPs.PLATEID1_large.loc[i], contoured_LIPs.PLATEID1_actual.loc[i]))
    #                 contoured_LIPs.loc[i, 'PLATEID1'] = int(
    #                     contoured_LIPs.PLATEID1_actual.loc[i])
    #             else:
    #                 # print('%s: large poly is %s, while actual is %s' % (contoured_LIPs.NAME_large.loc[i], contoured_LIPs.PLATEID1_large.loc[i], contoured_LIPs.PLATEID1_actual.loc[i]))
    #                 contoured_LIPs.loc[i, 'PLATEID1'] = 'FIX'


    #     else:
    #         print("... No polygons for surrounding ocean floor ('large LIP') specified, using deepest contour inside LIP to calculate LIPH")
    #         if pd.isna(contoured_LIPs.CON_ELEV.loc[i]) is False:
    #             contoured_LIPs.loc[i, 'LIPH'] = contoured_LIPs.CON_ELEV.loc[i] - contoured_LIPs.ELEV_min.loc[i]

    #         # --- outlines won't have a value for CON_ELEV
    #         if pd.isna(contoured_LIPs.CON_ELEV.loc[i]) is True:
    #             # calculate LIPH for outline.
    #             contoured_LIPs.loc[i, 'CON_ELEV'] = contoured_LIPs.ELEV.loc[i]
    #             contoured_LIPs.loc[i, 'LIPH'] = 0


        # # remove 'polygon' in LIP name
        # name = contoured_LIPs.NAME_large.loc[i]
        # new_name = name.replace('_polygon', '')
        # # print('name: %s \t new_name: %s' % (name, new_name))
        # contoured_LIPs.loc[i, 'NAME'] = new_name

    
    if gdf_LIP_large is not None:
        # delete unneeded columns
        contoured_LIPs = contoured_LIPs.drop(['TOAGE_large', 'FROMAGE_large', 'PLATEID1_large', 'TOAGE_actual', 'FROMAGE_actual', 'PLATEID1_actual', 'Id'], axis=1)
    
    # delete unneeded columns
    contoured_LIPs = contoured_LIPs.drop(['NAME_large', 'index'], axis=1)

    contoured_LIPs.sort_values(
        by=['NAME', 'LIPH'], inplace=True, ignore_index=True)

    contoured_LIPs = contoured_LIPs[contoured_LIPs['LIPH'].notna()]

    # save file
    # bug in this version of fiona/proj?
    # Workaround from https://github.com/geopandas/geopandas/issues/1697#issuecomment-878705571
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        contoured_LIPs.to_file('%s/contoured_%s_0Ma.shp' % (path_contoured_LIPs, feature_type))
    print('... Finished part 1!')

# ------------------------------------------
# --- Part 2: swell heights and optional buffer

# NOTE: These are copied from the previous workflow.
# NW edits: - Richards et al. relationship has been added to the swell functions
#             (using same workaround as the age-to-depth workflow)
#           - Additional comments have been added throughout
#           - NO CHANGE to the other age-depth relationships

# function to calculate the height multiplier for the buffer
# (dependant on the distance from the LIP)
def buffer_height_calc(distance, buffer_radius_deg):
    # calculate buffer height multiplier
    height_mul = np.cos(np.pi * distance / (buffer_radius_deg * 2))
    return height_mul

# function to calculate the swell constant needed to obtain necessary
# swell height for present day basement depths
def const_calc(age, con_elev, LIPH, cool, RHCW_age_depth_interp=None):
    """ Function to calculate the swell height and swell constant,
    based on present-day basement depth

    Inputs:
        - age (should be 0 Ma)
        - con_elev: contour depth (in metres below sea level)
        - LIPH: LIP height in metres (positive value)
        - cool: age-depth model
        - RHCW_age_depth_interp: if required (for RHCW cool model)
    """
    # swell_const = np.NaN
    time_from_creation = age - 0

    if 'hasterok' in cool:
        # Hasterok
        # swell decay equation for the LIP
        if time_from_creation <= 17.4:
            base_elev = -1 * (2500 + 414.5 * (time_from_creation ** 0.5))
            swell_height = con_elev - base_elev - LIPH
            swell_const = swell_height + 414.5 * (time_from_creation ** 0.5)
        elif time_from_creation > 17.4:
            base_elev = -1 * \
                (5609 - 2520 * np.exp(-0.034607 * time_from_creation))
            swell_height = con_elev - base_elev - LIPH
            swell_const = swell_height + \
                (3109 - 2520 * np.exp(-0.034607 * time_from_creation))

    elif 'GDH1' in cool:
        # stein and stein
        # swell decay equation for the LIP
        if time_from_creation <= 20:
            base_elev = -1 * (2600 + 365 * (time_from_creation ** 0.5))
            swell_height = con_elev - base_elev - LIPH
            swell_const = swell_height + 365 * (time_from_creation ** 0.5)
        elif time_from_creation > 20:
            base_elev = -1 * \
                (5651 - 2473 * np.exp(-0.0278 * time_from_creation))
            swell_height = con_elev - base_elev - LIPH
            swell_const = swell_height + \
                (3051 - 2473 * np.exp(-0.0278 * time_from_creation))

    elif 'PS_TBL' in cool:
        # parsons and sclater
        # swell decay equation for the LIP
        if time_from_creation <= 20:
            base_elev = -1 * (2500 + 350 * (time_from_creation ** 0.5))
            swell_height = con_elev - base_elev - LIPH
            swell_const = swell_height + 350 * (time_from_creation ** 0.5)
        elif time_from_creation > 20:
            base_elev = -1 * (6400 - 3200 * np.exp(-1 * time_from_creation / 62.8))
            swell_height = con_elev - base_elev - LIPH
            swell_const = swell_height + \
                (3900 - 3200 * np.exp(-1 * time_from_creation / 62.8))

    elif 'hayes' in cool:
        # hayes
        # swell decay equation for the LIP
        base_elev = -1 * (2700 + 300 * (time_from_creation ** 0.5))
        swell_height = con_elev - base_elev - LIPH
        swell_const = swell_height + 300 * (time_from_creation ** 0.5)

    elif 'static_swell' in cool:
        # Hasterok
        # swell decay equation for the LIP
        if time_from_creation <= 17.4:
            base_elev = -1 * (2500 + 414.5 * (time_from_creation ** 0.5))
            swell_height = con_elev - base_elev - LIPH
            swell_const = swell_height
        elif time_from_creation > 17.4:
            base_elev = -1 * \
                (5609 - 2520 * np.exp(-0.034607 * time_from_creation))
            swell_height = con_elev - base_elev - LIPH
            swell_const = swell_height

    elif 'RHCW18' in cool:
        # calculate base elevation - use dataframe to get depth for corresponding age
        depth = (RHCW_age_depth_interp.loc[RHCW_age_depth_interp['age'] == time_from_creation, 'depth'].item())
        base_elev = -1 * depth
        # print('base_elev', base_elev)
        swell_height = con_elev - base_elev - LIPH

        # second part ('depth difference') of the swell_const is the
        # difference between the depth for the time it formed to 0 Ma
        # (-2500 m at 0 Ma). This is what the other models do as well
        # (they just use the equation for it)
        # depth_difference = (RHCW_age_depth_interp.loc[RHCW_age_depth_interp['age'] == time_from_creation, 'depth'].item() - 2500)

        depth_difference = depth - 2500
        swell_const = swell_height + depth_difference

    return [swell_const, swell_height]

# function to calculate swell height back in time
def swell_calc(time, age, LIPH, cool, swell_const, RHCW_age_depth_interp=None):
    """ Function to calculate the swell height at a time in the past,
    based on the swell constant and other parameters

    Inputs:
        - age (should not be 0 Ma)
        - con_elev: contour depth (in metres below sea level)
        - LIPH: LIP height in metres (positive value)
        - cool: age-depth model
        - swell_const: present-day swell constant, calculated in const_calc
        - RHCW_age_depth_interp: if required (for RHCW cool model)
    """

    # round age to 2dp so that it will work with the RHCW table
    time_from_creation = np.round(age - time, 2)

    # Cooling models
    if 'hasterok' in cool:
        # Hasterok
        # swell decay equation for the LIP
        if time_from_creation <= 17.4:
            swell_height = (swell_const - 414.5 * (time_from_creation ** 0.5))
        elif time_from_creation > 17.4:
            swell_height = swell_const - \
                (3109 - 2520 * np.exp(-0.034607 * time_from_creation))

    elif 'GDH1' in cool:
        # stein and stein
        # swell decay equation for the LIP
        if time_from_creation <= 20:
            swell_height = (swell_const - 365 * (time_from_creation ** 0.5))
        elif time_from_creation > 20:
            swell_height = swell_const - \
                (3051 - 2473 * np.exp(-0.0278 * time_from_creation))

        # print(swell_height)
        # print(type(swell_height))
    elif 'PS_TBL' in cool:
        # parsons and sclater
        # swell decay equation for the LIP
        if time_from_creation <= 20:
            swell_height = (swell_const - 350 * (time_from_creation ** 0.5))
        elif time_from_creation > 20:
            swell_height = swell_const - \
                (3900 - 3200 * np.exp(-1 * time_from_creation / 62.8))

    elif 'hayes' in cool:
        # hayes
        # swell decay equation for the LIP
        swell_height = (swell_const - 300 * (time_from_creation ** 0.5))

    elif 'no_swell' in cool:
        # no swell
        swell_height = 0

    elif 'static_swell' in cool:
        swell_height = swell_const

    elif 'RHCW18' in cool:
        # second part ('depth difference') of the swell_const is the
        # difference between the depth for the time it formed to 0 Ma
        # (-2500 m at 0 Ma). This is what the other models do as well
        # (they just use the equation for it)

        # different ways to do this next step...
        # RHCW_age_depth_interp_2 = RHCW_age_depth_interp.set_index('age')
        # depth = np.array(RHCW_age_depth_interp_2[RHCW_age_depth_interp_2.index == time_from_creation])
        # RHCW_age_depth_interp_np = np.array(RHCW_age_depth_interp)
        # depth = RHCW_age_depth_interp_np[(RHCW_age_depth_interp_np[:,0]) == time_from_creation, 1][0]

        depth = (RHCW_age_depth_interp.loc[RHCW_age_depth_interp['age'] == time_from_creation, 'depth'].item())
        depth_difference = depth - 2500.0
        swell_height = swell_const - depth_difference

    if swell_height < 0.0:
        # swell_height = swell_height
        swell_height = 0
    # subtract the swell calculation from the lip-height
    height = LIPH + swell_height

    return [swell_height, height]


def create_geodataframe_LIPs(pygplates_recon_geom, reconstruction_time):
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
    recon_gdf['PLATEID2'] = None
    recon_gdf['FROMAGE'] = None
    recon_gdf['TOAGE'] = None
    recon_gdf['geometry'] = None
    recon_gdf['reconstruction_time'] = None
    recon_gdf = recon_gdf.set_crs(epsg=4326)

    date_line_wrapper = pygplates.DateLineWrapper()

    names                = []
    plateid1s            = []
    plateid2s            = []
    fromages             = []
    toages               = []
    geometrys            = []
    reconstruction_times = []
    sources = []
    con_elevs = []
    elevs = []
    lip_outlines = []
    elev_mins = []
    lip_heights = []

    for i, seg in enumerate(pygplates_recon_geom):
        if isinstance(seg, pygplates.ReconstructedFeatureGeometry):
            wrapped_polygons = date_line_wrapper.wrap(seg.get_reconstructed_geometry())
            for poly in wrapped_polygons:
                ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
                ring[:,1] = np.clip(ring[:, 1], -89.9, 89.9) # anything approaching the poles creates artefacts

                name = seg.get_feature().get_name()
                plateid = seg.get_feature().get_reconstruction_plate_id()
                conjid = seg.get_feature().get_conjugate_plate_id()
                from_age, to_age = seg.get_feature().get_valid_time()
                
                # get LIP specific things
                source = seg.get_feature().get_shapefile_attribute('Source')
                con_elev = seg.get_feature().get_shapefile_attribute('CON_ELEV')
                elev = seg.get_feature().get_shapefile_attribute('ELEV')
                lip_outline = seg.get_feature().get_shapefile_attribute('LIP_outlin')
                elev_min = seg.get_feature().get_shapefile_attribute('ELEV_min')
                lip_height = seg.get_feature().get_shapefile_attribute('LIPH')
                
                # append things
                names.append(name)
                plateid1s.append(plateid)
                plateid2s.append(conjid)
                fromages.append(from_age)
                toages.append(to_age)
                geometrys.append(shapely.geometry.Polygon(ring))
                reconstruction_times.append(reconstruction_time)
                sources.append(source)
                con_elevs.append(con_elev)
                elevs.append(elev)
                lip_outlines.append(lip_outline)
                elev_mins.append(elev_min)
                lip_heights.append(lip_height)

        elif isinstance(seg, pygplates.Feature):
            wrapped_polygons = date_line_wrapper.wrap(seg.get_reconstructed_geometry())
            for poly in wrapped_polygons:
                ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
                ring[:,1] = np.clip(ring[:, 1], -89.9, 89.9) # anything approaching the poles creates artefacts

                name = seg.get_feature().get_name()
                plateid = seg.get_feature().get_reconstruction_plate_id()
                conjid = seg.get_feature().get_conjugate_plate_id()
                from_age, to_age = seg.get_feature().get_valid_time()
                
                # get LIP specific things
                source = seg.get_feature().get_shapefile_attribute('Source')
                con_elev = seg.get_feature().get_shapefile_attribute('CON_ELEV')
                elev = seg.get_feature().get_shapefile_attribute('ELEV')
                lip_outline = seg.get_feature().get_shapefile_attribute('LIP_outlin')
                elev_min = seg.get_feature().get_shapefile_attribute('ELEV_min')
                lip_height = seg.get_feature().get_shapefile_attribute('LIPH')
                
                # append things
                names.append(name)
                plateid1s.append(plateid)
                plateid2s.append(conjid)
                fromages.append(from_age)
                toages.append(to_age)
                geometrys.append(shapely.geometry.Polygon(ring))
                reconstruction_times.append(reconstruction_time)
                sources.append(source)
                con_elevs.append(con_elev)
                elevs.append(elev)
                lip_outlines.append(lip_outline)
                elev_mins.append(elev_min)
                lip_heights.append(lip_height)

    # write to geodataframe
    recon_gdf['NAME'] = names
    recon_gdf['PLATEID1'] = plateid1s
    recon_gdf['PLATEID2'] = plateid2s
    recon_gdf['FROMAGE'] = fromages
    recon_gdf['TOAGE'] = toages
    recon_gdf['reconstruction_time'] = reconstruction_times
    recon_gdf['Source'] = sources
    recon_gdf['CON_ELEV'] = con_elevs
    recon_gdf['LIP_outlin'] = lip_outlines
    recon_gdf['ELEV'] = elevs
    recon_gdf['ELEV_min'] = elev_mins
    recon_gdf['LIPH'] = lip_heights

    recon_gdf['geometry'] = geometrys

    return recon_gdf


# --- main function for part 2: reconstruct things and then apply buffer
def rotation_LIP_shapefile_and_buffer(
    time,
    path_0_shp,
    rotation_filenames,
    path_rotated_polygons,
    feature_type,
    include_conjugate_LIPs,
    LIP_output_dir,
    LIP_model_name,
    cooling_model,
    large_LIP_path,
    buffer_radius_deg,
    RHCW_age_depth_interp=None,
):
    print('...... Working on %s Ma, creating %s/reconstructed_%s_%sMa.shp' % (time, path_rotated_polygons, feature_type, time))

    # ------------------------------------------------
    # --- read in files
    # because multiprocessing complains about it not being able to pickle pygplates things
    # read all the pygplates files into the function here instead
    LIPs_0Ma = pygplates.FeatureCollection(path_0_shp)

    # --- add conjugate LIPs if desired
    if include_conjugate_LIPs.lower() in ['true', '1', 't', 'y', 'yes']:
        # read in conjugate LIPs (from jupyter notebook)
        conj_shp = os.path.join(LIP_output_dir, "LIP_conjugates_0Ma.shp")
        LIP_conj = pygplates.FeatureCollection(conj_shp)
        print('...... Including conjugate LIPs')
        LIPs_0Ma.add(LIP_conj)
    else:
        # print("...... Not including conjugates LIPs")
        pass

    # ---read in rotation model
    rotation_model = pygplates.RotationModel(rotation_filenames)

    # ------------------------------------------------
    # reconstruct present-day LIP shapefile to time in the past
    rotated_polygon = []   # output of pygplates.reconstruct 
    
    pygplates.reconstruct(LIPs_0Ma,rotation_model, rotated_polygon, float(time), export_wrap_to_dateline=True)
    
    # convert pygplates reconstructed geometry (polygon) to geopandas dataframe
    gdf_merged_LIPs = create_geodataframe_LIPs(rotated_polygon, time)

    # ------------------------------------------------
    # Calculate swell parameters and apply optional buffer.
    # Separate into 0 Ma, and != 0 Ma
    # since we will save out a txt file of the swell parameters from 0 Ma.

    # check if there are any valid polygons at this time, otherwise skip.
    if gdf_merged_LIPs.empty is True:
        print('No LIPs reconstructed at %s Ma' % time)
        pass
    else:

        # ---
        # Import 0 Ma only
        if time == 0.0:

            # create empty geodataframe for the buffer geometries
            gdf_buffer = gpd.GeoDataFrame()
            gdf_buffer['geometry'] = None
            gdf_buffer = gdf_buffer.set_crs(epsg=4326)
            print('...... Calculating swell stats for %s Ma' % time)

            # if it is present day and not the 'no swell' cooling model, then calculate the swell constant
            if not cooling_model == 'no_swell':
                if RHCW_age_depth_interp is not None:
                    gdf_merged_LIPs[['swell_const', 'swell_height']] = gdf_merged_LIPs.apply(lambda row: const_calc(
                        row['FROMAGE'],
                        row['CON_ELEV'],
                        row['LIPH'],
                        cooling_model, RHCW_age_depth_interp), axis=1, result_type='expand')
                else:
                    gdf_merged_LIPs[['swell_const', 'swell_height']] = gdf_merged_LIPs.apply(lambda row: const_calc(
                        row['FROMAGE'],
                        row['CON_ELEV'],
                        row['LIPH'],
                        cooling_model), axis=1, result_type='expand')

                if large_LIP_path is not None:
                    # I think the following is only needed if we use the large LIPs?
                    # ONLY doing if we use the large LIP polygon, because it takes a while (~15 mins)

                    # Getting weird swell constant/height values for the 'outline' version (because of an issue with LIPH?),
                    # so use the value from the rest of the LIP
                    LIPs_grouped_name = gdf_merged_LIPs.groupby('NAME')['swell_const']

                    print('... Fixing swell constant in LIP outlines')
                    for i in range(len(gdf_merged_LIPs)):
                        for j in LIPs_grouped_name:
                            if gdf_merged_LIPs.NAME.loc[i] == j[0]:
                                pass
                            elif 'outline' in gdf_merged_LIPs['NAME'].loc[i] and j[0] in gdf_merged_LIPs['NAME'].loc[i]:
                                # change to equal the swell constant for the rest of the LIP
                                gdf_merged_LIPs.loc[i, 'swell_const'] = float(
                                    np.mean(j[1]))
                else:
                    pass

                # --- write out some swell stats
                gdf_merged_LIPs_outline = gdf_merged_LIPs[gdf_merged_LIPs['NAME'].str.contains(
                    'outline')]
                gdf_stats = gdf_merged_LIPs_outline[[
                    'NAME', 'FROMAGE', 'swell_const', 'swell_height']]

                stats = pd.DataFrame(gdf_stats)  # convert to pandas
                stats.to_csv(LIP_output_dir + '/%s/%s_swell_stats.txt' % (LIP_model_name, cooling_model),
                             sep='\t', header=['NAME', 'FROMAGE', 'swell_const', 'swell_height'], index=False)
                print('...... Saved swell constant to textfile')

            elif cooling_model == "no_swell":
                print('No swell')
                gdf_merged_LIPs['swell_const'] = 0

            # make sure to pass the RCHW relationship if it exists
            if RHCW_age_depth_interp is not None:
                # calculate swell meta
                # swell_meta[0] - the swell height. constant for all height values
                gdf_merged_LIPs[['swell_height', 'HEIGHT']] = gdf_merged_LIPs.apply(lambda row: swell_calc(
                    float(time), row['FROMAGE'], row['LIPH'], cooling_model, row['swell_const'], RHCW_age_depth_interp), axis=1, result_type='expand')

            else:
                # calculate swell meta
                # swell_meta[0] - the swell height. constant for all height values
                gdf_merged_LIPs[['swell_height', 'HEIGHT']] = gdf_merged_LIPs.apply(lambda row: swell_calc(
                    float(time), row['FROMAGE'], row['LIPH'], cooling_model, row['swell_const']), axis=1, result_type='expand')

            # create buffer if needed
            if "buffer" in cooling_model:
                # use only the outlines of the LIPs
                gdf_merged_LIPs_outline = gdf_merged_LIPs[gdf_merged_LIPs['NAME'].str.contains('outline')]

                print('...... Calculating buffer')
                # work out number of polygons to cover the buffer distance in increments of 0.1 degrees
                poly_no = int(buffer_radius_deg / 0.1)
                # iterate through each buffer polygon (starting closest to LIP)
                for j in range(poly_no):
                    poly_ID = j + 1
                    # make a copy of the outline file, where we can will overwrite geometry and height
                    gdf_buffer_tmp = gdf_merged_LIPs_outline.copy()

                    # save feature to new gpd
                    # rename 'polygon' to 'outline' in LIP name
                    name = gdf_buffer_tmp['NAME']
                    new_name = name.str.replace('outline', 'buffer')
                    # print(new_name)
                    gdf_buffer_tmp['NAME'] = new_name

                    # need to set the rows (geometry, name) before constant values,
                    # otherwise won't know how many rows to apply it to!
                    # now create the actual polygon buffer and assign the buffer height value
                    gdf_buffer_tmp['geometry'] = gdf_buffer_tmp['geometry'].buffer(
                        0.1 * poly_ID)

                    buffer_height = buffer_height_calc(
                        0.1 * poly_ID, buffer_radius_deg)
                    gdf_buffer_tmp['HEIGHT'] = buffer_height * \
                        gdf_buffer_tmp['swell_height']
                    gdf_buffer_tmp['buff_height'] = buffer_height

                    # append to buffer geodataframe
                    # gdf_buffer = gdf_buffer.append(gdf_buffer_tmp)
                    gdf_buffer = pd.concat([gdf_buffer, gdf_buffer_tmp])

            if gdf_buffer.empty is not True:
                # print('...... Saving buffer shapefile ')
                gdf_merged_LIPs_nooutlines = gdf_merged_LIPs[~gdf_merged_LIPs['NAME'].str.contains('outline')]
                # gdf_merged_LIPs = gdf_merged_LIPs_nooutlines.append(gdf_buffer)  # add buffer
                gdf_merged_LIPs = pd.concat([gdf_merged_LIPs, gdf_merged_LIPs_nooutlines]) # add buffer
                gdf_merged_LIPs.sort_values(by=['HEIGHT', 'NAME'], inplace=True, ignore_index=True, ascending=[True, True])
                gdf_merged_LIPs.reset_index(inplace=True)

                gdf_merged_LIPs = gdf_merged_LIPs.drop(['index'], axis=1)
                with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
                    gdf_merged_LIPs.to_file('%s/reconstructed_%s_%sMa.shp' % (path_rotated_polygons, feature_type, int(time)))
            else:
                # print('...... No buffer created, saving shapefile')
                gdf_merged_LIPs.sort_values(by=['HEIGHT', 'NAME'], inplace=True, ignore_index=True, ascending=[True, True])
                gdf_merged_LIPs.reset_index(inplace=True)
                with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
                    gdf_merged_LIPs.to_file('%s/reconstructed_%s_%sMa.shp' % (path_rotated_polygons, feature_type, int(time)))
        else:
            # Time is NOT 0 Ma.

            # print("...... Age is: " + str(time))
            # print("shape of %s Ma LIP file is: %s " % (str(time), gdf_merged_LIPs.shape))

            # --- create empty geodataframe for the buffer geometries
            gdf_buffer = gpd.GeoDataFrame()
            gdf_buffer['geometry'] = None
            gdf_buffer = gdf_buffer.set_crs(epsg=4326)

            # --- add swell stats to shapefile
            if not cooling_model == 'no_swell':

                # --- Import swell_stats text file and add a version without outline
                stats_outline = pd.read_csv(LIP_output_dir + '/%s/%s_swell_stats.txt' % (LIP_model_name, cooling_model), sep='\t')

                # copy and rename so that we don't have outlines in our names
                stats = stats_outline.copy()
                stats['NAME'] = stats['NAME'].str.replace('_outline', '')
                # stats = stats.append(stats_outline)
                stats = pd.concat([stats, stats_outline])

                # merge with LIP geometries
                gdf_merged_LIPs = gdf_merged_LIPs.merge(stats)

            elif cooling_model == "no_swell":
                gdf_merged_LIPs['swell_const'] = 0

            # calculate swell meta - swell_meta[0]: the swell height. constant for all height values
            # Calculating swell height and LIP height')

            # Pass the RHCW_age_depth file to function if it has been imported
            if RHCW_age_depth_interp is not None:
                gdf_merged_LIPs[['swell_height', 'HEIGHT']] = gdf_merged_LIPs.apply(lambda row: swell_calc(
                    float(time), row['FROMAGE'], row['LIPH'], cooling_model, row['swell_const'], RHCW_age_depth_interp), axis=1, result_type='expand')

            # if RHCW_age_depth_interp doesn't exist, we're probably not using the RHCW relationship
            else:
                gdf_merged_LIPs[['swell_height', 'HEIGHT']] = gdf_merged_LIPs.apply(lambda row: swell_calc(
                    float(time), row['FROMAGE'], row['LIPH'], cooling_model, row['swell_const']), axis=1, result_type='expand')
                # # swell_meta[1] - LIP height: will vary with LIPH

            print("shape of %s Ma LIP file is now: %s " % (str(time), gdf_merged_LIPs.shape))

            # ----
            # create buffer if desired
            if "buffer" in cooling_model:
                # use only the outlines of the LIPs
                gdf_merged_LIPs_outline = gdf_merged_LIPs[gdf_merged_LIPs['NAME'].str.contains(
                    'outline')]
     
                # print('... calculating buffer')
                # work out number of polygons to cover the buffer distance in increments of 0.1 degrees
                poly_no = int(buffer_radius_deg / 0.1)
                # iterate through each buffer polygon (starting closest to LIP)
                for j in range(poly_no):
                    poly_ID = j + 1
                    # make a copy of the outline file, where we can will overwrite geometry and height
                    gdf_buffer_tmp = gdf_merged_LIPs_outline.copy()

                    # save feature to new gpd
                    # rename 'polygon' to 'outline' in LIP name
                    name = gdf_buffer_tmp['NAME']
                    new_name = name.str.replace('outline', 'buffer')
                    # print(new_name)
                    gdf_buffer_tmp['NAME'] = new_name

                    # need to set the rows (geometry, name) before constant values,
                    # otherwise won't know how many rows to apply it to!
                    # now create the actual polygon buffer and assign the buffer height value
                    gdf_buffer_tmp['geometry'] = gdf_buffer_tmp['geometry'].buffer(
                        0.1 * poly_ID)

                    buffer_height = buffer_height_calc(
                        0.1 * poly_ID, buffer_radius_deg)
                    gdf_buffer_tmp['HEIGHT'] = buffer_height * \
                        gdf_buffer_tmp['swell_height']
                    gdf_buffer_tmp['buff_height'] = buffer_height

                    # append to buffer geodataframe
                    # gdf_buffer = gdf_buffer.append(gdf_buffer_tmp)
                    gdf_buffer = pd.concat([gdf_buffer, gdf_buffer_tmp])

            # ----
            # if the buffer exists, append to geodataframe before saving shapefile
            if gdf_buffer.empty is not True:
                print('... saving buffer shapefile ')
                gdf_merged_LIPs_nooutlines = gdf_merged_LIPs[~gdf_merged_LIPs['NAME'].str.contains('outline')]
                # gdf_merged_LIPs = gdf_merged_LIPs_nooutlines.append(gdf_buffer)  # add buffer
                gdf_merged_LIPs = pd.concat([gdf_merged_LIPs_nooutlines, gdf_buffer]) # add buffer... fixing for pandas 2.0, hope this works

                gdf_merged_LIPs.sort_values(by=['HEIGHT', 'NAME'], inplace=True, ignore_index=True, ascending=[True, True])
                gdf_merged_LIPs.reset_index(inplace=True)

                gdf_merged_LIPs = gdf_merged_LIPs.drop(['index'], axis=1)
                with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
                    gdf_merged_LIPs.to_file('%s/reconstructed_%s_%sMa.shp' % (path_rotated_polygons, feature_type, time))

            # save out shapefile
            else:
                print('... no buffer created, saving shapefile')
                print("shape of %s Ma LIP file without buffer is now: %s " % (str(time), gdf_merged_LIPs.shape))

                gdf_merged_LIPs.sort_values(by=['HEIGHT', 'NAME'], inplace=True, ignore_index=True, ascending=[True, True])
                gdf_merged_LIPs.reset_index(inplace=True)
                with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
                    gdf_merged_LIPs.to_file('%s/reconstructed_%s_%sMa.shp' % (path_rotated_polygons, feature_type, time))

# ------------------------------------------
# --- Part 3: create netcdf from shapefile

# Set of functions for shifting longitudes for polygons that cross the antimeridian
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


def modify_longitudes_gdf(gdf_LIPs, time):
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
    # print('... fixing longitudes for %s Ma' % time)
    # loop through each row
    for i in range(len(gdf_LIPs)):
        # print('%s of %s for %s Ma' % (i, len(gdf_LIPs), time))
        
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
                # ages.append(gdf_LIPs['FROMAGE'].loc[i])
                # plateids.append(gdf_LIPs['PLATEID1'].loc[i])

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
                    # ages.append(gdf_LIPs['FROMAGE'].loc[i])
                    # plateids.append(gdf_LIPs['PLATEID1'].loc[i])

            else:
                # geometry wasn't split, append to list
                geometries.append(split_geometry_neg.geoms[0])
                height.append(gdf_LIPs['HEIGHT'].loc[i])
                names.append(gdf_LIPs['NAME'].loc[i])
                # ages.append(gdf_LIPs['FROMAGE'].loc[i])
                # plateids.append(gdf_LIPs['PLATEID1'].loc[i])

    print('... making new gdf with fixed longitudes for %s Ma' % time)
    # make geodataframe
    gdf_LIPs_out = gpd.GeoDataFrame()
    gdf_LIPs_out['NAME'] = None
    gdf_LIPs_out['geometry'] = None
    gdf_LIPs_out = gdf_LIPs_out.set_crs(epsg=4326)

    # add things to it
    gdf_LIPs_out['NAME'] = names
    gdf_LIPs_out['HEIGHT'] = height
    gdf_LIPs_out['geometry'] = geometries
    # gdf_LIPs_out['FROMAGE'] = ages
    # gdf_LIPs_out['PLATEID1'] = plateids

    # DOES THIS SORTING MAKE SENSE? I don't know...
    gdf_LIPs_out.sort_values(by=['HEIGHT', 'NAME'], inplace=True, ignore_index=True, ascending=[True, True])
    

    return gdf_LIPs_out


def convert_shapefile_to_raster(
    gdf_LIPs,
    z_column,
    time,
    grid_spacing,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    feature_type,
    cooling_model,
    path_output_grids,
):
    """
    Convert LIP shapefile (as geodataframe) into a netcdf using rasterio/rasterize
    Defaults to 0.1° and -180/180/-90/90.
    Then pass the raster to xarray, so we can save the netcdf out nicely.

    """

    # see if basement_depth has previously been imported (probably not), and
    # make the output grid the same resolution/region.
    # Otherwise, default to 0.1° -180/180/-90/90

    try:
        basement_depth

    except NameError:
        # print('...... Defaulting to 0.1° global grid')
        lat_shape = int((180 / grid_spacing) + 1)
        lon_shape = int((360 / grid_spacing) + 1)


        output_shape = (lat_shape, lon_shape)

        output_transform = rasterio.Affine(
            grid_spacing, 0.0, lon_min - (grid_spacing/2), 0.0, grid_spacing, lat_min -(grid_spacing/2))
#        NOTE ref point is bottom left corner)
        lats = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
        lons = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)


        # # NOTE ref point is top left corner)
        # output_transform = rasterio.Affine(
        #     grid_spacing, 0.0, lon_min - (grid_spacing/2), 
        #     0.0, -grid_spacing, lat_max + (grid_spacing/2))
        # lats = np.arange(lat_max, lat_min - grid_spacing, -grid_spacing)
        # lons = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)


        # # print('...... Defaulting to 0.1° global grid')
        # output_shape = (1801, 3601)
        # output_transform = rasterio.Affine(
        #     0.1, 0.0, -180.05, 0.0, 0.1, -90.05)
        # # NOTE ref point is bottom left corner)
        # lats = np.arange(-90, 90.1, 0.1)
        # lons = np.arange(-180., 180.1, 0.1)

    else:
        # print('...... Copying resolution from basement depth netcdf')
        output_shape = basement_depth.shape
        output_transform = basement_depth.rio.transform()
        lats = basement_depth.y
        lons = basement_depth.x

    print('... creating %s raster for %s Ma' % (feature_type, time))
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

    # round lats and lons, to ensure we can safely add them
    # because xarray is annoying sometimes
    ds_lip_raster['lat'] = np.round(ds_lip_raster.lat.values, 1)
    ds_lip_raster['lon'] = np.round(ds_lip_raster.lon.values, 1)

    ds_lip_raster['z'].attrs = {
        'long_name': 'Height', 'units': 'm',
        'actual_range': np.array([np.nanmin(ds_lip_raster.z), np.nanmax(ds_lip_raster.z)], dtype=np.float32)}
    ds_lip_raster['lat'].attrs = {
        'long_name': "latitude", 'standard_name': "latitude", 'units': "degrees_north",
        'actual_range': np.array([np.nanmin(ds_lip_raster.lat), np.nanmax(ds_lip_raster.lat)], dtype=np.float32)}
    ds_lip_raster['lon'].attrs = {
        'long_name': "longitude", 'standard_name': "longitude", 'units': "degrees_east",
        'actual_range': np.array([np.nanmin(ds_lip_raster.lon), np.nanmax(ds_lip_raster.lon)], dtype=np.float32)}
        
    # global attributes
    if feature_type == 'LIPs':   
        ds_lip_raster.attrs['title'] = "Large igneous province (LIP) height at %s Ma using %s" % (
            time, cooling_model)
        ds_lip_raster.attrs['history'] = "created %s" % (
            datetime.now().strftime('%Y-%m-%d %H:%M'))
    else:
        ds_lip_raster.attrs['title'] = "Seamount height at %s Ma using %s" % (
            time, cooling_model)
        ds_lip_raster.attrs['history'] = "created %s" % (
            datetime.now().strftime('%Y-%m-%d %H:%M'))
    # compress so it doesn't take a ridiculous amount of space!
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_lip_raster.data_vars}

    ds_lip_raster.to_netcdf('%s/reconstructed_%s_%sMa.nc' % (path_output_grids, feature_type, time), encoding=encoding)

    # I don't think this is needed anymore.
    # # fix grids because they appear upside down in some things for some reason...
    # # I have NO clue why except it's because of gdal.
    # call_system_command(['gdal_translate', '-of', 'netCDF', '-co', 'WRITE_BOTTOMUP=NO', '-co', 'COMPRESS=DEFLATE', 
    #     '%s/reconstructed_%s_%sMa-tmp.nc' % (path_output_grids, feature_type, time), '%s/reconstructed_%s_%sMa.nc' % (path_output_grids, feature_type, time)])

    # remove tmp grid
    # call_system_command(['rm', '%s/reconstructed_%s_%sMa-tmp.nc' % (path_output_grids, feature_type, time)])
    
    print('... saved netcdf for %s Ma' % time)

# --- Check if the shapefile exists before trying to create the netcdf grid
# Don't do anything if the shapefile doesn't exist!
def mp_wrapper_for_shapefile_to_raster(
    time,
    path_rotated_polygons,
    feature_type,
    cooling_model,
    grid_spacing,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    path_output_grids,
):
    LIP_filename = '%s/reconstructed_%s_%sMa.shp' % (
        path_rotated_polygons, feature_type, time)

    print(LIP_filename)
    if os.path.isfile(LIP_filename):
        gdf_LIPs = gpd.read_file(LIP_filename)

        if "buffer" in cooling_model:
            print("... fixing longitudes in buffer")
            gdf_LIPs_fixed = modify_longitudes_gdf(gdf_LIPs, time)
        else:
            gdf_LIPs_fixed = gdf_LIPs
        convert_shapefile_to_raster(
            gdf_LIPs_fixed,
            'HEIGHT',
            time,
            grid_spacing,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            feature_type,
            cooling_model,
            path_output_grids,
        )
    else:
        print('... no LIP shapefile for %s Ma' % time)
        pass

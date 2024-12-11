# %% [markdown]
# Steps are:
# 
# - select a reconstruction time
# - the code determines which paleogeography stage this falls within, gets the start and end times
# - load the relevant precomputed multipoint files, and in the process assign an integer to the different types for use in interpolation steps (e.g. set land to be 1, shallow marine to be 2, etc)
# 
# - for land and marine
# 
# 

# %%
import glob
import os
import re
import sys

DIRNAME = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(DIRNAME, "..", "Paleogeography"))

import numpy as np
# import pygplates
from joblib import Parallel, delayed
# from scipy import interpolate

# import paleogeography as pg
# import paleogeography_tweening as pgt
from . import paleotopography as pt
# import polygon_processing as pp

# from proximity_query import *
# from create_gpml import create_gpml_regular_long_lat_mesh
# import points_in_polygons
# from sphere_tools import sampleOnSphere
# import points_spatial_tree

# gplates default model
# reconstruction_basedir = './palegeography-polygons-cookie-cut-by-default-static-polygons/'
# rotation_file = [f'{basedir}/Muller2019-Young2019-Cao2020_CombinedRotations.rot']

# Matthews model
# reconstruction_basedir = './Paleogeography_Matthews2016_410-2Ma_Shapefiles/'
reconstruction_basedir = os.path.join(DIRNAME, "Paleogeography_Matthews2016_410-2Ma_Shapefiles")
rotation_file = [f'{reconstruction_basedir}/Global_EB_250-0Ma_GK07_Matthews++.rot',
                                          f'{reconstruction_basedir}/Global_EB_410-250Ma_GK07_Matthews++.rot']


tween_basedir = './tween_feature_collections/'
file_format = 'gpmlz'

output_dir = './Paleotopography_Grids/'
os.makedirs(output_dir, exist_ok=True)

netcdf3_output = False


COBterrane_file = f'{reconstruction_basedir}/Global_EarthByte_GeeK07_COB_Terranes_Matthews_etal.gpml'

# Clennett et al. model
model_dir = "/Users/chris/OneDrive - The University of Sydney (Staff)/Dropbox/Work/EarthBytePlateMotionModel-ARCHIVE/Global_Model_WD_Internal_Release_2019_v2_Clennett_NE_Pacific"
rotation_file = f"{model_dir}/CombinedFiles/CombinedRotations.rot"
COBterrane_file = f"{model_dir}/StaticGeometries/AgeGridInput/CombinedTerranes.gpml"

#agegrid_file_template = '/Users/Simon/Data/AgeGrids/Agegrids_30m_20151002_2015_v1_r756/agegrid_30m_%d.grd'
# agegrid_file_template = '/Users/Simon/cloudstor/age_grids/2016_v5_r1031/raw/sphtmp_mask_%0.1fMa.nc'
agegrid_file_template = ""


#############################################
## Set the heights for different environment
#############################################
depth_for_unknown_ocean = -1000
# ----------------------------------
shallow_marine_elevation = -200.
# ----------------------------------
lowland_elevation = 200.
# ----------------------------------
# max_mountain_elevation = 1500.
max_mountain_elevation = 3100  # 3100 + 200 = 3300m
# NOTE - this height is actually the mountain height IN ADDITION TO the lowland height
# so that the maximum absolute elevation would be [lowland_elevation + max_mountain_elevation]
# TODO should call this 'mountain_relief'???
#############################################

# the grid sampling for the output
sampling = 1.0

# this number controls how small polygons are exclude when merging the COB terranes into 
# land/sea masking polygons
# area_threshold = 0.0001
area_threshold = 0.0

# used for quadtree
subdivision_depth = 2

# this buffer defines the smoothness of the topography at the transition from 'lowland' to 'mountain'
# the distance defined here is the distance over which heights ramp from the lowland elevation to the 
# mountain elevation defined above. (the ramping takes place from the edge of the mountain range inwards
# towards the mountain interior). Any parts of the mountain range greater than this buffer distance from 
# the edge will have a uniform height equal to max_mountain_elevation
mountain_buffer_distance_degrees = 0.001
#mountain_buffer_distance_degrees = 2.

# choose here either 'ocean' or 'land'
# this determines which grid takes precedence where both the age grid and the 
# paleogeographies overlap and contain valid values
land_or_ocean_precedence = 'land'

# this number is used in the final grdfilter step to smooth the output 
# NOTE this value is ignored if 'merge_with_bathymetry' is set to False
grid_smoothing_wavelength_kms = 400.

time_min = 0.
time_max = 170.
time_step = 1.

merge_with_bathymetry = False


####################################################

def main():

    # make a sorted list of the (midpoint) times for paleogeography polygons
    tmp = glob.glob(f'{reconstruction_basedir}/*/')

    paleogeography_timeslice_list = [float(re.findall(r'\d+Ma+',tm)[1][:-2]) for tm in tmp]
    # prepend 0 to the list, since this is not covered in published paleogeography sequence
    paleogeography_timeslice_list.append(0.0)
    paleogeography_timeslice_list.sort()
    paleogeography_timeslice_list = np.array(paleogeography_timeslice_list)
    print(paleogeography_timeslice_list)
    
    num_cpus = 8

    times = np.arange(time_min, time_max + time_step, time_step)
    # times = [15]

    if num_cpus==1:
        
        for reconstruction_time in times:
            if paleogeography_timeslice_list[-1]<=reconstruction_time:
                break #reconstruction time out of range
            pt.paleotopography_job(reconstruction_time, paleogeography_timeslice_list, 
                                tween_basedir, reconstruction_basedir, output_dir, 
                                file_format, rotation_file, COBterrane_file, agegrid_file_template,
                                lowland_elevation, shallow_marine_elevation, max_mountain_elevation, depth_for_unknown_ocean, 
                                sampling, mountain_buffer_distance_degrees, area_threshold,
                                grid_smoothing_wavelength_kms, merge_with_bathymetry, land_or_ocean_precedence,
                                netcdf3_output)

    else:
        Parallel(n_jobs=num_cpus, verbose=20)(delayed(pt.paleotopography_job) \
                                (reconstruction_time, paleogeography_timeslice_list, 
                                tween_basedir, reconstruction_basedir, output_dir, 
                                file_format, rotation_file, COBterrane_file, agegrid_file_template,
                                lowland_elevation, shallow_marine_elevation, max_mountain_elevation, depth_for_unknown_ocean, 
                                sampling, mountain_buffer_distance_degrees, area_threshold,
                                grid_smoothing_wavelength_kms, merge_with_bathymetry, land_or_ocean_precedence,
                                netcdf3_output)
                                for reconstruction_time in times)
                              
                              
if __name__ == "__main__":
    main()

"""From create_LIP_conjugates.ipynb"""
import os
import warnings

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import ptt
import pygplates
import shapely


def create_geodataframe_LIPS(pygplates_recon_geom, reconstruction_time):
    """ This function converts the LIP feature from pygplates into a GeoDataFrame.
    This makes it easier to apply various shapely-based transformations and helps to 
    avoid plotting artefacts due to the dateline.
    
    Input: 
        - pygplates.ReconstructedFeatureGeometry (i.e., output of pygplates.reconstruct)
        - recontruction time - this is just for safekeeping in the geodataframe!
    Output: 
        - gpd.GeoDataFrame of the LIP, with some of the shapefile attributes preserved. """
    
    
    # create new and empy geodataframe
    recon_gpd = gpd.GeoDataFrame()
    recon_gpd['NAME'] = None
    recon_gpd['PLATEID1'] = None
    recon_gpd['PLATEID2'] = None
    recon_gpd['FROMAGE'] = None
    recon_gpd['TOAGE'] = None
    recon_gpd['geometry'] = None
    recon_gpd['reconstruction_time'] = None
    recon_gpd['LIPH'] = None
    recon_gpd = recon_gpd.set_crs(epsg=4326)
    
    date_line_wrapper = pygplates.DateLineWrapper()

    names                = []
    plateid1s            = []
    plateid2s            = []
    fromages             = []
    toages               = []
    geometrys            = []
    reconstruction_times = []
    LIP_height           = []
    lrg_elev             = []
    contour_elev         = []
    
    for i, seg in enumerate(pygplates_recon_geom):
        
        wrapped_polygons = date_line_wrapper.wrap(seg.get_reconstructed_geometry())
        for poly in wrapped_polygons:
            ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
            ring[:,1] = np.clip(ring[:,1], -89, 89) # anything approaching the poles creates artefacts
                    
            name = seg.get_feature().get_name()
            plateid = seg.get_feature().get_reconstruction_plate_id()
            conjid = seg.get_feature().get_conjugate_plate_id()
            from_age, to_age = seg.get_feature().get_valid_time()
                  
            # get LIP specific attributes
            liph = seg.get_feature().get_shapefile_attribute('LIPH')
            lrg_el = seg.get_feature().get_shapefile_attribute('LRG_ELEV')
            con_elev = seg.get_feature().get_shapefile_attribute('CON_ELEV')
            
            # append things
            names.append(name)
            plateid1s.append(plateid)
            plateid2s.append(conjid)
            fromages.append(from_age)
            toages.append(to_age)
            geometrys.append(shapely.geometry.Polygon(ring)) 
            reconstruction_times.append(reconstruction_time)
            LIP_height.append(liph)
            lrg_elev.append(lrg_el)     
            contour_elev.append(con_elev) 

    # write to geodataframe
    recon_gpd['NAME'] = names
    recon_gpd['PLATEID1'] = plateid1s
    recon_gpd['PLATEID2'] = plateid2s
    recon_gpd['FROMAGE'] = fromages
    recon_gpd['TOAGE'] = toages
    recon_gpd['geometry'] = geometrys
    recon_gpd['reconstruction_time'] = reconstruction_times
    recon_gpd['LIPH'] = LIP_height
    recon_gpd['LRG_ELEV'] = lrg_elev
    recon_gpd['CON_ELEV'] = contour_elev
    
    return recon_gpd


def create_geodataframe_general(pygplates_recon_geom, reconstruction_time):
    """ This is a general function to convert reconstructed features (e.g. 
    reconstructed coastlines) from pygplates into a GeoDataFrame. This helps
    avoid plotting artefacts. Note that the input geometry must be a polygon.
    
    Input: 
        - pygplates.ReconstructedFeatureGeometry (i.e., output of pygplates.reconstruct)
        - recontruction time - this is just for safekeeping in the geodataframe!
    Output: 
        - gpd.GeoDataFrame of the feature"""
    
    # create new and empy geodataframe
    recon_gpd = gpd.GeoDataFrame()
    recon_gpd['NAME'] = None
    recon_gpd['PLATEID1'] = None
    recon_gpd['PLATEID2'] = None
    recon_gpd['FROMAGE'] = None
    recon_gpd['TOAGE'] = None
    recon_gpd['geometry'] = None
    recon_gpd['reconstruction_time'] = None
    recon_gpd = recon_gpd.set_crs(epsg=4326)
    
    date_line_wrapper = pygplates.DateLineWrapper()

    names                = []
    plateid1s            = []
    plateid2s            = []
    fromages             = []
    toages               = []
    geometrys            = []
    reconstruction_times = []
    
    for i, seg in enumerate(pygplates_recon_geom):
        
        wrapped_polygons = date_line_wrapper.wrap(seg.get_reconstructed_geometry())
        for poly in wrapped_polygons:
            ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
            ring[:,1] = np.clip(ring[:,1], -89, 89) # anything approaching the poles creates artefacts
                    
            name = seg.get_feature().get_name()
            plateid = seg.get_feature().get_reconstruction_plate_id()
            conjid = seg.get_feature().get_conjugate_plate_id()
            from_age, to_age = seg.get_feature().get_valid_time()     
            # append things
            names.append(name)
            plateid1s.append(plateid)
            plateid2s.append(conjid)
            fromages.append(from_age)
            toages.append(to_age)
            geometrys.append(shapely.geometry.Polygon(ring)) 
            reconstruction_times.append(reconstruction_time)

    # write to geodataframe
    recon_gpd['NAME'] = names
    recon_gpd['PLATEID1'] = plateid1s
    recon_gpd['PLATEID2'] = plateid2s
    recon_gpd['FROMAGE'] = fromages
    recon_gpd['TOAGE'] = toages
    recon_gpd['geometry'] = geometrys
    recon_gpd['reconstruction_time'] = reconstruction_times
    
    return recon_gpd


def create_geodataframe_topologies(topologies, reconstruction_time):
    """ This is a function to convert topologies from pygplates into a GeoDataFrame
    This helps select the closed topological plates ('gpml:TopologicalClosedPlateBoundary',
    and also helps resolve plotting artefacts from crossing the dateline. 
    This function does NOT incorporate various plate boundary types into the geodataframe!
    
    Input: 
        - pygplates.Feature. This is designed for `topologies`, which comes from:
              resolved_topologies = ptt.resolve_topologies.resolve_topologies_into_features(
                                        rotation_model, topology_features, reconstruction_time)
              topologies, ridge_transforms, ridges, transforms, trenches, trench_left, trench_right, other = resolved_topologies
        - recontruction time - this is just for safekeeping in the geodataframe!
    Output: 
        - gpd.GeoDataFrame of the feature"""
    
    # function for getting closed topologies only
    # i.e., the plates themselves, NOT all the features for plotting!
    
    # set up the empty geodataframe
    recon_gpd = gpd.GeoDataFrame()
    recon_gpd['NAME'] = None
    recon_gpd['PLATEID1'] = None
    recon_gpd['PLATEID2'] = None
    recon_gpd['FROMAGE'] = None
    recon_gpd['TOAGE'] = None
    recon_gpd['geometry'] = None
    recon_gpd['reconstruction_time'] = None
    recon_gpd['gpml_type'] = None
    recon_gpd = recon_gpd.set_crs(epsg=4326)

    # some empty things to write stuff to
    names                = []
    plateid1s            = []
    plateid2s            = []
    fromages             = []
    toages               = []
    geometrys            = []
    reconstruction_times = []
    gpml_types           = []
    
    # a dateline wrapper! so that they plot nicely and do nice things in geopandas
    date_line_wrapper = pygplates.DateLineWrapper()
    
    for i, seg in enumerate(topologies):
        gpmltype = seg.get_feature_type()
        
        # polygon and wrap
        polygon = seg.get_geometry()
        wrapped_polygons = date_line_wrapper.wrap(polygon)
        for poly in wrapped_polygons:
            ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
            ring[:,1] = np.clip(ring[:,1], -89, 89) # anything approaching the poles creates artefacts
            for wrapped_point in poly.get_exterior_points():
                wrapped_point_lat_lon = wrapped_point.get_latitude(), wrapped_point.get_longitude()
            
            # might result in two polys - append to loop here (otherwise we will be missing half the pacific etc)
            name = seg.get_name()
            plateid = seg.get_reconstruction_plate_id()
            conjid = seg.get_conjugate_plate_id()
            from_age, to_age = seg.get_valid_time()
            
            names.append(name)
            plateid1s.append(plateid)
            plateid2s.append(conjid)
            fromages.append(from_age)
            toages.append(to_age)
            geometrys.append(shapely.geometry.Polygon(ring)) 
            reconstruction_times.append(reconstruction_time)
            gpml_types.append(str(gpmltype))
    
    # write to geodataframe
    recon_gpd['NAME'] = names
    recon_gpd['PLATEID1'] = plateid1s
    recon_gpd['PLATEID2'] = plateid2s
    recon_gpd['FROMAGE'] = fromages
    recon_gpd['TOAGE'] = toages
    recon_gpd['geometry'] = geometrys
    recon_gpd['reconstruction_time'] = reconstruction_times
    recon_gpd['gpml_type'] = gpml_types
    
    return recon_gpd


def create_conjugate_lip(
    original_lip_gpd,
    lip_name,
    plate_topologies,
    x_flip,
    y_flip,
    rotation,
    x_off,
    y_off,
    to_age,
    LIP_conjugate_outdir,
    rotation_model,
):
    """ Function to manually create a conjugate LIP.  This function uses various 
    geopandas/shapely transformations to create a suitable conjugate polygon. It 
    then updates the conjugate feature's plateID (PLATEID1) to match the topological
    plate that it is mostly lying upon. There is also an option to 'subduct' the 
    feature, by manually updating its TOAGE.
    
    The function will also 'reverse reconstruct' the conjugate LIP, so that it 
    will correctly reconstruct within GPlates. This is needed because the geometry 
    is NOT created at present-day.
    
    Inputs: 
        - `original_lip_gpd`: GeoDataFrame of the original LIP
        - `lip_name`: name of the LIP of interest. It must match the spelling exactly
        - `plate_topologies`: closed plate topologies as a geodataframe. i.e. from  
              gpd_topologies = create_geodataframe_topologies(topologies, reconstruction_time)
              gpd_topologies_plates = gpd_topologies[(gpd_topologies['gpml_type'] == 'gpml:TopologicalClosedPlateBoundary')]
        - Parameters to modify position of conjugate LIP:
                - `x_flip`, `y_flip`: these parameters describe which axis to 
                   *mirror* upon (i.e. `y_flip = -1` will mirror along the y-axis).
                   Only one of these should be set to -1
                - `rotation`: how much to rotate the polygon by so it is along
                   the desired plate boundary etc.
                - `x_off`, `y_off`: how much to offset the polygon along 
                   the x- and y- axes. 
                - `to_age`: *TOAGE* of the conjugate LIP, so it will 
                   (pseudo)subduct and not reappear in a different ocean
    Output:
        - Shapefile of the individual conjugate LIP, saved in the 
          LIP_conjugate_outdir directory
    """    
  
    # --- get specific LIP out using name
    # single_lip = original_lip_gpd[original_lip_gpd['NAME'] == lip_name]
    
    # use contains so that we also get the outline out (needed for the buffer step)
    single_lip = original_lip_gpd[original_lip_gpd['NAME'].str.contains(lip_name)]
    # print(single_lip)
    # --- Transform geometry so that the conjugate LIP is where we want it
    # unary_union.centroid --> to flip around a common centroid point
    # otherwise it won't rotate together
    
    # TO DO: change crs so area is preserved? Need to think about which is best....
    flipped_lip = single_lip.scale(xfact=x_flip, yfact=y_flip, origin=single_lip.unary_union.centroid)
    rotated_lip = flipped_lip.rotate(rotation, origin=single_lip.unary_union.centroid)
    shifted_lip = rotated_lip.translate(xoff=x_off, yoff=y_off)
    
    # output lip
    conj_lip = single_lip.drop(columns=['geometry'])
    
    # update geometry
    conj_lip['geometry'] = shifted_lip
    conj_lip = conj_lip.set_geometry('geometry', crs=original_lip_gpd.crs)

    # --- Determine new plateID
    # combine LIP into single polygon
    conj_lip_dissolved = conj_lip.dissolve()
    
    # overlap plates ontop of LIPs
    conj_lips_overlay = gpd.overlay(conj_lip_dissolved, plate_topologies, how = 'intersection')
    conj_lips_overlay['area'] = conj_lips_overlay.geometry.area
    
    # sort by area (largest first) and reset index
    conj_lips_sorted = conj_lips_overlay.sort_values('area', ascending=False, ignore_index=True)
    # dissolve boundaries - will keep attributes for first row
    conj_lips_sorted_dissolved = conj_lips_sorted.dissolve()
    
    # for saving the output..
    conj_lip_out = conj_lip
    # update PLATEID1 to be the plateID that the conjugate LIP is mostly on (based on area)
    conj_lip_out['PLATEID1'] = conj_lips_sorted_dissolved.PLATEID1_2.values[0]
        
    # --- Update TOAGE so that the LIP subducts
    conj_lip_out['TOAGE'] = to_age
    
    # --- Save shapefile
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        conj_lip_out.to_file('%s/%s.shp' % (LIP_conjugate_outdir, conj_lip_out['NAME'].values[0]))
    
    # --- Reverse reconstruct saved shapefile
    # This is so GPlates knows the geometry is from FROMAGE, NOT present day.
    pygplates.reverse_reconstruct('%s/%s.shp' % (LIP_conjugate_outdir, conj_lip_out['NAME'].values[0]), rotation_model, conj_lip_out.FROMAGE.values[0])
    
    return conj_lip_out


def reconstruct_to_LIP_emplacement_and_create_conj(
    lip_name,
    reconstruction_time,
    LIP_contour_poly,
    rotation_model,
    topology_features,
    LIP_conjugate_outdir,
    x_flip=1,
    y_flip=1,
    rotation=0,
    x_off=0,
    y_off=0,
    to_age=0,
):
    """ Function to reconstruct the LIP and create the conjugate LIP.
    
    This function requires the following functions:
        - create_geodataframe_LIPS
        - create_geodataframe_topologies
        - create_conjugate_lip
    
    NOTE: if create_conjugate_lip parameters are not set (x_flip, y_flip, rotation,
    x_off, y_off, to_age), default values will be used, which will just create an
    identifcal LIP polygon.
    
    This function will also reconstruct the LIP using pygplates, and resolve topologies
    using plate tectonic tools (ptt).
    
    """
    
    # lip_name = 
    # reconstructed LIPs to desired time
    reconstructed_LIPs = []
    pygplates.reconstruct(LIP_contour_poly, rotation_model, reconstructed_LIPs, reconstruction_time)
    
    # ---
    # use plate tectonic tools to get topologies
    resolved_topologies = ptt.resolve_topologies.resolve_topologies_into_features(
        rotation_model, topology_features, reconstruction_time)
    
    # all the topologies at X time
    topologies, ridge_transforms, ridges, transforms, trenches, trench_left, trench_right, other = resolved_topologies
    
    # create geodataframes
    gpd_LIPs = create_geodataframe_LIPS(reconstructed_LIPs, reconstruction_time)
    
    gpd_topologies = create_geodataframe_topologies(topologies, reconstruction_time)
    gpd_topologies_plates = gpd_topologies[(gpd_topologies['gpml_type'] == 'gpml:TopologicalClosedPlateBoundary')]
    
    # get LIPs that were only erupated at the reconstruction time
    gpd_LIPs_emplaced = gpd_LIPs[gpd_LIPs['FROMAGE'] == reconstruction_time]
    
    # create (and save) conjugate LIP using other definition
    gpd_conj = create_conjugate_lip(
        gpd_LIPs_emplaced,
        lip_name,
        gpd_topologies_plates,
        x_flip,
        y_flip,
        rotation,
        x_off,
        y_off,
        to_age,
        LIP_conjugate_outdir,
        rotation_model,
    )

    return gpd_LIPs_emplaced, gpd_conj, gpd_topologies_plates


def create_lip_conjugates(
    LIP_contour_poly,
    rotation_model,
    topology_features,
    LIP_conjugate_outdir,
):
    lips = [
        {
            "lip_name": "Southern_Kerguelen_31",
            "reconstruction_time": 110,
            "x_flip": 1,
            "y_flip": -1,
            "rotation": -50,
            "x_off": 5,
            "y_off": 11,
        },
        {
            "lip_name": "Central_Kerguelen_29",
            "reconstruction_time": 100,
            "x_flip": 1,
            "y_flip": -1,
            "rotation": -35,
            "x_off": 2,
            "y_off": 6,
        },
        {
            "lip_name": "Roo_Rise_51",
            "reconstruction_time": 136,
            "x_flip": 1,
            "y_flip": -1,
            "rotation": 20,
            "x_off": 0,
            "y_off": 1,
            "to_age": 80.,
        },
        {
            "lip_name": "Shatsky_Rise-Tamu_68",
            "reconstruction_time": 144.4,
            "x_flip": -1,
            "y_flip": 1,
            "rotation": 70,
            "x_off": 2,
            "y_off": 1,
            "to_age": 60.,
        },
        {
            "lip_name": "Shatsky_Rise-Ori_69",
            "reconstruction_time": 134.0,
            "x_flip": -1,
            "y_flip": 1,
            "rotation": 50,
            "x_off": 5,
            "y_off": 2,
            "to_age": 60.,
        },
        {
            "lip_name": "Shatsky_Rise-Shirshov_70",
            "reconstruction_time": 128.0,
            "x_flip": -1,
            "y_flip": 1,
            "rotation": 50,
            "x_off": 5,
            "y_off": 2,
            "to_age": 60.,
        },
        {
            "lip_name": "Hess_Rise_24",
            "reconstruction_time": 110.0,
            "x_flip": -1,
            "y_flip": 1,
            "rotation": 50,
            "x_off": 5,
            "y_off": 2,
            "to_age": 50.,
        },
        {
            "lip_name": "ManihikiI_35",
            "reconstruction_time": 118.0,
            "x_flip": -1,
            "y_flip": 1,
            "rotation": -30,
            "x_off": 10.5,
            "y_off": -2,
            "to_age": 15.,
        },
        {
            "lip_name": "Ontong_Java_Plateau_48",
            "reconstruction_time": 122.0,
            "x_flip": 1,
            "y_flip": -1,
            "rotation": 25,
            "x_off": 3,
            "y_off": -16,
            "to_age": 88,
        },
    ]
    for kwargs in lips:
        reconstruct_to_LIP_emplacement_and_create_conj(
            LIP_contour_poly=LIP_contour_poly,
            rotation_model=rotation_model,
            topology_features=topology_features,
            LIP_conjugate_outdir=LIP_conjugate_outdir,
            **kwargs
        )
    print('... Saving merged conjugate shapefile to %s:' % LIP_conjugate_outdir)

    # delete the merged file first if it already exists
    if os.path.isfile('%s/LIP_conjugates_0Ma.shp' % LIP_conjugate_outdir):
        os.remove('%s/LIP_conjugates_0Ma.shp' % LIP_conjugate_outdir)
        
        
    # get path for all the conjuate LIP shapefiles
    conj_LIP_shape_path = [os.path.join(LIP_conjugate_outdir, i) for i in os.listdir(LIP_conjugate_outdir) if i.endswith(".shp")]

    # create merged geodataframe
    gdf_conjLIP_shapes = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in conj_LIP_shape_path], ignore_index=True), 
                                        crs=gpd.read_file(conj_LIP_shape_path[0]).crs)

    gdf_conjLIP_shapes['conjugate'] = 'Yes'  # added as column for now.... 
    # save shapefile
    gdf_conjLIP_shapes.to_file('%s/LIP_conjugates_0Ma.shp' % LIP_conjugate_outdir)

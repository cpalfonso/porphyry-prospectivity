import os

import pygplates

_VALID_EXTENSIONS = (
    ".gpml",
    ".gpmlz",
    ".gpml.gz",
    ".dat",
    ".pla",
    ".rot",
    ".grot",
    ".shp",
    ".geojson",
    ".json",
    ".gpkg",
    ".gmt",
    ".vgp",
)


def load_paleogeography(pg_dir, env_list, env_field="ENV", time=None):
    if time is not None:
        time = float(time)
    out = []
    if os.path.isdir(pg_dir):
        filenames = [os.path.join(pg_dir, i) for i in os.listdir(pg_dir)]
    elif os.path.isfile(pg_dir):
        filenames = [pg_dir]
    else:
        raise FileNotFoundError("File/directory not found: {}".format(pg_dir))
    filenames = [
         i for i in filenames
         if str(i).endswith(_VALID_EXTENSIONS)
    ]
    features = pygplates.FeaturesFunctionArgument(filenames).get_features()

    for feature in features:
        if time is not None and (not feature.is_valid_at_time(time)):
                continue
        env = feature.get_shapefile_attribute(env_field)
        if env is not None and env in env_list:
            feature.set_shapefile_attribute("Layer", env)
            out.append(feature)
    return pygplates.FeatureCollection(out)

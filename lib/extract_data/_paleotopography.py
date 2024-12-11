import requests
import zipfile
from pathlib import Path

import pygplates
from gplately.grids import rasterise

PALEOTOPO_BASENAME = "bg-14-5425-2017-supplement.zip"
PALEOTOPO_URL = "https://bg.copernicus.org/articles/14/5425/2017/bg-14-5425-2017-supplement.zip"


def download_polygons(
    dest=PALEOTOPO_BASENAME,
    url=PALEOTOPO_URL,
    chunk_size=1024000,
    verbose=False,
):
    if verbose:
        try:
            from tqdm import tqdm
        except ImportError:
            verbose = False

    if chunk_size is None:
        chunk_size = 1024000
    path = Path(dest)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with path.open(mode="wb") as f:
            it = r.iter_content(chunk_size=chunk_size)
            if verbose:
                it = tqdm(it)
            for chunk in it:
                f.write(chunk)
    return path


def extract_zipfile(filename=PALEOTOPO_BASENAME, dest=None):
    if dest is None:
        dest = Path(Path(filename).parent, "bg-14-5425-2017-supplement")
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(filename).extractall(path=dest)
    return dest


def download_extract(
    filename=PALEOTOPO_BASENAME,
    dest=None,
    url=PALEOTOPO_URL,
    chunk_size=1024000,
    verbose=False,
):
    filename = download_polygons(
        dest=filename,
        url=url,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    return extract_zipfile(filename=filename, dest=dest)

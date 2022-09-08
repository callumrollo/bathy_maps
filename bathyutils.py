import xarray as xr
import numpy as np
from pathlib import Path
import os


def gebco_subset(path_to_folder_str, extent, bathy_nc=False):
    """
    Extracts bathy data from a global GEBCO .nc file from an area specified by the use
    :param path_to_folder: string of path to the folder, specified by user
    :param extent: list with four items which are extent of desired geotiff [South, North, West, East]
    e.g. [49. 50.5, -5, 2] (if using gebco or emodnet)
    :return: numpy arrays of lon, lat and bathymetry
    """
    extent = list(extent)
    if extent[2]>180:
        extent[2] = extent[2] - 360
    if extent[3]>180:
        extent[3] = extent[3] - 360
    print('Fetching GEBCO data...')
    path_to_folder = Path(path_to_folder_str)
    if path_to_folder.is_file():
        gebco = xr.open_dataset(path_to_folder)
    else:
        path_to_gebco = list(Path(path_to_folder).joinpath().glob("*.nc"))
        if not path_to_gebco:
            print('No netcdf files found in location supplied. Check that you pointed to a .nc file or a folder containing one. Aborting')
            exit(1)
        gebco = xr.open_dataset(path_to_gebco[0])
    print("Subsettting GEBCO data")
    subset = gebco.sel(lon=slice(extent[2], extent[3]), lat=slice(extent[0], extent[1]))
    "print GEBCO bathy fetch successful"
    if bathy_nc==True:
        ## To save our bathymetry data
        subset.to_netcdf(Path(os.getcwd())/'bathy_subset.nc')
        print('bathy subset written at ' + str(Path(os.getcwd())/'bathy_subset.nc'))
    return np.array(subset.lon), np.array(subset.lat), np.array(subset.elevation)



def patch_row(tiles, s_lim, n_lim, w_lim, e_lim):
    base_tile = xr.open_dataset(tiles[0])
    ds_sub = base_tile.sel(lon=slice(w_lim, e_lim), lat=slice(s_lim, n_lim))
    for next_tile_path in tiles[1:]:
        next_tile = xr.open_dataset(next_tile_path)
        wlim_no_overlap = np.max((w_lim, ds_sub.lon.max()))
        next_tile_sub = next_tile.sel(lon=slice(wlim_no_overlap, e_lim), lat=slice(s_lim, n_lim))
        ds_sub = xr.concat((ds_sub, next_tile_sub), dim="lon")
    return ds_sub


def patch_col(tiles, s_lim, n_lim, w_lim, e_lim):
    base_tile = xr.open_dataset(tiles[0])
    ds_sub = base_tile.sel(lon=slice(w_lim, e_lim), lat=slice(s_lim, n_lim))
    for next_tile_path in tiles[1:]:
        next_tile = xr.open_dataset(next_tile_path)
        n_lim_no_overlap = np.min((n_lim, ds_sub.lat.min()))
        next_tile_sub = next_tile.sel(lon=slice(w_lim, e_lim), lat=slice(s_lim, n_lim_no_overlap))
        ds_sub = xr.concat((next_tile_sub, ds_sub), dim="lat")
    return ds_sub


def emod_subset(extent, path_to_emodnet, buffer=0.2):
    """
    Selects the EMODnet bathymetry tiles that cover the user defined area and
    combines them for a seamless bathymetry. Returns lon, lat and bathy.
    :param extent: list with four items which are extent of desired geotiff [South, North, West, East]
    e.g. [49. 50.5, -5, 2] (if using gebco or emodnet)
    :param path_to_emodnet: string or path to the folder containing EMODnet .nc files, specified by user
    :param buffer: size in degrees of extra area around specified coordinates to return. Detaults to 0.2.
    :return: xarray object of subset bathymetry
    """
    path_to_emodnet = Path(path_to_emodnet)
    if not path_to_emodnet.is_dir():
        raise ValueError(f"Supplied directory {path_to_emodnet} does not exist")

    tiles_paths = list(Path(path_to_emodnet).glob("*.nc"))
    print(f"Found tiles {tiles_paths}")
    relevant_tiles = []
    s_lim = extent[0] - buffer
    n_lim = extent[1] + buffer
    w_lim = extent[2] - buffer
    e_lim = extent[3] + buffer
    for tile in tiles_paths:
        ds = xr.open_dataset(tile)
        x1 = max(w_lim, ds.lon.min())
        y1 = max(s_lim, ds.lat.min())
        x2 = min(e_lim, ds.lon.max())
        y2 = min(n_lim, ds.lat.max())
        if x1 < x2 and y1 < y2:
            relevant_tiles.append(tile)
    num_tiles = len(relevant_tiles)
    if num_tiles == 0:
        raise ValueError("No relevant tiles found. Check that your requested area is within the EMODnet tiles you have")
    print(f"Found {num_tiles} tiles with relevant data: {relevant_tiles}")
    if num_tiles == 1:
        sub_tile = ds.sel(lon=slice(w_lim, e_lim), lat=slice(s_lim, n_lim))
        return sub_tile
    relevant_tiles.sort()
    tile_names = [x.name for x in relevant_tiles]
    row_letters = [name[0] for name in tile_names]
    col_nums = [name[1] for name in tile_names]
    if len(np.unique(row_letters)) == 1:
        print("data in one row, patching lon")
        return patch_row(relevant_tiles, s_lim, n_lim, w_lim, e_lim)
    if len(np.unique(col_nums)) == 1:
        print("data in one column, patching lat")
        return patch_col(relevant_tiles, s_lim, n_lim, w_lim, e_lim)
    
    print("Performing 2D patch")
    ds_rows = []
    for row_letter in np.unique(row_letters):
        row_tiles = [path_to_emodnet / name for name in tile_names if name[0] == row_letter]
        ds_rows.append(patch_row(row_tiles, s_lim, n_lim, w_lim, e_lim))
    ds_sub = ds_rows[0]
    for next_tile in ds_rows[1:]:
        n_lim_no_overlap = np.min((n_lim, ds_sub.lat.min()))
        next_tile_sub = next_tile.sel(lon=slice(w_lim, e_lim), lat=slice(s_lim, n_lim_no_overlap))
        ds_sub = xr.concat((next_tile_sub, ds_sub), dim="lat")
    return ds_sub

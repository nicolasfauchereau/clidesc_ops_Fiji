import os
import pathlib
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pyproj

import argparse

"""
parse the command line arguments
"""

parser = argparse.ArgumentParser()

parser.add_argument('-d','--dpath', dest='dpath', type=str, default='/Users/nicolasf/data/TRMM/daily', \
help='the path to the TRMM folder, default is /Users/nicolasf/data/TRMM/daily')

parser.add_argument('-o','--opath', dest='opath', type=str, default='/Users/nicolasf/operational/clidesc_ops_Fiji/outputs', \
help='the path where to save txt files containing the interpolated TRMM data (on Fiji DEM)')

#parser.add_argument('-n','--ncpath', dest='nc_path', type=str, default='/Users/nicolasf/operational/clidesc_ops_Fiji/outputs', \
#help='the path where to save the netcdf files containing interpolated values')

vargs = vars(parser.parse_args())

# pop out the arguments

dpath = pathlib.Path(vargs['dpath'])
opath = pathlib.Path(vargs['opath'])
#nc_path = pathlib.Path(vargs['nc_path'])

# define the coordinates for the Fiji DEM
x  = np.arange(1797500, 1797500 + 1056 * 500, 500)
y =  np.arange(3551420, 3551420 + 1900 * 500, 500)

# define the projection string for the Fiji DEM

proj_4_string = '+proj=tmerc +lat_0=-17 +lon_0=178.75 +k=0.99985 +x_0=2000000 +y_0=4000000 +ellps=WGS72 +towgs84=0,0,4.5,0,0,0.554,0.2263 +units=m +no_defs'

# casts into 2D arrays
xarr, yarr = np.meshgrid(x, y)

p = pyproj.Proj(proj_4_string)


# calculates coordinates in Lat and Lon
lons = np.empty_like(xarr).astype(np.float64)
lats = np.empty_like(yarr).astype(np.float64)

for ilat in range(yarr.shape[0]):
    for ilon in range(yarr.shape[1]):
        xp = xarr[ilat, ilon]
        yp = yarr[ilat, ilon]
        lon, lat = p(xp, yp, inverse=True)
        if lon < 0:
            lon += 360
        lons[ilat, ilon] = lon
        lats[ilat, ilon] = lat

# get the list of TRMM files

lfiles_trmm = list(dpath.glob("3B42RT_daily.????.??.??.nc"))

lfiles_trmm.sort()

# get the last file

fname = lfiles_trmm[-1]

print(fname)

# get the date for the file

datestr = str(fname).split('.')[1:-1]

datestr = map(int, datestr)

date = datetime(*datestr)

# open the file

trmm = xr.open_dataset(fname)

trmm = trmm.squeeze()

# restrict to the Fiji domain

trmm = trmm.sel(lon=slice(lons.flatten().min(), lons.flatten().max()), lat=slice(lats.flatten().min(), lats.flatten().max()))

lats_arr = xr.DataArray(lats, dims=['y','x'], coords={'y': y, 'x' : x})

lons_arr = xr.DataArray(lons, dims=['y','x'], coords={'y': y, 'x' : x})

# Interpolate (Nearest Neighbors and Linear Interpolation)

trmm_interp_nn = trmm.interp(lon=lons_arr, lat=lats_arr, method='nearest')

trmm_interp_lin = trmm.interp(lon=lons_arr, lat=lats_arr, method='linear')

# saves to netcdf

#trmm_interp_nn.to_netcdf(nc_path / 'trmm_{:%Y%m%d}_interp_NN_Fiji_DEM.nc'.format(date))

#trmm_interp_lin.to_netcdf(nc_path / 'trmm_{:%Y%m%d}_interp_Linear_Fiji_DEM.nc'.format(date))

trmm_interp_nn_data = trmm_interp_nn['trmm'].data

trmm_interp_nn_data[np.isnan(trmm_interp_nn_data)] = -9999.

trmm_interp_lin_data = trmm_interp_lin['trmm'].data

trmm_interp_lin_data[np.isnan(trmm_interp_lin_data)] = -9999.

header="""ncols = 1056
nrows = 1900
xllcorner = 1797500
yllcorner = 3551420
cellsize = 500
NODATA_value = -9999."""

#np.savetxt(opath / 'trmm_{:%Y%m%d}_interp_NN_Fiji_DEM.txt'.format(date), trmm_interp_nn_data, header=header, comments='')

#np.savetxt(opath / 'trmm_{:%Y%m%d}_interp_Linear_Fiji_DEM.txt'.format(date), trmm_interp_lin_data, header=header, comments='')

# ### saves in non-scientific format

np.savetxt(opath / 'trmm_{:%Y%m%d}_interp_NN_Fiji_DEM_fmt.txt'.format(date), trmm_interp_nn_data, header=header, comments='', fmt='%8.4f')

np.savetxt(opath / 'trmm_{:%Y%m%d}_interp_Linear_Fiji_DEM_fmt.txt'.format(date), trmm_interp_lin_data, header=header, comments='', fmt='%8.4f')

# cleanup 

now = datetime.now()

lfiles_txt = list(opath.glob("*.txt"))

for f in lfiles_txt: 
    modtime = os.path.getmtime(f)
    modtime = datetime.fromtimestamp(modtime) 
    if (modtime < (now - timedelta(days=10))): 
        os.remove(f) 
    else:
        pass

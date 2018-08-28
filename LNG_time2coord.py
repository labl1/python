# to extract the coordinates from an aerosol layer, having it's time (and altitude)
# output = file containing for each point lon, lat, alt, time

# input =
# - file created with Create_LNG_Netcdf_level1.py
# - file containing time andf altitude of (manually) detected aerosols layers (usefull tool : https://apps.automeris.io/wpd/)
# to extract the coordinates from an aerosol layer, having its time (and altitude)
# output = file containing for each point lon, lat, alt, time

import os
import numpy as np
from warnings import warn
from collections import defaultdict
from netCDF4 import Dataset
import time as ttime
import glob as glob

dname = '/home/labl/Bureau/ABC2_files/netcdf/'

flightnb = '9'
dday = '06'

fname = 'LNG2_ABC_level1_201709'+dday+'_Vol'+ flightnb +'.nc'

# ascii file containing the time and altitude of the aerosol layer which coordinates have to be extracted
time_alt_files = glob.glob('/home/labl/Bureau/Vol'+flightnb+'_*layer*')
outfile_dir = '/home/labl/Bureau/full_coord/'

def get_lon_lat(time_data, lon_data, lat_data, t_hour):
    """ function to return the longitude and latitude corresponding to a time interval -
    or more precisely the average coordinates of the airplane during this time interval
    the *_data variables are from the netcdf level 1 files
    t_hour is the time (in hour UTC)for which the coordinates of the airplane are needed """

    # convert hour UTC into the decimal day of year

    t = t_hour / 24. + int(time_data[0].data)
    dt = 5. / 3600. / 24.

    indices = [i for i, elem in enumerate(time_data[:].data)
               if abs(elem - t) < dt]

    if len(indices) == 0:
        warn('No data corresponding to the time looked for at +/- 5 seconds - try +/- 30s')
        dt = 6. * dt
        indices = [i for i, elem in enumerate(time_data[:].data)
                   if abs(elem - t) < dt]
        if len(indices) == 0:
            warn('time : no data at t +/- 30 seconds - try +/- 2 min')
            dt = 4. * dt
            indices = [i for i, elem in enumerate(time_data[:].data)
                       if abs(elem - t) < dt]
    if len(indices) == 0:
        warn('time : no data at t +/- ' +str(dt) + ' seconds')
        avg_lon = np.nan
        avg_lat = np.nan
    else:
        avg_lon = np.mean([float(lon_data[i].data) for i in indices])
        avg_lat = np.mean([float(lat_data[i].data) for i in indices])

    return avg_lon, avg_lat, t


# 1 read file for the flight

flight_data = Dataset(dname + fname, 'r')

# time / alt file
lon_layer = []
lat_layer = []

for file in time_alt_files:
    t_all = []
    alt = []
    lon = []
    lat = []
    with open(file, 'r') as f, open(outfile_dir + os.path.basename(file) + '_coord.txt','w') as fout:
        for line in f:
            t0 = float(line.strip().split(';')[0])
            alt0 = float(line.strip().split(';')[1])
            lon0, lat0, t_doy =  get_lon_lat(flight_data.variables['time'], flight_data.variables['longitude'],
                               flight_data.variables['latitude'], t0)
            # actually not necessary here to create the lists as written one by one ...
            lon.append(lon0)
            lat.append(lat0)
            t_all.append(t0)
            alt.append(alt0)
            # write longitude, latitude, altitude of the points ; as well as the time (decimal day of year), and the decimal hour
            fout.write("%3.3f \t %3.3f \t %2.3f \t %3.5f \t %2.2f \n" % (float(lon0), float(lat0), alt0, t_doy, t0))

# print in file, with limited number of digits
# print("{0:.4f}".format(a))  => use only 3 digits ( meaning 110 meters for lat / lon and 1 m for height ; note that the ploane travels several hundred meters during the 5 sec measurement) ; BUt use .5f for time
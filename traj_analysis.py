# program to analyse the Z backtrajectories result
# => compute maps of the average vertical max displacment over the 105 back traj hours
# => identify how often an air parcel has raised of 3km or more over 3hours
# => create a netcdf file with all the results for plotting (maps or cross-sections)


# read the input files

import numpy as np
import matplotlib as mpl
from netCDF4 import Dataset
from math import pi
import warnings
from various_utils import get_profile_mnh, plot_2D_colormap

dir_in = '/home/labl/Bureau/'
filelist = ['XA540.1.SEP13.001cct.trajZ.nc','XA540.1.SEP13.002cct.trajZ.nc' ]
outfname = dir_in + 'quick_test.nc'

ni = 512
nj = 450
nlev = 65

dzth = 3. # threshold for 3-hour change in height (km)
#traj_all = np.zeros(len(file_in_liste),)

# carefull : memory limits
#dz_max_all = np.zeros(len(filelist),nlev,nj,ni)  # NB : one such float64 array corresponds to 28% of nuwa memory
#dz_max_3h_all = np.zeros(len(filelist),nlev,nj,ni)
#dz_min_3h_all = np.zeros(len(filelist),nlev,nj,ni)

nb_dZ_max    = np.zeros([nlev, nj, ni])
nb_dzp       = np.zeros([nlev, nj, ni])
nb_dzm       = np.zeros([nlev, nj, ni])
nb_dz_trj    = np.zeros([nlev, nj, ni])
dz_trj_sum   = np.zeros([nlev, nj, ni])
dz_max_3h_th = np.zeros([nlev, nj, ni])
dz_min_3h_th = np.zeros([nlev, nj, ni])
dZ_max_sum   = np.zeros([nlev, nj, ni])
dZ_max_max   = np.zeros([nlev, nj, ni])

dz_max_3h_sum = np.zeros([nlev, nj, ni])
dz_min_3h_sum = np.zeros([nlev, nj, ni])
dz_max_3h_th2 = np.zeros([nlev, nj, ni])
dz_min_3h_th2 = np.zeros([nlev, nj, ni])
dz_max_3h_max = np.zeros([nlev, nj, ni])
dz_min_3h_min = np.zeros([nlev, nj, ni])
for file_in in filelist:

    nctraj = Dataset(dir_in+file_in,'r')
    print(dir_in+file_in)
    time  = nctraj.variables['time'][:]
    trajZ = nctraj.variables['trajZ'][:,1:-1,1:-1,1:-1]
    trajZ[np.where(trajZ < -998.)] = np.nan # traj file does not have proper NaNs but large negative values instead
    trajZ[np.where(trajZ > 998.)] = np.nan # ùodel top at 26 km so not possible to exceed it
    #altfile = '/home/labl/Bureau/TEST0_ALT_dia_file.nc'
    #alt = Dataset(altfile,'r').variables('ALT')[1:-1, 1:-1, 1:-1]

    min_trajZ = np.nanmin(trajZ,axis=0)
    max_trajZ = np.nanmax(trajZ,axis=0)
    dZ_max = max_trajZ - min_trajZ

    # min and max changes over 3 hours
    dz_max_3h = np.nanmax(trajZ[1:,:,:,:]-trajZ[:-1,:,:,:], axis=0)
    dz_min_3h = np.nanmin(trajZ[1:,:,:,:]-trajZ[:-1,:,:,:], axis=0)

    # altitude variation between start and end of the traj
    traj_end = trajZ[-1,:,:,:]
    traj_st  = trajZ[0,:,:,:]
    dz_trj = traj_end - traj_st
    #same shapes (65, 450, 512) but still error : TypeError: only integer scalar arrays can be converted to a scalar index
    # with or without [:] gives same error ..... i.e. when converting to np.arrays ....
    #dz_trj_sum = np.nansum(np.array(dz_trj_sum), np.array(dz_trj))
    # works like that:
    dz_trj_sum = np.nansum(np.stack((dz_trj_sum,dz_trj)),axis=0) # stack array along one extra dimension before summing the resulting array along this dim
    nb_dz_trj [np.isfinite(dz_trj)] += 1

    # avoid creating too big arrays
    dZ_max_sum = np.nansum(np.stack((dZ_max_sum,dZ_max)),axis=0)
    dZ_max_max = np.nanmax(np.stack((dZ_max_max,dZ_max)),axis=0)
    nb_dZ_max[np.isfinite(dZ_max)] += 1

    dz_max_3h_sum = np.nansum(np.stack((dz_max_3h_sum,dz_max_3h)),axis=0)
    dz_max_3h_max = np.nansum(np.stack((dz_max_3h_max,dz_max_3h)),axis=0)
    nb_dzp [np.isfinite(dz_max_3h)]  +=1

    dz_min_3h_sum = np.nansum(np.stack((dz_min_3h_sum,dz_min_3h)),axis=0)
    dz_min_3h_min = np.nanmin(np.stack((dz_min_3h_min,dz_min_3h)),axis=0)
    nb_dzm [np.isfinite(dz_min_3h)] +=1

    dz_max_3h_th[np.where(dz_max_3h > dzth)] += 1
    dz_min_3h_th[np.where(dz_min_3h < -dzth)] += 1

    dz_max_3h_th2[np.where(dz_max_3h > dzth*2)] += 1
    dz_min_3h_th2[np.where(dz_min_3h < -dzth*2)] += 1

# calculate the average values and create the netcdf file
dZ_max_avg    = dZ_max_sum    / nb_dZ_max
dz_max_3h_avg = dz_max_3h_sum / nb_dzp
dz_min_3h_avg = dz_min_3h_sum / nb_dzm
dz_trj_avg    = dz_trj_sum    / nb_dz_trj

# fraction of dz above rthreshold
dz_max_3h_th = dz_max_3h_th / nb_dzp *100.
dz_min_3h_th = dz_min_3h_th / nb_dzm *100.

dz_max_3h_th2 = dz_max_3h_th2 / nb_dzp *100.
dz_min_3h_th2 = dz_min_3h_th2 / nb_dzm *100.
# create netcdf file with the data
out_data = Dataset(outfname, 'w', format='NETCDF4_CLASSIC')

# file Dimensions
levell = out_data.createDimension('level', nlev)
ni = out_data.createDimension('ni', ni)
nj = out_data.createDimension('nj', nj)

level = out_data.createVariable('level', np.float64, ('level',))
#altitude =

# NB other apprioach : for each region create a PDF of height changes ==> but that's a lot of PDFs

dZ_max_avg1 = out_data.createVariable('avg_z_range', np.float32, ('level', 'nj', 'ni'))
dZ_max_avg1.longname='average back-trajectories height range [ avg(max(alt)-min(alt)) ] '
dZ_max_avg1.units = 'km'

dZ_max_max1 = out_data.createVariable('max_z_range', np.float32, ('level', 'nj', 'ni'))
dZ_max_max1.longname='max back-trajectories height range [ max(max(alt)-min(alt)) ] '
dZ_max_max1.units = 'km'

dz_trj_avg1 = out_data.createVariable('avg_traj_dz', np.float32, ('level', 'nj', 'ni'))
dz_trj_avg1.longname='back-trajectory average height change avg(alt[end-alt[st]) '
dz_trj_avg1.units = 'km'

dz_max_3h_avg1= out_data.createVariable('avg_3h_max_dz', np.float32, ('level', 'nj', 'ni'))
dz_max_3h_avg1.longname= 'Maximum 3-hourly height increase - all-traj avg'
dz_max_3h_avg1.units = 'km'

dz_min_3h_avg1 = out_data.createVariable('avg_3h_min_dz', np.float32, ('level', 'nj', 'ni'))
dz_min_3h_avg1.longname= 'Maximum 3-hourly height decrease - all-traj avg'
dz_min_3h_avg1.units = 'km'

dz_max_3h_max1= out_data.createVariable('max_3h_max_dz', np.float32, ('level', 'nj', 'ni'))
dz_max_3h_max1.longname= 'Maximum 3-hourly height increase - all-traj maximum'
dz_max_3h_max1.units = 'km'

dz_min_3h_min1 = out_data.createVariable('min_3h_min_dz', np.float32, ('level', 'nj', 'ni'))
dz_min_3h_min1.longname= 'Maximum 3-hourly height decrease - all-traj max'
dz_min_3h_min1.units = 'km'

frac_dzup = out_data.createVariable('frac_dzup3km', np.float32, ('level', 'nj', 'ni'))
frac_dzup.longname = 'Fraction of trajectories with dz > 3km in 3h'
frac_dzup.units = '%'

frac_dzdown = out_data.createVariable('frac_dzdown3km', np.float32, ('level', 'nj', 'ni'))
frac_dzdown.longname = 'Fraction of trajectories with dz < -3km in 3h'
frac_dzdown.units = '%'

frac_dzup2 = out_data.createVariable('frac_dzup6km', np.float32, ('level', 'nj', 'ni'))
frac_dzup2.longname = 'Fraction of trajectories with dz > 6km in 3h'
frac_dzup2.units = '%'

frac_dzdown2 = out_data.createVariable('frac_dzdown6km', np.float32, ('level', 'nj', 'ni'))
frac_dzdown2.longname = 'Fraction of trajectories with dz < -6km in 3h'
frac_dzdown2.units = '%'


longitude =  out_data.createVariable('longitude', np.float64, ('nj','ni'))
latitude =  out_data.createVariable('latitude', np.float64, ('nj','ni'))

level[:] = nctraj.variables['level'][1:-1]
longitude[:] = nctraj.variables['longitude'][1:-1,1:-1]
latitude[:]= nctraj.variables['latitude'][1:-1,1:-1]


dZ_max_avg1[:] = dZ_max_avg
dz_trj_avg1[:] = dz_trj_avg
dz_max_3h_avg1[:] = dz_max_3h_avg
dz_min_3h_avg1[:] = dz_min_3h_avg

dZ_max_max1[:] = dZ_max_max
dz_max_3h_max1[:] = dz_max_3h_max
dz_min_3h_min1[:] = dz_min_3h_min

frac_dzup[:] = dz_max_3h_th
frac_dzdown[:] = dz_min_3h_th
frac_dzup2[:] = dz_max_3h_th2
frac_dzdown2[:] = dz_min_3h_th2
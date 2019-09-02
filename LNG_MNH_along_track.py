# to get the MNH points correspounding to an LNG flight track
# only the spatial colocation is done, there is no time colocation
# the lidar data are not degraded, only the MNH points are extracted and the time-height colorplot is created

import numpy as np
import matplotlib as mpl
from netCDF4 import Dataset
from math import pi
import warnings
from various_utils import get_profile_mnh, plot_2D_colormap

# read lidar points and get MNH corresponding profiles


def create_alongF20_nc(outfname,variables,t_lidar,alt,time_mnh_all,alt_lidar,ilon,ilat,lon,lat,A1064,mnh_simulation_info, varnamelist,long_names_list,varunits_list):

    ## TODO: NB here we use levels, but this is a crude approximation over orography => replace by height  and then interpolate over regular grid e.g. the lidar grid for comparison  :
    # formula_terms = "s: level_w height: ZTOP orog: ZS";
    # formula_definition = "z(n,k,j,i)=s(k)*(height-orog(j,i))/height+orog(j,i)";

    out_data = Dataset(outfname, 'w', format='NETCDF4_CLASSIC')

    # file Dimensions
    altitude = out_data.createDimension('altitude', nb_vert_lev)
    time     = out_data.createDimension('time', nb_pro_lidar)
    altitude_LNG = out_data.createDimension('altitude_LNG', np.size(alt_lidar))



    times      = out_data.createVariable('time', np.float64, ('time',))
    altitudes  = out_data.createVariable('altitude', np.float32, ('time','altitude'))
    altitude_lidar = out_data.createVariable('altitude_LNG', np.float32, ('altitude_LNG',))

    var_dct={}

    for i in range(len(varnamelist)):
        var_dct["var"+str(i)] = out_data.createVariable(varnamelist[i], np.float32, ('time','altitude'))
        var_dct["var"+str(i)].long_name = long_names_list[i]
        var_dct["var"+str(i)].units = varunits_list[i]
        #standard CF names cloud_ice_mixing_ratio and cloud_liquid_water_mixing_ratio


    '''
    if len(varnamelist) == 1:
        var1 = out_data.createVariable('tracer', np.float32, ('time','altitude'))
        var1.long_name = 'MNH passive tracer'
    elif len(varname_list) == 6:
        var1 = out_data.createVariable('tracer1', np.float32, ('time','altitude'))
        var1.long_name = 'MNH passive tracer'
        var1.mnh_name  = varname[0]
        var2 = out_data.createVariable('tracer2', np.float32, ('time','altitude'))
        var2.long_name = 'MNH passive tracer'
        var2.mnh_name  = varname[1]

        var3 = out_data.createVariable('tracer3', np.float32, ('time','altitude'))
        var3.long_name = 'MNH passive tracer'
        var3.mnh_name  = varname[2]

        var4 = out_data.createVariable('tracer4', np.float32, ('time','altitude'))
        var4.long_name = 'MNH passive tracer'
        var4.mnh_name  = varname[3]
    '''


    hour_lidar = out_data.createVariable('hour_lidar', np.float32, ('time',))
    hour_lidar.long_name = 'LNG time'
    hour_lidar.units     = 'hour UTC'
    hour_mnh = out_data.createVariable('hour_mnh', np.int16, ('time',))
    hour_mnh.long_name = 'Meso-NH output times'
    hour_mnh.units = 'hour UTC'

    latitudes  = out_data.createVariable('latitude', np.float32, ('time',))
    longitudes = out_data.createVariable('longitude', np.float32, ('time',))
    lon_index = out_data.createVariable('mnh_lon_index', np.int16, ('time',))
    lon_index.long_name = 'Meso-NH X-direction index corresponding to the lidar data'
    lat_index = out_data.createVariable('mnh_lat_index', np.int16, ('time',))
    lat_index.long_name = 'Meso-NH longitude index corresponding to the lidar data'

    ABC_1064nm = out_data.createVariable('ABC_1064nm', np.float32,
    ('time','altitude_LNG'), fill_value =  -9999.999)
    ABC_1064nm.long_name = 'Apparent backscattered signal at 1064nm' #'Attenuated backscatter coefficient at 1064 nm'
    ABC_1064nm.units     = 'arbitrary units' #'km-1 sr-1'
    ABC_1064nm.valid_min = 0.

    latitudes.units  = 'degree_north'
    latitudes.long_name = 'Latitude'
    latitudes.standard_name = 'latitude_north'

    longitudes.units = 'degree_east'
    longitudes.long_name = 'Longitude'
    longitudes.standard_name = 'longitude_east'

    altitudes.units = 'm'
    altitudes.standard_name = 'altitude'
    altitudes.long_name = 'MNH altitude AMSL'


    altitude_lidar.units = 'km AMSL'
    altitude_lidar.long_name = 'Altitude for LNG backscatter'


    times.units = 'days since 2016-12-31 00:00:00'  # this is the day of year
    times.long_name = 'Time of the lidar profiles'
    times.calendar = 'gregorian'
    times.standard_name = 'time'

    out_data.description = 'Meso-NH data extracted at F20 LNG profile locations ' \
                            + mnh_simulation_info


    times[:] = t_lidar
    altitudes [:]  = alt
    altitude_lidar [0:] = alt_lidar

    longitudes[:] = lon
    latitudes[:] = lat
    lon_index[:] = ilon
    lat_index[:] = ilat
    ABC_1064nm[:] = A1064
    for i in range(len(varnamelist)):
        var_dct["var"+str(i)][:] = variables[i,:,0:nb_vert_lev] # [var_id,time,vert_lev]
        try:
            print('shape variables and var_dct:')
            print(variables.shape())
            print( var_dct["var"+str(i)][:].shape() )
        except:
            print('warning : cannot print')


    #var1[:] = tracer
    # put additional variables here if wanted

    hour_lidar [:] = ( t_lidar%1 ) * 24.
    hour_mnh   [:] = np.int16(time_mnh_all)

    out_data.close()

## ************************** MAIN **************************************** ##

# Variables to edit
infile = '/home/labl/Bureau/Lidars/LNG_DATA/ABC2_files/netcdf/LNG2_ABC_level1_20170905_Vol6.nc'

alt_MNH_file = '/home/labl/Bureau/TEST0_ALT_dia_file.nc'

simu_name = 'XA54b'

outname = '/home/labl/Bureau/20170905_Vol6_MNH_'+simu_name+'_pTracer.nc' # output netcdf file with the MNH data

nb_vert_lev = 65

out_freq = 3 # output frequency in hours

nbtracers=4

other_var=["RCT","RIT"] # non-tracer variables to plot as extra contours; e.g. cloud variables

long_names = ["MNH passive tracer 1: all BC sources - mask", "MNH passive tracer 2: Small BC sources centre-South - mask ", "MNH passive tracer 3: BC sources South-East - mask " , "MNH passive tracer 4: all BC sources - float ",
              'MNH Cloud mixing ratio', 'MNH Ice mixing ratio']

# day in september 2017
dd = '05'
inres = 12. # meso-NH simulation resolution

mnh_simulation_info = 'Meso-NH 5.4.2 simulation '+ simu_name + ' at ' + str(inres) +  ' km resolution' \
                      'with: ' \
                      ' ni = 514 (x-direction)' \
                      ' nj = 452 (y-direction)' \
                      ' level = 67 ' \
                      ' and  lon0 = 10 °E ; lat0 = -15 °N' \
                      'Meso-NH files used :' \
                      ' '


#

varname_list=['SVT00'+str(i) for i in range(1,nbtracers+1)]
for i in range(len(other_var)):
    varname_list.append(other_var[i])

nbvar=len(varname_list)


nb_out = 24/out_freq

ncfile1 = Dataset(infile, 'r')
lat = ncfile1.variables['latitude'][:]
lon = ncfile1.variables['longitude'][:]
time = ncfile1.variables['time'][:]
obs_type_flag = ncfile1.variables['obs_type_flag'][:] # (0: Nadir, 1: zenith, 2: admin)
ABC_1064nm = ncfile1.variables['ABC_1064nm'][:,:]
alt_lidar = ncfile1.variables['altitude'][:]
#  Read MNH file at the closes time from the file tstart

nb_pro_lidar = np.size(time)


t_int_h = np.array((np.round(( time%1 ) * nb_out ))).astype(int) # round to the nearest integer hour (or segment number in case output is not hourly) to read the nearest mnh file

t_unique = np.unique(t_int_h).astype(int)
time_mnh_all_2D = np.zeros([nb_pro_lidar, nb_vert_lev])
var_all = np.zeros([nbvar, nb_pro_lidar, nb_vert_lev])

alt = np.zeros([nb_pro_lidar, nb_vert_lev])
time_mnh_all = np.zeros(nb_pro_lidar)
ilon = np.zeros(nb_pro_lidar)
ilat = np.zeros(nb_pro_lidar)


for tt in t_unique:
    if simu_name == 'TESTA':
        mnh_file_nn = 'BG54b'
    else:
        mnh_file_nn = simu_name
    if simu_name!='XA54b' and simu_name!='XA540':
        mnh_file = mnh_file_nn + '.1.SEP' + dd + '.0' + str(tt).zfill(2) +'_selectedVar.nc' #'BG54b.1.SEP' + dd + '.0 '+  tt.zfill(2) + '.nc'
    else:
        mnh_file = mnh_file_nn + '.1.SEP' + dd + '.0' + str(tt).zfill(2) +'_SVT_CLD.nc'
    mnh_simulation_info = mnh_simulation_info + mnh_file + ' '

    for i in np.where(t_int_h == tt)[0]: # select the indices corresponding to a given hour (so that a different MNH file is used for each of them)
        # read MNH corresponding file
        mnh_dir = '/home/labl/Bureau/TEST0_12km_ncfiles/'
        if simu_name == 'TEST0':
            mnh_dir  = '/home/labl/Bureau/MNH_simulations/TEST0_noConvParam_12km_free/ncfiles/'
        elif simu_name == 'TESTA':
            mnh_dir  = '/home/labl/Bureau/MNH_simulations/TESTA_noConvParam_12km_nudged/TESTA_12km_ncfiles/'
        elif simu_name == 'BG54b':
            mnh_dir  = '/home/labl/Bureau/MNH_simulations/BG54b_12km/ncfiles/'
        else:
            mnh_dir  = '/home/labl/Bureau/'

        #ncMNH = Dataset(mnh_dir+mnh_file, 'r')
        #tracer = ncMNH.variables['SVT001']
        #mnh_lat = ncMNH.variables['latitude']
        #mnh_lon


        variables_all, alt1, varunits_all, time_mnh,indlon,indlat, lllon,lllat  = get_profile_mnh(mnh_file, mnh_dir, varname_list, inres, lat[i], lon[i],
                                                          nan_val=999., inunits=[], lclosest=True,alt_file=alt_MNH_file)
        alt[i,0:nb_vert_lev] = alt1[1:-1]
        var_all[:,i,0:nb_vert_lev] = variables_all[:,1:-1]
        time_mnh_all[i] = time_mnh[0]
        ilon[i] = indlon
        ilat[i] = indlat

        # print(np.size(tracer))
        # uyse ilon, ilat to get other variables if needed
        # TODO: again, this could be made much faster than this brute-force approach

# save the extracted data (coordinates in the meso-NH files, mnh_time, flight_time, flight_lon, flight_lat) in a netcdf file for easy reusing and plotting

create_alongF20_nc(outname, var_all, time, alt, time_mnh_all, alt_lidar, ilon, ilat, lon, lat, ABC_1064nm, mnh_simulation_info,
                   varname_list, long_names, varunits_all)

'''
time_mnh_all_2D[:, :] = np.transpose(np.array([time_mnh_all for x in range(nb_vert_lev)]))

norm = mpl.colors.Normalize(vmin=0.,vmax=0.15)

plot_2D_colormap(time_mnh_all_2D, alt/1000., tracer_all,
                out_name='passive_tracer_MNH_'+simu_name+'_vol6', cnorm=norm, out_path_fig='/home/labl/Bureau/', out_type='png',
                lyinvert=False, cmap='jet',
                title=simu_name+' passive tracer', xlabel='Time (days in year 2017)', ylabel='Altitude (km)',ymax = 10., ymin=0.)

plot_2D_colormap(time_mnh_all_2D, alt, tracer_all,
                out_name='test', cnorm=Normalise, out_path_fig='/home/labl/Bureau/', out_type='pdf',
                lyinvert=True, cmap='gist_ncar',
                title='', xlabel='', ylabel='')
'''
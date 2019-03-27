#!/usr/bin/python3 
# to read and analysre MSG 10.8 microns observations and meso-nh data

from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# some of my own functions:
from various_utils import create_hist1D_plot, LT_hour, plot_2D_colormap

# Meso-NH filenames of the type  exp_name+ '.1.' + 'SEP' + DD + '.0' + HH + 'dia_MSG108.nc'
# (e.g. for 2nd September 01UTC: BIG54.1.SEP02.001dia_MSG108)
# MSG filenames of type "OBSER.8." + YYYY + MM + DD + HH + mm +'.nc'  e.g. OBSER.8.201709020200.nc 

# ----------------------- variables to edit ---------------------------
# directories containing the MSG and Meso-NH netcdf file
filedir_obs='/home/labl/Bureau/MSG_OBS_WAL54/'
filedir_mnh='/home/labl/Bureau/MSG_mnh_WAL54b/BG54b_01-130908_15UTC_MSG108/'
filedir_mnh2='/home/labl/Bureau/MSG_mnh_WAL54b/BG54b_01-130908_15UTC_MSG108/'
# Meso-NH experiment name
mnh_exp='BG54b'

# output directory 
out_dir='/home/labl/Bureau/BG54b_figures/'

YYYYMM   = "201709"
strmonth = 'SEP'
# start and end day (from 00UTC to 23UTC)
dstart =3  #1
dend   =11 #13

lobs = False     # using MSG obs or not for histograms (NB: mind the possible differences between MNH grids and the processed MSG data from JPC)
lobs_2D = False # 2D maps from obseravtions
# file used fior lat and lon coordinates of the gridded model (and MSG) points
coord_file = filedir_mnh + 'BG54b.1.SEP02.001dia_MSG108.nc' #''BG54b.1.SEP03.021dia_MSG108.nc'

# ----------------------- functions ----------------------------------- #

def get_BT108_file(flag,daterange,datem1range,hrange):
# flag: 0 for MSG OBS and 1 (or more) for the diagnosed meso-nh MSG 10.8 BT
# NB: this is for a day in September 2017 (201709)
# daterange: array of strings, format DD , e.g. ['01','02','03']
# datem1range : date - 1, same as date but for the days before
# hrange: array containing the hours UTC to be considered - must be continuous integers ; for full day put range(24)
# output :BT108_all is in format [nx,ny,time]

    jj=0
    for day in daterange:
        datem1 = YYYYMM + datem1range[jj].zfill(2)
        date   = YYYYMM + daterange[jj].zfill(2)
        if flag ==0 : # MSG observations 
                fn     = 'OBSER.8.' + date
                suffix = '00.nc'
                filedir=filedir_obs
        elif flag==1: # Meso-NH diag
                fn      = mnh_exp + '.1.' + strmonth + daterange[jj].zfill(2)   + '.0'
                fnm1    = mnh_exp + '.1.' + strmonth + datem1range[jj].zfill(2) + '.0'
                suffix  = 'dia_MSG108.nc'
                filedir = filedir_mnh

         # read the files and create an hUTC value
        for i in hrange:
            hh=str(i).zfill(2)
            if i == 0 and flag==1: # the meso-NH file 24 correspond to the next day at 00UTC => this needs correction
                filename = fnm1 + '24' + suffix
                hh='00'
            elif i < 24:
                filename = fn + hh + suffix
            else:
                print("hh = " + hh)
                raise ValueError('hour must be between 00 and 24')
    
            dataset = Dataset(filedir + filename)
            BT108   = dataset.variables['MSG2_108BT']
            BT108   = np.expand_dims(BT108, axis=2)
            if i == min(hrange) and jj == 0:
                BT108_all=BT108
            else:
                BT108_all=np.append( BT108_all,BT108,axis=2)
                
        #TODO :expand array to have the day in the last column 
                
        jj=jj+1
            
    return BT108_all
    
# ------------------------------------------------------------------------------------------------

def create_2D_hist_LT(array3D, hrange, lon_arr, nbins=100, daily_tsteps = 24):
    # array3D : input 3D array (x,y,time), muist contain hourly data, and full days
    # (time dimension being a multiple of 24, or of daily_tsteps if nuùmber of timestep per day is not 24)
    # hrange  : tuple containing the min and max value for the hist bins
    # lon_arr : longitude array of variables
    # daily_tsteps : the number of time steps per day (24 when using hourly data)
    # ***** Important Note *******
    #  the output histogram will in any case be HOURLY (even if the input is at higher time frequency)
    # (to modify that the function LT_hour in various_utils has to be modified)
    arr_shape = np.shape(array3D)
    nb_days = arr_shape[2]/float(daily_tsteps)
    if abs(nb_days - int(nb_days)) > 1.0E-6:
        raise ValueError("number of days is not an integer, niot suited for diurnal cycle analysis")
    else:
        nb_days = int(nb_days)

    array4D = array3D.reshape(arr_shape[0], arr_shape[1], nb_days, daily_tsteps)
    # 1) initialise arrays
    h_arr = np.empty_like(array4D)
    hist_2D_LT = np.empty(shape=(100, 24))

    # ***** Important note ********
    # we neglect hereafter the very small changes in LT with the date - over 10 days it's not more than a few minutes
    #  of error - to be very accurate, calc h_arr[:,:,k,i] calling LT_hour with explicit specification of the date
    for i in range(daily_tsteps):
        h_arr[:, :, 0, i] = LT_hour(i,lon_arr, ref_yr=2008, ref_month=9, ref_day=7).reshape(arr_shape[0], arr_shape[1])
    for k in range(nb_days)[1:]:
        h_arr[:, :, k, :] = h_arr[:, :, 0, :]

    # histogrqms calculation
    for i in range(24): # NB: THE OUTPUT HISTOGRAM IS HOURLY NO MATTER THE INPUT frequency (daily_tsteps)
        # 2 use it for selecting the data = for i from 1 to 24 create one histogram
        posi = np.where(np.abs(h_arr - i) < 0.1)
        hist_2D_LT[:, i], bins = np.histogram(array4D[posi],
                                            range=hrange, bins=nbins, normed=False) # if needed normqlisqtion to be done at the end

    bin_centers = (bins[1:] + bins[:-1]) * 0.5

    return hist_2D_LT, bin_centers

def plot_averages(hist2D_LT_obs, hist2D_LT_mnh, bin_centers,minBT_str, out_path_fig = out_dir, llobs = True):
    if llobs : hist_temp     = np.empty_like(hist2D_LT_obs)
    hist_temp_mnh = np.empty_like(hist2D_LT_mnh)
    for i in range(24):
        if llobs : hist_temp[:,i]= (hist2D_LT_obs[:,i] * bin_centers) / np.sum(hist2D_LT_obs [:,i])
        hist_temp_mnh[:,i]= ( hist2D_LT_mnh[:,i] * bin_centers) / np.sum( hist2D_LT_mnh[:,i])

    if llobs : avr_obs = np.sum(hist_temp, axis = 0)
    avr_mnh = np.sum(hist_temp_mnh, axis = 0)

    fig1 = plt.figure()
    ax1  = fig1.add_subplot(1, 1, 1)
    if llobs : ax1.plot(range(24),avr_obs, label='MSG OBS')
    ax1.plot(range(24),avr_mnh, label='Meso-NH')
    ax1.set_ylabel('Domain-average Brightness Temperature')
    ax1.set_xlabel('Local Time (hour)')
    ax1.set_title('Domain-average diurnal cycle of BT ( BT < ' + minBT_str + ' K  ) ')

    ax1.set_xticks(np.arange(0,25,2))
    ax1.set_xlim(0,24)
    ax1.grid()
    ax1.invert_yaxis()
    ax1.legend()

    fig1.savefig(out_path_fig + '/' + 'Average_BT_diurnal_cycle_' + minBT_str + 'K.pdf')
# --------------------------------- MAIN --------------------------------------------------------- 

# read lon and lat of datapoints - usefull for local time
datacoord = Dataset(coord_file)
lon   = datacoord.variables['LON']
# lat   = datacoord.variables['LAT'] # lat is not needed

# create input range in the right format
daterange   = [str(x).zfill(2) for x in range(dstart,dend+1)]
daterangem1 = [str(x).zfill(2) for x in range(dstart-1,dend)]

# get the meso-NH and MSG data

if lobs:
    BT108_all_obs = get_BT108_file(0, daterange, daterangem1, range(24)) # from 00UTC to 23UTC each day

BT108_all_mnh = get_BT108_file(1, daterange, daterangem1, range(24))

# replace 999. in mnh files by NaN
BT108_all_mnh[ BT108_all_mnh > 998. ]= np.NaN
print(BT108_all_mnh.shape)
if lobs:
    legend_list=('OBS03-11','MNH03-11') # legends for 1D - histograms
else:
    legend_list=('MNH03-11')
# ranges of temperature for histograms, keep the same minimum for all
hrange0=(182,            328)
hrange1=(np.min(hrange0),260)
hrange2=(np.min(hrange0),230)

for hrange in [ hrange0, hrange1, hrange2 ]:

    BTmax=str(np.max(hrange))
    # create the 1D-histograms plots
    if lobs:
        create_hist1D_plot((BT108_all_obs, BT108_all_mnh), legend_list, hrange0,
                       'BT108_03-11_distribution', out_dir, xlabel='10.8 microns Brightness temperature (K)',
                       title='BT distribution (BT < ' + BTmax + ' K)')
        # -------- create 2D histograms BT - local Time  ------- --------------------------------------- #
        hist2D_LT_obs, bins_LT = create_2D_hist_LT(BT108_all_obs, hrange, lon)
        plot_averages(hist2D_LT_obs, hist2D_LT_mnh, bins_LT, BTmax, out_path_fig=out_dir)

    else:
        create_hist1D_plot((BT108_all_mnh,), legend_list, hrange0,
                       'BT108_03-11_distribution', out_dir, xlabel='10.8 microns Brightness temperature (K)',
                       title='BT distribution (BT < ' + BTmax + ' K)')
        # -------- create 2D histograms BT - local Time  ------- --------------------------------------- #

    hist2D_LT_mnh, bins_LT = create_2D_hist_LT(BT108_all_mnh, hrange,lon)

    plot_averages(hist2D_LT_mnh, hist2D_LT_mnh, bins_LT, BTmax, out_path_fig = out_dir, llobs = lobs)



    # TODO also plot individual histograms per time (?)
    if lobs_2D:
        plot_2D_colormap(range(24), bins_LT, hist2D_LT_obs,
                    out_name='2D_hist_dirnal_MSG_BT108_LocalTime_'+ BTmax, out_path_fig=out_dir,
                    cnorm=LogNorm(vmin=np.min(hist2D_LT_mnh[np.nonzero(hist2D_LT_mnh)]),
                                  vmax=np.max(hist2D_LT_mnh)),
                    title='MSG BT 10.8 microns distribution (BT < ' + BTmax + ' K)', xlabel='Local Time (hour)',
                    ylabel='Brightness temperature (K)')
        plot_2D_colormap(range(24), bins_LT, hist2D_LT_obs/np.sum(hist2D_LT_obs),
                    cnorm=LogNorm(vmin=np.min(hist2D_LT_mnh[np.nonzero(hist2D_LT_mnh)])/np.sum(hist2D_LT_mnh),
                                  vmax=np.max(hist2D_LT_mnh)/np.sum(hist2D_LT_mnh)),
                    out_name='2D_hist_dirnal_MSG_BT108_LocalTime_norm_'+ BTmax, out_path_fig=out_dir,
                    title='MSG BT 10.8 microns normalised distribution (BT < ' + BTmax + ' K)', xlabel='Local Time (hour)',
                    ylabel='Brightness temperature (K)')


    plot_2D_colormap(range(24), bins_LT, hist2D_LT_mnh,
                 cnorm=LogNorm(vmin=np.min(hist2D_LT_mnh[np.nonzero(hist2D_LT_mnh)]),
                                  vmax=np.max(hist2D_LT_mnh)),
                    out_name='2D_hist_dirnal_MNH_BT108_LocalTime_'+ BTmax, out_path_fig=out_dir,
                    title='Meso-NH BT 10.8 microns distribution (BT < ' + BTmax + ' K)', xlabel='Local Time (hour)',
                    ylabel='Brightness temperature (K)')

    plot_2D_colormap(range(24), bins_LT, hist2D_LT_mnh/np.sum(hist2D_LT_mnh),
                    out_name='2D_hist_dirnal_MNH_BT108_LocalTime_norm_'+ BTmax, out_path_fig=out_dir ,
                    cnorm=LogNorm(vmin=np.min(hist2D_LT_mnh[np.nonzero(hist2D_LT_mnh)]) / np.sum(hist2D_LT_mnh),
                                  vmax=np.max(hist2D_LT_mnh) / np.sum(hist2D_LT_mnh)),
                    title='Meso-NH BT 10.8 microns normalised distribution (BT < ' + BTmax + ' K)', xlabel='Local Time (hour)',
                    ylabel='Brightness temperature (K)')




# ------ ------ end of script ------ ------

'''
# 1D histograms plots at selected times
def create_hist1D_plot(array_tuple, legend_tuple, hrange,
		       out_name, out_path, out_type='pdf',
                       nbins=100, lnorm=True,
                       yyscale='log',xlabel='',title=''):
'''
## old stuff : for calculatinh histograms without changing to Local time
'''
# NB our doimain is probably too big to look at a domain-wide diurnal cycle 
#(although the actual difference between LT 2D hist and UTC 2D hist is rather moderate)

def create_2d_hist_diurnal(hrange, array3D, lnorm):
# function to get tyhe 2D histogram BT-hour of the day (00-23)
# hbins: BT bins (Kelvin)
# array3D: array with shape = x,y,time  (here timeseries of 2D 10.8microns BT from MSH or Meso-NH)
# lnorm : normalised or not? (this is for a normalisation of the 2D hist)
    
    nb_days = np.shape(array3D)[2]/24.
    
    #reshape to x,y,h, day
    if abs(nb_days - int(nb_days)) > 1.0E-6:
        raise ValueError("number of days is not an integer, niot suited for diurnal cycle analysis")
    else:
        nb_days = int(nb_days)
        array3D = array3D.reshape([array3D.shape[0], array3D.shape[1], nb_days, 24])
        
    #calculate histograms for each
    for i in range(24):
        hist1D,hbins = np.histogram(array3D[:,:,:,i], range=hrange, bins = 100, normed = False)
        hist1D = np.expand_dims(hist1D, axis = 1)
        if i==0:
            hist2D = hist1D
        else:
            hist2D=np.append(hist2D, hist1D, axis = 1)
    
    if lnorm:
        hist2D = hist2D/np.sum(hist2D)
    
    bin_centers = (hbins[1:]+hbins[:-1])*0.5
    

    
    return hist2D, bin_centers
    

# create the 2D-histograms plots 
# calc hist
hist2D_obs,bin_centers =  create_2d_hist_diurnal(hrange2, BT108_all_obs, True)
hist2D_mnh,bin_centers =  create_2d_hist_diurnal(hrange2, BT108_all_mnh, True)
# plot

plot_2D_colormap(range(24), bin_centers, hist2D_obs, cnorm=LogNorm(vmin=np.min(hist2D_obs[np.nonzero(hist2D_obs)]),
                                                               vmax=np.max(hist2D_obs)),
             out_name='2D_hist_dirnal_MSG_BT108', out_type='pdf',
             lyinvert=True, cmap='gist_ncar',
             title='MSG BT 10.8 microns normalised distribution (BT < 230 K)', ylabel='Brightness temperature (K)',
              xlabel='Hour UTC')

plot_2D_colormap(range(24), bin_centers, hist2D_mnh, cnorm=LogNorm(vmin=np.min(hist2D_obs[np.nonzero(hist2D_obs)]),
                                                               vmax=np.max(hist2D_obs)),
                 out_name='2D_hist_dirnal_MNH_BT108', out_type='pdf',
                 lyinvert=True, cmap='gist_ncar',
                 title='Meso-NH BT 10.8 microns normalised distribution (BT < 230 K)', ylabel='Brightness temperature (K)',
                  xlabel='Hour UTC')
'''
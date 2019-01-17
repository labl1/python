#!/usr/bin/python3
# or use specific anaconda3 environment
# various_utils
# module containing various usefull functions 
# L. Labbouz, July 2018

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker, cm, colors
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap as get_cmap
from netCDF4 import Dataset
from math import pi
import os

def str2float(list_str):
# tranforms a list of string into a list of floats
    list_float = [float(i) for i in list_str]
    return list_float


def calc_local_hour(yr,month,day,H_UTC,min_UTC, lon):
# function to calculate the local solar time 
# (i.e. the hour angle of the sun + 12 h)
# inputs
# date UTC : yr, month, day, H_UTC, min_UTC
# lon: longitude in degrees East
# outputs
# local solar hour (decimal)

    import ephem
    import datetime

    sun=ephem.Sun()
    # set date, longitude manually
    o = ephem.Observer()
    o.date = datetime.datetime(yr, month, day, H_UTC, min_UTC, 0) # some utc time ; could replace 0 by seconds
    o.lon = lon * ephem.degree # 
    sun.compute(o)
    hour_angle = o.sidereal_time() - sun.ra
    solartime  = ephem.hours(hour_angle + ephem.hours('12:00')).norm  # norm for 24h
    # nb: lat has no impact on solar time
    time_str=str(solartime)
    h, m, s = time_str.split(':')
    local_h = int(h) + int(m)/60. + float(s)/3600.
    
    return local_h


def LT_hour(hourUTC, lon_arr, ref_yr, ref_month, ref_day):
    # Calculate local hour from hour UTC and latitude; for most precise calculation yr, month and day are needed
    # nb: INTEGER HOURS ONLY ARE CONSIDERED HERE; ALL NON-INTEGER VALUE IS ROUNDED

    # this is the difference between local hour and UTC hour
    # LT = UTC - delta_to_LT
    delta_to_LT = np.array([round(calc_local_hour(ref_yr, ref_month, ref_day, 12, 0, llon) - 12.)
                            for llon in np.hstack(lon_arr)])

    hour = (hourUTC + delta_to_LT) % 24.
    return hour

# ------------------------------------------------------------------------------------------------------ #

def create_hist1D_plot(array_tuple, legend_tuple, hrange, 
		       out_name, out_path, out_type='pdf',
                       nbins=100, lnorm=True, 
                       yyscale='log',xlabel='',title=''):
# function to create 1D histograms; It generates two figures (for frequency qnd cumulqtive frequency)
# array_tuple  : a tuple (or list, or array) containing all the arrays to be used for histograms 
#                any shape of array, np array, tuple is allowed, np.histogram will broadcast it.
# legend_tuple : a tuple (or list, or array) of labels to be used in legend. Must be strings.
# out_name     : name of tyhe figure to output
# out_type     : str containing the type of the output file (default 'pdf')
# hrange       : min and max for the histogram bins (in a list or array or tuple)
# nbins        : number or bins in hrange OR actual bin values in which case it overwrites rmin and rmax
# lnorm        : normed histograms if true 
# yyscale      : 'linear' or 'log' y-axis 
# xlabel,title : for the figure title / xlabel

    nbplt = 0
    for array in array_tuple:
        hist, bins = np.histogram(array, range=(hrange[0],hrange[1]),bins=nbins, density=lnorm)
        bin_centers = (bins[1:]+bins[:-1])*0.5
        if nbplt == 0:
            fig1 = plt.figure()
            ax1  = fig1.add_subplot(1, 1, 1)
            fig2 = plt.figure()
            ax2  = fig2.add_subplot(1, 1, 1)
            
            ax1.plot(bin_centers, hist, label=legend_tuple[nbplt])
            ax1.set_yscale(yyscale)
            
            ax2.plot(bin_centers, np.cumsum(hist), label=legend_tuple[nbplt])
            ax2.set_yscale(yyscale)
            
            if lnorm:
                ax1.set_ylabel('Frequency')
                ax2.set_ylabel('Cumulative frequency')
            else:
                ax1.set_ylabel('Number of occurences')
                ax2.set_ylabel('Cumulative number of occurences')

            ax1.set_xlabel(xlabel)
            ax1.set_title(title)
            ax2.set_xlabel(xlabel)
            ax2.set_title(title)
        else:

            print('nbplt: ', nbplt)
            print('legend', legend_tuple )
            ax1.plot(bin_centers, hist,            label=legend_tuple[nbplt])
            ax2.plot(bin_centers, np.cumsum(hist), label=legend_tuple[nbplt])
                    
        nbplt = nbplt + 1
        
    # display legends
    ax1.legend()
    ax2.legend()
    # save the figures 
    fig1.savefig(out_path + '/' + out_name + '.'       + out_type)
    fig2.savefig(out_path + '/' + out_name + '_cumul.' + out_type)

# --------*********************************************************---------------------------------------------- #

def plot_2D_colormap(x, y, array2D,
                out_name, cnorm, out_path_fig, out_type='pdf',
                lyinvert=True, cmap='gist_ncar',
                title='', xlabel='', ylabel=''):
    # function to plot 2dimensional histograms as colormap
    # NB: can also be used to plot any 2-D colormap, no matter the input
    # array2D dimension must be consistent with x and y
    # outname : output file name
    # cnorm: either LogNorm or Normalise (or other norm from matplotlib.colors)

    # NB: giving a default norm leads to strange things, like if the first call of LogNorm()
    # would determine the results of the next one and hence possibly lead to color limits not fitting the data

    fig1 = plt.figure()
    ax1  = fig1.add_subplot(1, 1, 1)
    im = ax1.pcolor(x, y, array2D, cmap=cmap, norm=cnorm)
    if lyinvert:
        ax1.invert_yaxis()
    fig1.colorbar(im,ax=ax1)
    ax1.set_xlabel(xlabel)
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    fig1.savefig(out_path_fig + '/' + out_name + '.' + out_type)
    plt.close(fig1)


def find_indices(lst, condition):
    ''' return indices verifying condition
    the condition has to be given using lambda e.g.
     lambda x : x > 3
     (to return indices correspounbding to all values greater than 3)'''
    return [i for i, elem in enumerate(lst) if condition(elem)]

### attention: en development ###
# plot profile
# TODO add function for 2D height - time profiles

def get_profile_mnh(infile, indir, varname, inres, loc_lat, loc_lon,
                    nan_val=999., inunits=[]):
    ''' function to get the average profile from the nearest model gridcolumns
     the return array is the average of all the model gridcolumn that are less than inres km away from (loc_lat,loc_lon)
     infile : input file name
     indir  : input directory full path
     varname: name of tyhe variable to retrieve
     inres  : resolution of the input data (km)
     loc_lat: latitude of desired location for the profiles
     loc_lon: longitude of ...
     nan_val: value of NaNs in input file
     inunits: units can be specified here, otherwise it will be read in netcdf file and if absent "unk" will be used
     '''

    ilat = []
    latval = []
    ilon = []
    lonval = []
    inres_lat = inres / 111.  # in degrees of latitude
    ncfile1 = Dataset(indir + infile, 'r')
    lat = ncfile1.variables['LAT'][:, :]
    lon = ncfile1.variables['LON'][:, :]

    var = ncfile1.variables[varname][:, :]

    for index, item in enumerate(lat[:, 0]):
        if np.abs(item - loc_lat) <= inres_lat:
            ilat = ilat + [index]  # store all the indices corresponding to the criteria
            latval = latval + [item]

    # MNH resolution in degrees of longitude
    inres_lon = inres_lat / np.cos(np.max(latval) * pi / 180.)
    for index, item in enumerate(lon[0, :]):
        if np.abs(item - loc_lon) <= inres_lon:
            ilon = ilon + [index]  # store all the indices corresponding to the criteria
            lonval = lonval + [item]

    vardim = len(ncfile1.variables[varname].shape)
    if inunits == []:
        try:
            varunits = ncfile1.variables[varname].units
        except:
            varunits = 'unk'
            warn('variable %s units are not specified in the netcdf'
                 'file neither in the function call - set to unk' % (varname))
    else:
        varunits = inunits

    if vardim == 4:
        # alt=ncfile1.variables['ALT'][lev,:,:]
        alt = ncfile1.variables['level'][:]
        # NB: not much sense here as altitude is a scalar - TODO put instead the actual altitude AMSL (rather than the level => use dia files)
        altmin = np.nanmin(alt)
        altmax = np.nanmax(alt)
        altavg = np.nanmean(alt)

    else:
        raise ValueError('vardim is %i instead of 4 for variable %s in file %s ' % (vardim, varname, infile))

    var[np.where(var == nan_val)] = np.nan  # or float('nan')

    var_loc = [var[0, :, i, j] for i in ilon for j in ilat]
    var_avg = np.nanmean(var_loc, axis=0)

    return var_avg, alt, varunits


def plot_profile_mnh(infile, indir, allnames, inres, loc_lat, loc_lon, outdir, outname, outftype='ps',
                     colmap='rainbow', colorlev=10, inunits=[],
                     cticks=[], proj='merc', nan_val=999., ymin=0., ymax=20.e3,
                     dlatlabel=4., dlonlabel=4., lev=5, islog=False,
                     xlabel=[], ylabel=[], title=[]):
    ''' loc = lon, lat of the location at which a profile is to be plotted
        inres = model resolution in km '''

    ilat = []
    latval = []
    ilon = []
    lonval = []
    inres_lat = inres / 111.  # in degrees of latitude

    if type(allnames) == str:
        allnames = [allnames]
    ii = 1

    # create one fig for all variables (to be put in subpolots)
    fig1 = plt.figure()

    for varname in allnames:


        var_avg, alt, varunits = get_profile_mnh(infile, indir, varname, inres, loc_lat, loc_lon,
                nan_val, inunits)
        # TODO instead of alt put the actual altitude AMSL (rather than the level => use dia files)


        ax1 = fig1.add_subplot(1, len(allnames), ii)
        im = ax1.plot(var_avg, alt)

        ax1.set_ylim([ymin, ymax])
        if xlabel==[]:
            ax1.set_xlabel(varname+' ('+varunits+')')
        else:
            ax1.set_xlabel(xlabel)

        ax1.set_title(title)
        if ii == 1 and ylabel != [] :
            ax1.set_ylabel(ylabel)

        ii = ii + 1

    # save fig
        fig1.savefig(outdir + '/' + outname + '.' + outftype)
        plt.close(fig1)

def netcdf2geo_map(infile,indir,varname, outdir, outftype = 'ps',
                   colmap='rainbow',colorlev = 10, cmin = np.nan, cmax=np.nan,
                   cticks=[], proj='merc', nan_val=999.,
                   dlatlabel = 4., dlonlabel = 4. , lev = 5, alt_max = 999.e3, lsum = False, islog=False, coordfile = []):
    ''' plot geographical maps of a given quantity (2D or 3D in which case a level should be specified
        ifile:    name of the input netcdf file
        idir:     input directory full path
        outdir:   output directory
        varname:  name of the variable to be plotted
        colmap:   colormap used for plotting
        colorlev: number of color levels (between either min / max of array or passed cmin, cmax values)
                  OR array of values each one correspounding to a color level
        cmin    : min value for colormap
        cmax    : max value for colormap
        islog:    if True a logarithmic colorscale will be used
        lev  :    model level at which the map will be ploted (for 3D variable)
        cticks:   where tick will be put in the colorbar
        dlatlabel: labels on the latitude axis will be put every dlatlabel degrees
        coordfile : full path of the file containing the coordinates LAT and LON, if this is not the input file itself
                  : leave empty if using the input file
        lsum      : if true, variable will be sum along the vertical from the ground to alt_max (m) (or to the highest model level)


         '''
# TODO: quite slow -> Why ?? =>> make it faster
    if np.size(colorlev) == 1 and np.isreal(cmin) and np.isreal(cmax):
        colorlev = np.linspace(cmin, cmax, colorlev)


# TODO plot profile + better projection (?)
    ncfile1 = Dataset(indir+infile,'r')
    if coordfile != []:
        coorddata = Dataset(coordfile)
        lat=coorddata.variables['LAT'][1:-1,1:-1]
        lon=coorddata.variables['LON'][1:-1,1:-1]
    else:
        lat=ncfile1.variables['LAT'][1:-1,1:-1]
        lon=ncfile1.variables['LON'][1:-1,1:-1]
    latmin=np.nanmin(lat)
    latmax=np.nanmax(lat)
    lonmin=np.nanmin(lon)
    lonmax=np.nanmax(lon)
    '''
    print('latmin lat max, lonmin / max',latmin,latmax,lonmin,lonmax)
    print('corners')
    print(lat[0,0], lon[0,0])
    print(lat[-1,0], lon[-1,0])
    print(lat[-1,-1], lon[-1,-1])
    print(lat[0,-1], lon[0,-1])
    '''

    vardim = len(ncfile1.variables[varname].shape)
    try:
        varunits = ncfile1.variables[varname].units
    except:
        varunits = ''

    if vardim == 4:
        # alt=ncfile1.variables['ALT'][lev,:,:]
        alt = ncfile1.variables['level'][1:-1]
        # NB: not much sense here as altitude is a profile - TODO put instead the actual altitude AMSL (rather than the level => use dia files)
        #altmin = np.nanmin(alt)
        altmax = np.nanmax(alt)
        #altavg = np.nanmean(alt)
        if lsum:
            var = ncfile1.variables[varname][0, 1:-1, 1:-1, 1:-1]  # excludes 1st and last value as they are meaningless
            lsumall=False
            if alt_max >= altmax:
                lsumall=True
                var= np.sum(var,axis=0)  # sum over level
            elif alt_max > 0:
                ialt=np.nan
                for index, item in enumerate(alt):
                    if item <= alt_max and item > 0.:
                        ialt = index  # store the largest index satisfying the criteria
                if np.isreal(ialt):
                    var = var[0:ialt+1,:,:]
                    var = np.sum(var,axis=0)
            else:
                print('alt_max =', alt_max)
                raise(ValueError,'invalid altmax value')
        else:
            var=ncfile1.variables[varname][0, lev, 1:-1, 1:-1]
    elif vardim == 3:
        var=ncfile1.variables[varname][0, 1:-1, 1:-1]
    elif vardim == 2:
        var=ncfile1.variables[varname][1:-1, 1:-1]
    else:
        raise ValueError('vardim is %i instead of 2, 3 or 4 for variable %s in file %s ' % (vardim, varname, infile) )
    var[np.where(var==nan_val)]=float('nan')

    m = Basemap(resolution='h',   projection=proj,
                llcrnrlat=lat[0,0], urcrnrlat=lat[-1,-1],
                llcrnrlon=lon[0,0], urcrnrlon=lon[-1,-1])
    m.drawcountries(linewidth=0.4,color = 'grey')
    m.drawcoastlines(linewidth=0.5, color = 'grey')
    m.drawparallels(np.arange(round(latmin,1),round(latmax,1),dlatlabel),labels=[1,0,0,0],
                color='grey',linewidth=0.1,fontsize=8) #,labelstyle='+/-'
    m.drawmeridians(np.arange(round(lonmin,1),round(lonmax,1),dlonlabel),labels=[0,0,0,1],rotation=45,
                color='grey',linewidth=0.1,fontsize=8) #,labelstyle='+/-'


#cticks = [290,295,300,305,310,315,320,325,330,335,340,345,350,355,360]

    x,y = m(lon,lat)
    if not islog:
        cs = m.contourf(x,y,var,
                levels = colorlev
                ,cmap=get_cmap(colmap),extend='both')
    else:
        cs = m.contourf(x, y, var, norm=colors.LogNorm(),
                        levels = colorlev,
                        cmap=get_cmap(colmap), extend='both') # extend changed from neither to both
    if len(cticks) > 2:
        cb=plt.colorbar(cs,ticks=cticks)
    else:
        cb=plt.colorbar(cs)


    if vardim <= 3:
        plt.title(varname+' '+infile[:-3])
        nom_fig=varname+infile[:-3]
    elif not lsum:
        plt.title(varname+' l '+str(lev)+ ' ( ~'+ str(int(round(alt[lev]))) +'m ) '+infile[:-3])
        nom_fig=varname+'_'+infile[:-3]+'_lev'+str(lev)
    else:
        if lsumall:
            plt.title(varname + '(vertically summed) ' + infile[:-3])
            nom_fig=varname+'_'+infile[:-3]+'_vertsum'
        else:
            plt.title(varname + '(summed up to ~'+str(int(round(alt_max)))+'m height)' + infile[:-3])
            nom_fig=varname+'_'+infile[:-3]+'_vertsum'+str(int(round(alt_max)))


    cb.set_label(varunits, labelpad=-40, y=1.05, rotation=0)

    plt.savefig(outdir+nom_fig+'.'+outftype,format=outftype, bbox_inches = 'tight',
    pad_inches = 0)
    plt.close()


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
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cft

from math import sqrt
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

    if nbplt ==1:
        return bin_centers, hist



# --------*********************************************************---------------------------------------------- #

def plot_2D_colormap(x, y, array2D,
                out_name, cnorm, out_path_fig, out_type='pdf',
                lyinvert=True, cmap='gist_ncar',
                title='', xlabel='', ylabel='', ymax=[], ymin=[]):
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
    if ymax!=[] and ymin!=[]:
        ax1.set_ylim([ymin,ymax])

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

def get_indices_coords(lon,lat,loc_lat,loc_lon, inres,lclosest):
    ''' return the indices ilon and ilat corresponding to all the points less than inres_lat degrees away in latitude,
        and the corresponding distance in longitude; if lcolsest only the closest point is returned
        used in get_point_mnh and get_profile_mnh
         input :
         loc_lat, loc_lon : coordinates of the point to be searched for in lon,lat
         inres_lat : resolution of the input latitude and longitudes (in degrees)

         '''
    ilat = []
    ilon = []
    latval = []
    lonval = []
    if lclosest:
        ddlat = []
        ddlon =[]

    inres_lat = inres / 111.  # in degrees of latitude

    for index, item in enumerate(lat[:, 0]):
        dlat = np.abs(item - loc_lat)
        if dlat <= inres_lat:
            ilat = ilat + [index]  # store all the indices corresponding to the criteria
            latval = latval + [item]
            if lclosest:
                ddlat= ddlat + [dlat]

    # MNH resolution in degrees of longitude
    inres_lon = inres_lat / np.cos(np.max(latval) * pi / 180.)
    for index, item in enumerate(lon[0, :]):
        dlon = np.abs(item - loc_lon)
        if dlon <= inres_lon:
            ilon = ilon + [index]  # store all the indices corresponding to the criteria
            lonval = lonval + [item]
            if lclosest:
                ddlon= ddlon + [dlon]

    # minimum distance
    if lclosest:
        #TODO: make it faster by searching only in a subset of the array and hence avoiding useless calculations
        # replace multiple value to average by single value to use
        ddist = np.array([( (lon[0, i] - loc_lon ) * np.cos(np.mean(lat[i,j]) * pi / 180.) )**2  +
                          ( (lat[j, 0] - loc_lat) )**2  for i in ilon for j in ilat])
        ii, jj = np.unravel_index(ddist.argmin(), (len(ilon), len(ilat)))
        # correspounding lon and lat
        #llon = lon[ii, jj]
        #llat = lat[ii, jj]
        # replace position only by the closest
        ilon = ilon[ii]
        ilat = ilat[jj]

    return ilon,ilat#,llon,llat


def get_profile_mnh(infile, indir, varnamelist, inres, loc_lat, loc_lon,
                    nan_val=999., inunits=[], lclosest=False, alt_file=[]):
    ### !!!! THERE WAS A MISTAKE : ALL THE PLOTS MADE  BEFORE 4 JULY 2019 were WRONG (indices were swaped)
    ''' function to get the average profile from the nearest model gridcolumns
     the return array is the average of all the model gridcolumn that are less than inres km away from (loc_lat,loc_lon)
     infile : input file name
     indir  : input directory full path
     varnamelist: name of the variable to retrieve or the list of variable names (e.g. ['var1', 'var2'] )
     inres  : resolution of the input data (km)
     loc_lat: latitude of desired location for the profiles
     loc_lon: longitude of ...
     nan_val: value of NaNs in input file
     inunits: units can be specified here, otherwise it will be read in netcdf file and if absent "unk" will be used
     lclosest : if true returns only the profile closest to the specified location (rather than the average of the closest profiles)
     alt_file : name of the file containing the altitudes AMSL
     nbvar = number of variables
    '''


    if isinstance(varnamelist,str):
        varnamelist=[varnamelist]

    print("reading "+indir + infile)
    ncfile1 = Dataset(indir + infile, 'r')
    try:
        lat = ncfile1.variables['LAT'][:, :]
        lon = ncfile1.variables['LON'][:, :]
    except:
        lat = ncfile1.variables['latitude'][:, :]
        lon = ncfile1.variables['longitude'][:, :]
    time = ncfile1.variables['time']

    ilon,ilat = get_indices_coords(lon,lat,loc_lat,loc_lon, inres, lclosest)
    varunits_all = []
    nbvar = len(varnamelist)
    nvar=0

    for varname in varnamelist:
        if nvar == 0 :
            nblev= len(ncfile1.variables[varname][0, :, 0, 0])
            var_avg_all = np.zeros([nbvar,nblev])

        var = ncfile1.variables[varname][:, :, :, :]

        vardim = len(ncfile1.variables[varname].shape)
        if inunits == []:
            try:
                varunits = ncfile1.variables[varname].units
            except:
                varunits = 'unk'
                warnings.warn('variable %s units are not specified in the netcdf'
                     'file neither in the function call - set to unk' % (varname))
        else:
            varunits = inunits

        if alt_file != []:
            alt_nc = Dataset(alt_file,'r')
            alt=alt_nc.variables['ALT'][:,ilat,ilon] ## double check order
        elif vardim == 4:
            # alt=ncfile1.variables['ALT'][lev,:,:]
            alt = ncfile1.variables['level'][:]
            # NB: not much sense here as altitude is a scalar - TODO put instead the actual altitude AMSL (rather than the level => use dia files)
            #altmin = np.nanmin(alt)
            #altmax = np.nanmax(alt)
            #altavg = np.nanmean(alt)
        else:
            raise ValueError('vardim is %i instead of 4 for variable %s in file %s ' % (vardim, varname, infile))

        if ( np.size(np.where(var == nan_val)) > 0 ):
            var[np.where(var == nan_val)] = np.nan  # or float('nan')

        if not lclosest:
            var_loc = [var[0, :, i, j] for i in ilat for j in ilon]
            var_avg = np.nanmean(var_loc, axis=0)
        else:
            var_avg = var[0, :, ilat,ilon]

        varunits_all.append(varunits)
        var_avg_all[nvar,:]=var_avg
        nvar+=1

    return var_avg_all, alt, varunits_all, time,ilon,ilat, lon[ilat,ilon], lat[ilat,ilon]

def get_point_mnh(infile, indir, varname, inres, loc_lat, loc_lon,
                    nan_val=999., inunits=[], lclosest=False):
    # order ilat,ilon corrected on 5th July 2019
    ''' function to get the value of a vriable at a given lon-lat point by averaging MNH neighbooring gridpoints
     the return array is the average of all the model gridcolumn that are less than inres km away from (loc_lat,loc_lon)
     infile : input file name
     indir  : input directory full path
     varname: name of tyhe variable to retrieve
     inres  : resolution of the input data (km)
     loc_lat: latitude of desired location for the profiles
     loc_lon: longitude of ...
     nan_val: value of NaNs in input file
     inunits: units can be specified here, otherwise it will be read in netcdf file and if absent "unk" will be used
     lclosest : if True no averaging is done and the closest location is used instrad
     '''

    ncfile1 = Dataset(indir + infile, 'r')
    lat = ncfile1.variables['LAT'][:, :]
    lon = ncfile1.variables['LON'][:, :]
    time = ncfile1.variables['time']
    var = ncfile1.variables[varname][:, :,:]

    ilon,ilat = get_indices_coords(lon,lat,loc_lat,loc_lon, inres, lclosest)


    vardim = len(ncfile1.variables[varname].shape)
    if inunits == []:
        try:
            varunits = ncfile1.variables[varname].units
        except:
            varunits = 'unk'
            warnings.warn('variable %s units are not specified in the netcdf'
                 'file neither in the function call - set to unk' % (varname))
    else:
        varunits = inunits


    if vardim != 3 :
        raise ValueError('vardim is %i instead of 4 for variable %s in file %s ' % (vardim, varname, infile))

    var[np.where(var == nan_val)] = np.nan  # or float('nan')
    if not lclosest:
        var_loc = [var[0, i, j] for i in ilat for j in ilon]
        var_avg = np.nanmean(var_loc, axis=0)
    else:
        var_avg = var[0, ilat, ilon]

    return var_avg,varunits, time, ilon, ilat


def plot_profile_mnh(infile, indir, allnames, inres, loc_lat, loc_lon, outdir, outname, outftype='ps',
                     colmap='rainbow', colorlev=10, inunits=[],
                     cticks=[], proj='merc', nan_val=999., ymin=0., ymax=20.e3,
                     dlatlabel=4., dlonlabel=4., lev=5, islog=False,
                     xlabel=[], ylabel=[], title=[]):
    ''' loc = lon, lat of the location at which a profile is to be plotted
        inres = model resolution in km
        outname = output file name
        '''

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


        var_avg, alt, varunits = get_profile_mnh(infile, indir, varname, inres, loc_lat, loc_lon, nan_val, inunits)
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
                   colmap='rainbow',colorlev = 10, cmin = [], cmax=[],
                   cticks=[], proj='merc', nan_val=999.,
                   dlatlabel = 4., dlonlabel = 4. , lev = 21, alt_max = 999.e3, lsum = False, islog=False, coordfile = [],
                   ladd_arrow_wind=False, windfile=[], LSwind=False, cmapextend = 'both', extravar_contour=[], cmap_extra='Greys'):
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
        lswind    : to use the large-scale wind (coming from the coupling files) rather than the actueal model wind
        cmapextend: extend the colorbar both direction beyond the colorbar limits (NB : no available with log colorscale)
         '''
# TODO: quite slow -> Why ?? =>> make it faster
    lcollev = False
    if np.size(colorlev) == 1 and cmin!=[] and cmax!=[]:
        colorlev = np.linspace(cmin, cmax, colorlev)
        lcollev = True
    elif np.size(colorlev) > 1:
        lcollev = True

# TODO plot profile + better projection (?)
    ncfile1 = Dataset(indir+infile,'r')
    if coordfile != []:
        coorddata = Dataset(coordfile)
        lat=coorddata.variables['LAT'][1:-1,1:-1]
        lon=coorddata.variables['LON'][1:-1,1:-1]
    else:
        try:
            lat=ncfile1.variables['LAT'][1:-1,1:-1]
            lon=ncfile1.variables['LON'][1:-1,1:-1]
        except:
            lat=ncfile1.variables['latitude'][1:-1,1:-1]
            lon=ncfile1.variables['longitude'][1:-1,1:-1]

    latmin=np.nanmin(lat)
    latmax=np.nanmax(lat)
    lonmin=np.nanmin(lon)
    lonmax=np.nanmax(lon)

    # lat and lon centre points (used for projection)
    latcen = np.mean(lat[:,0])
    loncen = np.mean(lon[0,:])
    '''
    print('latmin lat max, lonmin / max',latmin,latmax,lonmin,lonmax)
    print('corners')
    print(lat[0,0], lon[0,0])
    print(lat[-1,0], lon[-1,0])
    print(lat[-1,-1], lon[-1,-1])
    print(lat[0,-1], lon[0,-1])
    '''
    try :
        vardim = len(ncfile1.variables[varname].shape)
    except:
        vardim = 4 # for windspeed case, where varname cannot be read directly
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
            if varname=='windspeed':
                var= np.sqrt(ncfile1.variables['UT'][0, lev+1, 1:-1, 1:-1]** 2 +
                 ncfile1.variables['VT'][0, lev+1, 1:-1, 1:-1]**2)
            elif varname == 'LSwindspeed':
                var= np.sqrt(ncfile1.variables['LSUM'][0, lev+1, 1:-1, 1:-1]**2 +
                 ncfile1.variables['LSVM'][0, lev+1, 1:-1, 1:-1]**2)
            elif varname == 'winddiff':
                pass
            else:
                var= ncfile1.variables[varname][0, lev+1, 1:-1, 1:-1]

    elif vardim == 3:
        var=ncfile1.variables[varname][0, 1:-1, 1:-1]
    elif vardim == 2:
        var=ncfile1.variables[varname][1:-1, 1:-1]
    else:
        raise ValueError('vardim is %i instead of 2, 3 or 4 for variable %s in file %s ' % (vardim, varname, infile) )

    if ladd_arrow_wind:
        if varname != 'winddiff':
            if LSwind:
                uname = 'LSUM'
                vname = 'LSVM'
            else:
                uname = 'UT'
                vname='VT'

            if windfile == []:
                UU = ncfile1.variables[uname][0, lev+1, 1:-1, 1:-1]
                VV = ncfile1.variables[vname][0, lev+1, 1:-1, 1:-1]
            else:
                ncfilew=Dataset(windfile,'r')
                UU = ncfilew.variables[uname][0, lev+1, 1:-1, 1:-1]
                VV = ncfilew.variables[vname][0, lev+1, 1:-1, 1:-1]
        else:
            if windfile == []:
                UU = ncfile1.variables['UT'][0, lev+1, 1:-1, 1:-1] - ncfile1.variables['LSUM'][0, lev+1, 1:-1, 1:-1]
                VV = ncfile1.variables['VT'][0, lev+1, 1:-1, 1:-1] - ncfile1.variables['LSVM'][0, lev+1, 1:-1, 1:-1]
            else:
                ncfilew=Dataset(windfile,'r')
                UU = ncfilew.variables['UT'][0, lev+1, 1:-1, 1:-1] - ncfilew.variables['LSUM'][0, lev+1, 1:-1, 1:-1]
                VV = ncfilew.variables['VT'][0, lev+1, 1:-1, 1:-1] - ncfilew.variables['LSVM'][0, lev+1, 1:-1, 1:-1]
            var = np.sqrt(UU**2 + VV**2)





## TODO replace by cartopy as basemap is deprecated
## as the lon - lat coordinates are calculated, the choice of projection is free - although Mercator with PGD central points is used in MNH
## (actually still quite surprisefd of this behaviour and having square-gridcolumn in DEGREES meaning resolution is not constant in km, I would have assumed the exact opposite!)
    m = Basemap(resolution='h',   projection=proj,
                llcrnrlat=lat[0,0], urcrnrlat=lat[-1,-1],
                llcrnrlon=lon[0,0], urcrnrlon=lon[-1,-1])
    m.drawcountries(linewidth=0.4,color = 'grey')
    m.drawcoastlines(linewidth=0.5, color = 'grey')
    m.drawparallels(np.arange(round(latmin,1),round(latmax,1),dlatlabel),labels=[1,0,0,0],
                color='grey',linewidth=0.1,fontsize=8) #,labelstyle='+/-'
    m.drawmeridians(np.arange(round(lonmin,1),round(lonmax,1),dlonlabel),labels=[0,0,0,1],rotation=45,
                color='grey',linewidth=0.1,fontsize=8) #,labelstyle='+/-'


    '''
    # watch updates of gridliner == not sure it supports rotated labels ...
    proj = cartopy.crs.Mercator()
    ax = plt.axes(projection=proj)
    
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
    ax.set_extent([lonmin, lonmax, latmin, latmax])
    ax.gridlines(crs=proj, draw_labels=True)
    
    gl = ax.gridlines(crs=proj, draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(round(-18.65),round(57.23),dlatlabel))
    gl.xlabels_top = False
    gl.ylabels_right = False
    # plt.xticks(rotation='45') ## doesn't work here, need to apply rotation on grilines gl
    
    '''


    #cticks = [290,295,300,305,310,315,320,325,330,335,340,345,350,355,360]

    x,y = m(lon,lat)
    if not islog:
        if lcollev:
            cs = m.contourf(x,y,var,
                levels = colorlev,
                cmap=get_cmap(colmap),extend=cmapextend)
        else:
            cs = m.contourf(x,y,var,
            cmap=get_cmap(colmap))
    else:
        if lcollev:
            cs = m.contourf(x, y, var, norm=colors.LogNorm(),
                        levels = colorlev,
                        cmap=get_cmap(colmap) ) #, extend=cmapextend)
        else:
            cs = m.contourf(x, y, var, norm=colors.LogNorm(),
            cmap=get_cmap(colmap)) # extend changed from neither to both

    if len(cticks) > 2:
        cb=plt.colorbar(cs,ticks=cticks)
    else:
        cb=plt.colorbar(cs)

    if not islog:
        log_tag = ''
    else:
        log_tag = '_log'

    if vardim <= 3:
        plt.title(varname+' '+infile[:-3])
        nom_fig=varname+'_'+infile[:-3] + log_tag
    elif not lsum:
        plt.title(varname+' l '+str(lev)+ ' ( ~'+ str(int(round(alt[lev]))) +'m ) '+infile[:-3])
        nom_fig=varname+'_'+infile[:-3]+'_lev'+str(lev) + log_tag
    else:
        if lsumall:
            plt.title(varname + '(vertically summed) ' + infile[:-3])
            nom_fig=varname+'_'+infile[:-3]+'_vertsum' + log_tag
        else:
            plt.title(varname + '(summed up to ~'+str(int(round(alt_max)))+'m height)' + infile[:-3])
            nom_fig=varname+'_'+infile[:-3]+'_vertsum'+str(int(round(alt_max))) +log_tag


    cb.set_label(varunits, labelpad=-40, y=1.05, rotation=0)


    if ladd_arrow_wind:
        m.quiver(x[0::10,0::10],y[0::10,0::10], UU[0::10,0::10], VV[0::10,0::10])

    if extravar_contour != []:
        m.contour(x,y,extravar_contour, cmap=get_cmap(colmap) )

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plt.savefig(outdir+'/'+nom_fig+'.'+outftype,format=outftype, bbox_inches = 'tight',
    pad_inches = 0)
    plt.close()


def var2map(var1,lon1,lat1,  day, fig_title='',maplimits=[-18.,38.,-36.,10.],
            out_path_fig='/home/labl/Bureau/', out_name='test', out_type='png'):
    # maplimits = [lonmin, lonmax, latmin, latmax]
    dlonlabel=10. # label latitude every 10 degrees
    #minlon=-18.65
    #maxlon=57.23

    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)


    # plot  -- try cartopy
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    ax.add_feature(cft.BORDERS, linestyle='-', alpha=.5)



    gl.ylabels_right = False
    gl.xlabels_top   = False
    '''
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(round(maplimits[0]),round(maplimits[1]),dlonlabel))
    '''

    cp = plt.contourf(lon1, lat1, var1,
                      transform=ccrs.PlateCarree(),
                      cmap=get_cmap('gnuplot') ,vmin=10., vmax=3210, levels= np.linspace(10, 3210, 21), alpha=0.92)
    plt.colorbar(cp)

    # for some reason color keyword doesn't work
    '''cp2 = plt.contourf(lon_flag, lat_flag, conv_flag[:, :, dday - stday], transform=ccrs.PlateCarree(),
                       colors = 'w', levels = [1,24], alpha=0.35)#cmap=get_cmap('Greys'), levels=[1, 5, 10])
    '''

    ax.coastlines()
    ax.add_feature(cft.LAKES, alpha=0.8)
    ax.add_feature(cft.OCEAN, alpha=0.8)
    ax.set_extent(maplimits, crs=ccrs.PlateCarree())
    plt.title(fig_title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    fig.savefig(out_path_fig + '/' + out_name +'_'+day+'SEP.' + out_type)
    plt.close(fig)


### a TESTER / FINIR (21 Aout 2019)
def tracer_vert_int(tracer_file,tracer_name,day,alt_file):
    svt_nc=Dataset(tracer_file,'r')
    svt=svt_nc.variables[tracer_name][0, 1:-1, 1:-1, 1:-1]
    #lon = svt_nc.variables['LON'][0, 1:-1, 1:-1, 1:-1] # LON
    #lat = svt_nc.variables['LAT'][0, 1:-1, 1:-1, 1:-1] # LAT
    lon=svt_nc.variables['longitude'][1:-1, 1:-1] # LON
    lat = svt_nc.variables['latitude'][1:-1, 1:-1] # LON
    #try:
    #    rho=svt_nc.variables['RHOREFZ'][0, 1:-1, 1:-1, 1:-1]
    #except:
        #temp = svt_nc.variables['TEMP'][0, 1:-1, 1:-1, 1:-1] + 273.15
    tht = svt_nc.variables['THT'][0, 1:-1, 1:-1, 1:-1]
    P   = svt_nc.variables['PABST'][0, 1:-1, 1:-1, 1:-1]  # ou PRESS * 100.
    Ra  = 287.058
    rho = P / ( ( tht/(100000./P)**0.286 ) * Ra)
    alt_nc = Dataset(alt_file, 'r')
    dalt = alt_nc.variables['ALT'][2:, 1:-1, 1:-1] - alt_nc.variables['ALT'][ 1:-1, 1:-1, 1:-1]
    svt_i = rho * dalt * svt

    svt_int = np.sum(svt_i,axis=0)
    threshold = 10.
    svt_int[np.where(svt_int < threshold)] = np.NaN
    print(np.shape(svt_int))

    # plot tracer
    var2map(svt_int, lon, lat, day, fig_title='', maplimits=[-18., 38., -36., 10.],
            out_path_fig='/home/labl/Bureau/', out_name='test', out_type='png')

    '''plot_2D_colormap(svt_int, alt, tracer_all,
                out_name='passive_tracer_MNH'+exp+'vol6', cnorm=norm, out_path_fig='/home/labl/Bureau/', out_type='png',
                lyinvert=False, cmap='jet',
                title=exp+' passive tracer', xlabel='Time (days in year 2017)', ylabel='Altitude (km)',ymax = 10., ymin=0.)'''


    return svt_int


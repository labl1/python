#!/usr/bin/python3
# or use specific anaconda3 environment
# various_utils
# module containing various usefull functions 
# L. Labbouz, July 2018

import numpy as np 
from matplotlib import pyplot as plt

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

    nbplt=0
    for array in array_tuple:
        hist, bins = np.histogram(array, range=(hrange[0],hrange[1]),bins=nbins, normed=lnorm)
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

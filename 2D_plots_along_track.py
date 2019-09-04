# this python pscrpt reads from a netcdf file containing the along-tract fields and generates the correspounding 2D graphs
import numpy as np
from various_utils import plot_2D_colormap
from netCDF4 import Dataset
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import ticker, cm, colors
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap as get_cmap
import argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="input filename")
parser.add_argument("expname", type=str, help="meso-NH experiment name")
parser.add_argument("flight_number", type=str, help="AEROCLO-sA flight number")

args = parser.parse_args()

exp = args.expname

filename=args.filename
 '''
def plot_2D_along_LNG(x, y, array2D, ice, cloud,
                out_name, nnorm, out_path_fig, out_type='pdf',
                lyinvert=True, ccmap='gist_ncar', dust=[],
                title='', xlabel='', ylabel='', ymax=[], ymin=[], lpltLNG=False, LNGarray2D=[]):
    fig1 = plt.figure()
    if not lpltLNG:
        ax1 = fig1.add_subplot(1, 1, 1)
    else:
        ax1 = fig1.add_subplot(2, 1, 1)

    im = ax1.pcolor(x, y, array2D, cmap=ccmap, norm=nnorm)
    if lyinvert:
        ax1.invert_yaxis()
    if ymax != [] and ymin != []:
        ax1.set_ylim([ymin, ymax])
    # add extra contours for clpouds and dust for instance
    CS1 = ax1.contour(time_mnh_all_2D, alt, cloud, colors='blue', levels=[1.e-5, 1.E-4, 5.e-4])
    CS2 = ax1.contour(time_mnh_all_2D, alt, ice, colors='white', levels=[1.e-5, 1.E-4, 5.e-4])
    ## CS3 = ax1.contour(time_mnh_all_2D, alt, dust, colors='yellow', levels=[1.e-5, 1.E-4, 5.e-4])


    ## from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html#sphx-glr-gallery-images-contours-and-fields-contour-label-demo-py
    # Define a class that forces representation of float to look a certain way
    # This remove trailing zero so '1.0' becomes '1'
    '''
    class nf(float):
        def __repr__(self):
            s = f'{self:.2E}'
            return f'{self:.0E}' if s[-1] == '0' else s

    # Recast levels to new class
    CS1.levels = [nf(val) for val in CS1.levels]
    CS2.levels = [nf(val) for val in CS2.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%%'
    else:
        fmt = '%r %%'
    '''
    ax1.clabel(CS1, CS1.levels, inline=True, fmt='%1.E', fontsize=10)
    ax1.clabel(CS2, CS2.levels, inline=True, fmt='%1.E',fontsize=10)

    fig1.colorbar(im, ax=ax1)
    ax1.set_xlabel(xlabel)
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    if lpltLNG:
        ax2 = fig1.add_subplot(2, 1, 2)
        im2 = ax2.pcolor(x, y, LNGarray2D, cmap=ccmap, norm=nnorm)

    fig1.savefig(out_path_fig + '/' + out_name + '.' + out_type)
    plt.close(fig1)



lyinvert = False
ymin=[]
ymax=[]
norm = mpl.colors.Normalize(vmin=0.,vmax=0.15)

exp = 'XA54b'
filename = '/home/labl/Bureau/20170905_Vol6_MNH_'+exp+'_pTracer.nc'
ncfile = Dataset(filename)  #'/home/labl/Bureau/20170905_Vol6_MNH_'+exp+'_pTracer.nc')
nb_vert_lev = 65
time_mnh_all = ncfile.variables['time'][:]
alt = ncfile.variables['altitude'][:,:] / 1000.
tracer_all = ncfile.variables['SVT001'][:,:]
tracer_all[np.where(tracer_all <= 0)] = np.NaN
rct = ncfile.variables['RCT'][:,:]
rit = ncfile.variables['RIT'][:,:]

DSTM33T = ncfile.variables['DSTM33T'][:,:]
DSTM32T = ncfile.variables['DSTM32T'][:,:]
DSTM31T = ncfile.variables['DSTM31T'][:,:]
ABC_1064nm = ncfile.variables['ABC_1064nm'][:,:]
DSTMtot = DSTM33T + DSTM32T + DSTM31T

#"RCT","RIT"] #,"DSTM33T","DSTM32T","DSTM31T"]
time_mnh_all_2D = np.transpose(np.array([time_mnh_all for x in range(nb_vert_lev)]))

plot_2D_along_LNG(time_mnh_all_2D, alt, tracer_all, rit,rct, dust=DSTMtot, lpltLNG=True, LNGarray2D=ABC_1064nm,
                out_name='passive_tracer_MNH'+exp+'vol6', nnorm=norm, out_path_fig='/home/labl/Bureau/', out_type='png',
                lyinvert=False, ccmap='gist_heat',
                title=exp+' passive tracer + rit (blue) and rct (black)', xlabel='Time (days in year 2017)', ylabel='Altitude (km)',ymax = 10., ymin=0.)
'''
norm = mpl.colors.Normalize(vmin=0.,vmax=3)

 
ABC_1064nm = ncfile.variables['ABC_1064nm'][:,:]
alt_lng =  ncfile.variables['altitude_LNG'][:]

alt_lng_2d, time_mnh_all_2 = np.meshgrid(alt_lng, time_mnh_all)

print(np.shape(time_mnh_all_2))
print(np.shape(alt_lng_2d))
print(np.shape(time_mnh_all_2D))

plot_2D_colormap(time_mnh_all_2, alt_lng_2d, ABC_1064nm,
                out_name='LNG_ABC1064_vol6', cnorm=norm, out_path_fig='/home/labl/Bureau/', out_type='png',
                lyinvert=False, cmap='jet',
                title='LNG Attenuated Backscatter passive tracer', xlabel='Time (days in year 2017)', ylabel='Altitude (km)', ymin=0.,ymax=10.)
'''
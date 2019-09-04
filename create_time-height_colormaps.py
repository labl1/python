''' program to create time-hight 2D plots of a given variable, at a given locatiopn, from MNH outputs'''

import numpy as np
from various_utils import get_profile_mnh
import matplotlib.pyplot as plt

site = 'HBAO'

indir = '/home/labl/Bureau/BG54b_12km/'
outdir = '/home/labl/Bureau/BG54b_12km/'
inres = 12.
varname = 'MRC'
filelist = ['BG54b.1.SEP02.0'+str(i).zfill(2)+'dia_all_selected.nc' for i in range(1,10)]
#filelist = ['BG54b.1.SEP'+str(dd).zfill(2)+'.0'+str(i).zfill(2)+'dia_all_selected.nc' for dd in range(1,14) for i in range(1,25) ]
nblev = 67 # number of actual vertiocal levels (excluding padding levels)
nbprofiles = len(filelist)

outname = varname '_' + site + '_BG54b.1.SEP01-14'
outftype = 'ps'

loc_lat_WBA = -22.979722
loc_lon_WBA = 14.645278
loc_lat_HBAO = -22.096247
loc_lon_HBAO = 14.259694

# get the dimension of the array for the colormap

var2D  = np.empty([nbprofiles,nblev])
timed  = np.empty(nbprofiles) # time in days

# get the data and fill the arrays
i=0
for infile in filelist:
    var_avg, alt, varunits, times = get_profile_mnh(infile, indir, varname, inres, loc_lat_HBAO, loc_lon_HBAO, inunits = ' g/kg ')

    var2D [i,:] = var_avg
    timed[i]  =  1. + times[:] / 3600. / 24.
    #timed [i] = 1. + times / 3600. / 24. # Time (days from 31/09/2017 00UTC)
    i+=1

alt = alt/1000.
# plot the 2D colormap --- (here or calling a fct if likely usefull elsewhere)
fig = plt.figure()
im = plt.pcolor(timed, alt,np.transpose(var2D), cmap='gnuplot2_r') # pink_r
ax = fig.gca()
ax.set_ylim([0., 7.])
ax.set_xlabel('Time (days since 31/8/2017 00UTC)')
plt.colorbar()
ax.set_ylabel('Altitude (km)')
ax.set_title('MNH '+varname+' at '+site)
# save
fig.savefig(outdir + '/' + outname + '.' + outftype)
plt.close(fig)
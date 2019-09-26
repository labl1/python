# program to plot the end points of trajectories + the trajectories ending there

# read longitude and latitude

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker, cm, colors
from matplotlib.cm import get_cmap as get_cmap
from netCDF4 import Dataset
from various_utils import var2map
import glob
import cartopy.crs as ccrs
import cartopy.feature as cft

indir = '/home/labl/Bureau/traj_stratocu_SEP06_004/'
infiles = glob.glob(indir+"traj_*.nc") ##
day="SEP06_004"
lmin=1
lmax=24
proj = ccrs.PlateCarree()
fig = plt.figure()
ax = plt.axes(projection=proj)
ax.coastlines()
ax.add_feature(cft.LAKES, alpha=0.8)
ax.add_feature(cft.OCEAN, alpha=0.8)

fig_title = 'Back-Traj ending over Sc region at Z<3km SEP06 12UTC'
maplimits = [-18., 38., -36., 10.]
ax.set_extent(maplimits, crs=proj)
plt.title(fig_title)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
ax.add_feature(cft.BORDERS, linestyle='-', alpha=.5)

gl.ylabels_right = False
gl.xlabels_top = False

for infile in infiles:
    data = Dataset(infile, 'r')
    print(infile)
    lon = data.variables['LON'][0,lmin:lmax,:]
    lat = data.variables['LAT'][0,lmin:lmax,:]
    alt = data.variables['Z'][0,lmin:lmax,:]
    alt[np.where(alt > 99.)]=np.nan
    alt[np.where(alt < -99.)]=np.nan
    lon[np.where(lon > 999. )]=np.nan
    lat[np.where(lat < -999.)]=np.nan
    lon[np.where(lon < -999. )]=np.nan
    lat[np.where(lat > 999.)]=np.nan

    var1=alt
    lon1=lon
    lat1=lat

    out_path_fig='/home/labl/Bureau/traj_stratocu_SEP06_004/FIG/'
    out_name= 'test' #os.path.basename(infile[0:-3])
    out_type='png'
    vvmin=0.
    vvmax=5.

    cp = ax.scatter(lon1, lat1, 1.5,
                      transform=ccrs.PlateCarree(),
                      cmap=get_cmap('jet'), c=var1, alpha=0.40,vmin=vvmin, vmax=vvmax)


cb=plt.colorbar(cp)
cb.set_label("Alt (km)", labelpad=-10, y=1.05, rotation=0)
fig.savefig(out_path_fig + '/' + out_name + '_' + day + 'SEP.' + out_type, dpi=500)

plt.close(fig)

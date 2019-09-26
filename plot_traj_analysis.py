# to plot the results from the trajectory analysis performed by traj_analysis

from various_utils import netcdf2geo_map
from numpy import linspace as linspace

indir = '/home/labl/Bureau/MNH_simulations/delta_Z_all_traj/'
outdir = '/home/labl/Bureau/'

varname = 'frac_dzup3km'
ccmin = 0.

level_liste = [15,20,25] #[30,33,36,40, 44]

ccmaxlist= [3, 3, 3]#[6.,12.,24.,48., 48]

i=0
for levnb in level_liste:
    ccmax=ccmaxlist[i]
    i=i+1
    varname = 'frac_dzup3km'
# XA540trajZ.nc
# XA54btrajZ.nc
# XA540-b_diff_trajZ.nc
    infile = 'XA540trajZ.nc'

    netcdf2geo_map(infile,indir,varname, outdir, outftype = 'png',
                   colmap='rainbow',colorlev = 13, cmin = 0.01, cmax=ccmax,
                   cticks=linspace(0,ccmax,13), proj='merc', nan_val=9.99E9,
                   dlatlabel = 10, dlonlabel = 10 , lev = levnb, llev_cleaned=True, alt_max = 15., lsum = False, islog=False, coordfile = [],
                   ladd_arrow_wind=False, windfile=[], LSwind=False, cmapextend = 'neither', extravar_contour=[], cmap_extra='Greys')

    infile = 'XA54btrajZ.nc'

# XA540trajZ.nc
# XA54btrajZ.nc
# XA540-b_diff_trajZ.nc

    netcdf2geo_map(infile,indir,varname, outdir, outftype = 'png',
                   colmap='rainbow',colorlev = 13, cmin = 0.01, cmax=ccmax,
                   cticks=linspace(0,ccmax,13), proj='merc', nan_val=9.99e9,
                   dlatlabel = 10, dlonlabel = 10 , lev = levnb, llev_cleaned=True, alt_max = 15., lsum = False, islog=False, coordfile = [],
                   ladd_arrow_wind=False, windfile=[], LSwind=False, cmapextend = 'neither', extravar_contour=[], cmap_extra='Greys')


    infile = 'XA540-b_diff_trajZ.nc'
    cfile = indir+'XA54btrajZ.nc' # for lon / lat vaules
    netcdf2geo_map(infile,indir,varname, outdir, outftype = 'png',
                   colmap='seismic',colorlev = 26, cmin = -ccmax, cmax=ccmax,
                   cticks=linspace(-ccmax,ccmax,13), proj='merc', nan_val=9.99e9,llev_cleaned=True,
                   dlatlabel = 10, dlonlabel = 10 , lev = levnb, alt_max = 15., lsum = False, islog=False, coordfile = cfile,
                   ladd_arrow_wind=False, windfile=[], LSwind=False, cmapextend = 'max', extravar_contour=[], colorbar_units='% pts')



###############################
    varname = 'frac_dzup6km'
    ccmax=ccmax/3.
    infile = 'XA540trajZ.nc'

    netcdf2geo_map(infile,indir,varname, outdir, outftype = 'png',
                   colmap='rainbow',colorlev = 13, cmin = 0.01, cmax=ccmax,
                   cticks=linspace(0,ccmax,13), proj='merc', nan_val=9.99E9,
                   dlatlabel = 10, dlonlabel = 10 , lev = levnb, llev_cleaned=True, alt_max = 15., lsum = False, islog=False, coordfile = [],
                   ladd_arrow_wind=False, windfile=[], LSwind=False, cmapextend = 'neither', extravar_contour=[], cmap_extra='Greys')

    infile = 'XA54btrajZ.nc'

# XA540trajZ.nc
# XA54btrajZ.nc
# XA540-b_diff_trajZ.nc

    netcdf2geo_map(infile,indir,varname, outdir, outftype = 'png',
                   colmap='rainbow',colorlev = 13, cmin = 0.01, cmax=ccmax,
                   cticks=linspace(0,ccmax,13), proj='merc', nan_val=9.99e9,
                   dlatlabel = 10, dlonlabel = 10 , lev = levnb, llev_cleaned=True, alt_max = 15., lsum = False, islog=False, coordfile = [],
                   ladd_arrow_wind=False, windfile=[], LSwind=False, cmapextend = 'neither', extravar_contour=[], cmap_extra='Greys')
    

    infile = 'XA540-b_diff_trajZ.nc'
    cfile = indir+'XA54btrajZ.nc' # for lon / lat vaules
    netcdf2geo_map(infile,indir,varname, outdir, outftype = 'png',
                   colmap='seismic',colorlev = 26, cmin = -ccmax, cmax=ccmax,
                   cticks=linspace(-ccmax,ccmax,13), proj='merc', nan_val=9.99e9,llev_cleaned=True,
                   dlatlabel = 10, dlonlabel = 10 , lev = levnb, alt_max = 15., lsum = False, islog=False, coordfile = cfile,
                   ladd_arrow_wind=False, windfile=[], LSwind=False, cmapextend = 'max', extravar_contour=[], colorbar_units='% pts')



#!/usr/bin/python
# based on Create_LNG_Netcdf.py this script is to process level 1 data (attenuated backscatter)
# to create LNG NetCDF files based on the original ASCII ABC2*.n.* or ABC2*.z.* files
# (files like ABC2_20170905.000.n.001059 )
# should work with both python 2.7 and python 3 as long as netCDF4 is installed
# L. Labbouz, Aug 2018

import numpy as np
from warnings import warn
from various_utils import str2float
from netCDF4 import Dataset
import time as ttime
import glob as glob

# ***************** Parameters to edit ************************************ #

outpath        = '/Users/labbouz/Desktop/RESEARCH/AEROCLO/ABC2_netcdf/'  # the output files will be written here
path_in_all    = '/Users/labbouz/Desktop/Aeroclo-sA_LNGdata/'            # directory containing all the LNG ASCII
name_in_type   = 'ArchiveAeroclo-sA_LNG_ABC_'                    # name of the per flight directories before "Vol + number"
in_fnames_type = 'ABC2*'   # individual input ASCII file names are of this form
datalev        = 'level1'  # level 1 data
expected_nblev= 2333 # number of vertical levels expected in the ASCII files (not needed but will print warning if different)
                     # but nlev must be constant within one flight 
header_lines  = 43   # default number of header lines (not needed but will print warning if different)
non_rd_hlines = 5    # number of lines at the end of tyhe Header that are not readable / not usefull to read (mandatory)

flight_number_range = range(6,15+1) # for processing flights number 6 to 15



# ************************** Function definitions ******************************* #

def readHeader(fname, nb_lines_default=header_lines):
    """Reads a data file `fname` and returns a dictionary of header metadata."""
    #adapted from http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/python_read_data_exercises_solutions.pdf

    f = open(fname)
    header = {}
    # read first line
    line = f.readline()
    # Strip any white space from line
    line = line.strip()
    key, value = line.split(": ")
    header[key] = []
    header[key].append(value)
    # read number of header lines
    if key == 'Nombre_lignes_en_tete':
        nb_lines = int(value)
    else:
        warn("number of lines in header not found - assume "+str(nb_lines_default))
        nb_lines = int(nb_lines_default)

    i = 1
    while i < nb_lines - non_rd_hlines:  # the last lines are only "---" and column names
        line = f.readline()
        # Strip any white space from line
        line = line.strip()
        key, value = line.split(": ")
        header[key] = []
        header[key].append(value)
        i += 1

    f.close()
    return header

def readData(fname, nb_lines=header_lines):
    """Reads a data file `fname` and populates dictionary: self.data."""
    #adapted from http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/python_read_data_exercises_solutions.pdf
    f = open(fname)
    data = {}

    line = f.readline()
    # Strip any white space from line
    line = line.strip()
    key, value = line.split(":")

    if key == 'Nombre_lignes_en_tete':
        nb_lines = int(value)
    else:
        warn("number of lines in header not found - assume "+str(header_lines))

    # Ignore the header
    for i in range(nb_lines - 1):  # -1 as first line already read
        f.readline()
    # Read in variable names (Attenuated_Backscatter_Coefficient)
    col_names = ['altitude',
                 'HSR_355nm',
                 '355nm',
                 '532nm',
                 '1064nm']
    for col_name in col_names:
        data[col_name] = []
        i = 0

    for line in f.readlines():
        # Strip any white space from line
        line = line.strip()
        values = line.split()

        for (i, value) in enumerate(values):
            col_name = col_names[i]
            data[col_name].append(value)

    f.close()
    return data

def read_data_list (path, infile):
    """reads the .dir file containig the file names"""
    """reads the .dir file containig the file names"""
    f = open(path + infile)
    line =  f .read().splitlines()
    out_list = [ path + '/' + x for x in line[1:] ]
    f.close()
    return out_list


from collections import defaultdict

def list_duplicates(seq):
    """ function to get the duplicates from a list, under the form (key, locs)
    where key gives the value that is duplicated and locs the (sorted) indices 
    where the value is found """
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)


def create_file(outfname, header_all, data_all, nblev, nbprofiles ):
    """function creating the actual NetCDF file and writing the variables"""

    dataset = Dataset(outfname, 'w', format='NETCDF4_CLASSIC')
    nblev = int(nblev)
    # check for duplicates
    location = [] 
    tt = str2float( header_all ['Jour_Julien'] )
    Long = str2float( header_all ['Long'] )    	
    Lat  = str2float( header_all ['Lat'] )
    visee = str2float( header_all ['Visee[0:nadi,1:zenith,2:adm]'] )
    A1064nm   =  str2float( data_all ['1064nm'] )  
    A532nm    = str2float( data_all ['532nm'] )
    A355nm    = str2float( data_all ['355nm'] )
    HSR_355nm = str2float( data_all ['HSR_355nm'] )

    # if there are duplicates select keep only the first value
    for dup in sorted(list_duplicates(tt)):
        location.extend(dup[1][1:])
    # delete elements in the reverse order to avoid having indices changing (here only one duplicate but essential if there are more of them)
    for i in sorted(location, reverse=True):
        del(tt[i])
        del(Long[i])
        del(Lat[i])
        del(visee[i])
        st = int(i*nblev)
        en = int(st + nblev)
        del(A1064nm[st:en])
        del(A532nm[st:en])
        del(A355nm[st:en])
        del(HSR_355nm[st:en])

    if len(tt) != int(nbprofiles):
        print('number of profiles : '+ str(nbprofiles) + ' Number of unique time steps : ' + str(len(tt)) )
        warn( str(int(nbprofiles) - len(tt)) +  ' duplicates have been removed from the time serie')
    
    # file Dimensions  
    nbprofiles =  len (tt)
    altitude = dataset.createDimension('altitude', nblev)
    time     = dataset.createDimension('time', nbprofiles)

    # Note that here lat and lon are not proper coordinates as 
    # each lat-lon point is fully determined by the time coordinate    

    times      = dataset.createVariable('time', np.float64, ('time',))

    altitudes  = dataset.createVariable('altitude', np.float32, ('altitude',))
    altitudes [0:]  = str2float( data_all['altitude'] )

    latitudes  = dataset.createVariable('latitude', np.float32, ('time',))
    longitudes = dataset.createVariable('longitude', np.float32, ('time',))

    # global attributes
    dataset.description   = 'Backscatter measurements from the airbone Lidar LNG2 during the 2017 AEROCLO-sA campaign. This is level 1 data (Attenuated backscatter)'
    dataset.instrument_PI = 'Cyrille Flamant, LATMOS, Paris, France'
    dataset.history       = 'NetCDF file created ' + ttime.ctime(ttime.time()) +' by L. Labbouz from original ASCII files (ABC2* files). The python scripts used can be obtained from Laurent Labbouz github repository: https://github.com/labl1/python'

    latitudes.units  = 'degree_north'
    latitudes.long_name = 'Latitude'
    latitudes.standard_name = 'latitude_north'

    longitudes.units = 'degree_east'
    longitudes.long_name = 'Longitude'
    longitudes.standard_name = 'longitude_east'

    altitudes.units = 'km'
    altitudes.standard_name = 'altitude'
    times.units = 'days since 2016-12-31 00:00:00'  # this is the day of year
    times.calendar = 'gregorian'
    times.standard_name = 'time'
    # other variables from the header could be created in the file here, if needed
    vtype = dataset.createVariable('obs_type_flag', np.int32, ('time',))
    vtype.long_name = 'Observation type flag (0: Nadir, 1: zenith, 2: admin)'

    # Create the actual 4-d variable
    ABC_HRS  = dataset.createVariable('ABC_HRS_355nm', np.float32,
    ('time','altitude'), fill_value = np.nan)
    ABC_HRS.long_name = 'High Spectral Resolution attenuated backscatter coefficient at 355 nm'
    ABC_HRS.units     = 'km-1 sr-1'
    ABC_HRS.valid_min = 0.

    ABC_355nm = dataset.createVariable('ABC_355nm', np.float32,
    ('time','altitude'), fill_value = np.nan)
    ABC_355nm.long_name = 'Attenuated backscatter coefficient at 355 nm'
    ABC_355nm.units     = 'km-1 sr-1'
    ABC_355nm.valid_min = 0.
    

    ABC_532nm = dataset.createVariable('ABC_532nm', np.float32,
    ('time','altitude'), fill_value = np.nan)
    ABC_532nm.long_name = 'Attenuated backscatter coefficient at 532 nm'
    ABC_532nm.units     = 'km-1 sr-1'
    ABC_532nm.valid_min = 0.


    ABC_1064nm = dataset.createVariable('ABC_1064nm', np.float32,
    ('time','altitude'), fill_value = np.nan)
    ABC_1064nm.long_name = 'Attenuated backscatter coefficient at 1064 nm'
    ABC_1064nm.units     = 'km-1 sr-1'
    ABC_1064nm.valid_min = 0.


    # write the data ; here this could be a loop aver multiple files

    times [:] = tt 

    longitudes [:]  = Long  # str2float( header_all ['Long'] )
    latitudes  [:]  = Lat   # str2float( header_all ['Lat'] )

    vtype [:]      = visee  # str2float( header_all ['Visee[0:nadi,1:zenith,2:adm]'] )

    ABC_1064nm [:] = A1064nm  #str2float( data_all ['1064nm'] )
    ABC_532nm  [:] = A532nm   # str2float( data_all ['532nm'] )
    ABC_355nm  [:] = A355nm   #str2float( data_all ['355nm'] )
    ABC_HRS    [:] = HSR_355nm # str2float( data_all ['HSR_355nm'] )

    # create file
    dataset.close()


## **********************************  MAIN *********************************************************** ##
# goes through all the flights
for volnb in flight_number_range:
    path_in = path_in_all + name_in_type + 'Vol' + str(volnb)
    # only one directory per flight for level 1 data (so it's simpler here than for level 0)
    fnames = glob.glob(path_in + '/'+in_fnames_type)

    i = 0
# reade all files for the flight volnb
    for name in fnames:
        if i == 0:
            name_alt = name
            data_all = readData(fname=name)
            header_all  = readHeader(fname=name)

            # write the data in reverse order if in zenith mode - we use the nadir mode direction as default
            if int(header_all['Visee[0:nadi,1:zenith,2:adm]'][0]) == int(1):
                for key in data_all.keys():
                    data_all[key]=[ij for ij in reversed(data_all[key])]

            flight_date = header_all['DATE'][0].replace(" ", "")
            nlev        = header_all['Nombre_lignes_mesures'][0]

            if int(header_all['Visee[0:nadi,1:zenith,2:adm]'][0]) != int(2): # 2 is for admin => skip them and redo this step
                if int(nlev) != int(expected_nblev):
                    warn('The number of vertical levels in the file' + name +
                         ' (' + nlev + ') is different from expected (' + str(expected_nblev) + ')')
                i = 1
        else:
            data = readData(fname=name)
            header = readHeader(fname=name)
            if int(header['Nombre_lignes_mesures'][0]) != int(nlev):  # all the files must have the same number of vertical levels
                raise ValueError( 'The number of vertical levels in the file' + name +
                                  ' (' + header['Nombre_lignes_mesures'][0] +
                                    ') is different from the one in tyhe first file read  '
                                    +  name_alt + ' (' + str(nlev) + '),' +
                                    ' the usual number expected being ' + str(expected_nblev) )
            # ALL THE DATA  from a given flight are put in the dictionaries header_all and data_all
            for col in data_all.keys() :
                # print(col)
                # altitude is a coordinate, and always the same so there is actually no need to read it over and over againb
                if col != 'altitude':
                    if int(header['Visee[0:nadi,1:zenith,2:adm]'][0]) == int(1):
                        data[col]=[ij for ij in reversed(data[col])]

                    data_all[col].extend(data[col])

                    # ou equivalent data_all[col] += data[col]
            for col in header_all.keys():
                header_all[col].extend(header[col])  # only one line per header

    outname = outpath + '/LNG2_ABC_' + datalev+'_' + flight_date + '_Vol' + str(volnb) +'.nc'
    
    # check if time variable has duplicates
    if len(header_all['Jour_Julien']) != len(set(header_all['Jour_Julien'])) : 
        print( len(header_all['Jour_Julien']) )
        print( len(set(header_all['Jour_Julien'])))
        warn('duplicate time value in ' + flight_date + ' Vol' + str(volnb) )
    # create netcdf file with Time, lat, lon and height as coordinates - on file per flight
    create_file(outfname = outname, header_all=header_all, data_all=data_all, nblev=nlev, nbprofiles=len(header_all['Long']) )


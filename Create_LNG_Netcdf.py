#!/usr/bin/python3
# to create LNG NetCDF files based on the original ASCII *.n.* or *.z.* files
# (type Vol12/000529.n/LNG2_20170908.000.n.001054)
# L. Labbouz, Aug 2018

from warnings import warn
from various_utils import str2float
from netCDF4 import Dataset
import numpy as np
import time as ttime
import glob as glob

# TODO ask Cyrille about:
# valid range ? negative altitude ? Altitude AMSL??
# negative backscatter values ? units ?
# other parameters needed

# ***************** Parameters that may have to be edited ******************* #

outpath     = '/mesonh/labl/LNG2/'                            # the output files will be written here
path_in_all = '/mesonh/chajp/WALVI/LIDAR/Aeroclo-sA_LNGdata/' # directory containing all the data :

expected_nblev= 2333 # number of vertical levels expected in the ASCII files (warning if different)
header_lines  = 43   # default number of header lines
non_rd_hlines = 5    # number of lines at the end of tyhe Header that are not readable / not usefull to read

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
        warn("number of lines in header not found - assume 43")

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

def read_data_list (infile):
    """reads the .dir file containig the file names"""
    f = open(infile)
    line = f.read().splitlines()
    out_list = line[1:]
    f.close()
    return out_list


def create_file(outfname, header_all, data_all, nblev, nbprofiles ):
    """function creating the actual NetCDF file and writing the variables"""

    dataset = Dataset(outfname, 'w', format='NETCDF4_CLASSIC')

    altitude = dataset.createDimension('altitude', nblev)
    lat      = dataset.createDimension('lat', nbprofiles)
    lon      = dataset.createDimension('lon', nbprofiles)
    time     = dataset.createDimension('time', nbprofiles)

    times      = dataset.createVariable('time', np.float64, ('time',))
    # double check that float 32 is enough
    altitudes   = dataset.createVariable('altitude', np.float32, ('altitude',))
    latitudes  = dataset.createVariable('latitude', np.float32, ('lat',))
    longitudes = dataset.createVariable('longitude', np.float32, ('lon',))

    # global attributes
    dataset.description   = 'Backscatter measurements from the airbone Lidar LNG2 during the 2017 AEROCLO-sA campaign'
    dataset.instrument_PI = 'Cyrille Flamant, LATMOS, Paris, France'
    dataset.history       = 'NetCDF file created ' + ttime.ctime(ttime.time()) +  ' from original ASCII files'

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
    #
    vtype = dataset.createVariable('obs_type_flag', np.int32, ('time',))
    vtype.long_name = 'Observation type flag (0: Nadir, 1: zenith, 2: admin)'
    vtype [:]      = str2float( header_all ['Visee[0:nadi,1:zenith,2:adm]'] )

    # Create the actual 4-d variable
    ABC_HRS = dataset.createVariable('ABC_HRS_355nm', np.float32,
    ('time','altitude','lat','lon'), fill_value =  -9999.999)
    ABC_HRS.long_name = 'High Spectral Resolution attenuated backscatter coefficient at 355 nm'
    ABC_HRS.units     = '???'

    ABC_355nm = dataset.createVariable('ABC_355nm', np.float32,
    ('time','altitude','lat','lon'), fill_value =  -9999.999)
    ABC_355nm.long_name = 'Attenuated backscatter coefficient at 355 nm'
    ABC_355nm.units     = '???'

    ABC_532nm = dataset.createVariable('ABC_532nm', np.float32,
    ('time','altitude','lat','lon'), fill_value =  -9999.999)
    ABC_532nm.long_name = 'Attenuated backscatter coefficient at 532 nm'
    ABC_532nm.units     = '???'

    ABC_1064nm = dataset.createVariable('ABC_1064nm', np.float32,
    ('time','altitude','lat','lon'), fill_value =  -9999.999)
    ABC_1064nm.long_name = 'Attenuated backscatter coefficient at 1064 nm'
    ABC_1064nm.units     = '???'

    # write the data ; here this could be a loop aver multiple files
    altitudes [:]  = str2float( data_all['altitude'] )
    times [:]      = str2float( header_all ['Jour_Julien'] )
    longitudes [:] = str2float( header_all ['Long'] )
    latitudes [:]  = str2float( header_all ['Lat'] )

    ABC_1064nm [:] = str2float( data_all ['1064nm'] )
    ABC_532nm  [:] = str2float( data_all ['532nm'] )
    ABC_355nm  [:] = str2float( data_all ['355nm'] )
    ABC_HRS    [:] = str2float( data_all ['HSR_355nm'] )

    # create file
    dataset.close()


## **********************************  MAIN *********************************************************** ##
# goes through all the flights
for volnb in flight_number_range:
    path_in = path_in_all + 'Vol' + str(volnb)
    # should be only one dir file
    dir_list = glob.glob( path_in + '/*.?/' )
    fnames = []
    for d in dir_list:
        # read filenamelist
        names = read_data_list(path_in + d + '/liste.dir')
        fnames +=  names # list of all the fullpath of one flight

    i = 0
# reade all files for the flight volnb
    for name in fnames:
        if i == 0:
            data_all = readData(fname=name)
            name_alt = name
            altitude_list = data_all['altitude']
            header_all = readHeader(fname=name)
            flight_date = header_all['DATE'].replace(" ", "")
            nlev = header_all['Nombre_lignes_mesures'][0]
            if int(nlev) != expected_nblev :
                warn( 'The number of vertical levels in the file' + name +
                                  ' (' + nlev + ') is different from expected (' + str(expected_nblev) + ')' )
            i = 1
        else:
            data = readData(fname=name)
            header = readHeader(fname=name)
            if int(header['Nombre_lignes_mesures'][0]) != nlev:  # all the files must have the same number of vertical levels
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
                    data_all[col].extend(data[col])
                    # ou equivalent data_all[col] += data[col]
            for col in header_all.keys():
                header_all[col].extend(header[col])  # only one line per header

    outname = outpath + '/LNG2_ABC_' + flight_date + '_Vol' + str(volnb)  + '.nc'
    # create netcdf file with Time, lat, lon and height as coordinates - on file per flight
    create_file(outfname = outname, header_all=header_all, data_all=data_all, nblev=nlev, nbprofiles=len(header_all['Long']) )


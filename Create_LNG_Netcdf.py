

from warnings import warn
from various_utils import str2float
from netCDF4 import Dataset
import numpy as np
import time as ttime

# functions adapted from
# http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/python_read_data_exercises_solutions.pdf
# to read LNG ASCII files (column data and header)

def readHeader(fname, nb_lines_default=43):
    "Reads a data file `fname` and returns a dictionary of header metadata."
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
        warn("number of lines in header not found - assume 43")
        nb_lines = int(nb_lines_default)

    i = 1
    while i < nb_lines - 5:  # -5 as the last lines are only "---" and column names
        line = f.readline()
        # Strip any white space from line
        line = line.strip()
        key, value = line.split(": ")
        header[key] = []
        header[key].append(value)
        print(i)
        i += 1

    f.close()
    return header

def readData(fname, nb_lines=43):
    "Reads a data file `fname` and populates dictionary: self.data."
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


## MAIN ##

fnames = ['/home/labl/Bureau/LNG_DATA/Vol12/000529.n/LNG2_20170908.000.n.001054',
          '/home/labl/Bureau/LNG_DATA/Vol12/000529.n/LNG2_20170908.000.n.113831',
          '/home/labl/Bureau/LNG_DATA/Vol12//133405.z/LNG2_20170908.000.z.134045']
i = 0
for name in fnames:
    if i == 0:
        data_all = readData(fname=name)
        altitude_list = data_all['altitude']
        header_all = readHeader(fname=name)
        i = i + 1
    else:
        data = readData(fname=name)
        header = readHeader(fname=name)
        for col in data_all.keys():
            # print(col)
            # altitude is a coordinate, and always the same so there is actually no need to read it over and over againb
            if col != 'altitude':
                data_all[col].extend(data[col])
            # ou equivalent data_all[col] += data[col]
        for col in header_all.keys():
            header_all[col].extend(header[col])  # only one line per header



# create netcdf file with Time, lat, lon and height as coordinates
# the aim is to create one single file per flight

# create onje netcdf file per profile and then concatenate them ?
# create the file


# TODO modify this bits here to read through list of files
outname = 'write_filename_full-path'
dataset = Dataset(outname, 'w', format='NETCDF4_CLASSIC')

nblev = 2333
nbprofiles = 1  # read the list of profiles for each leg and or fo the all flight (using the list of legs as well)
# create dimensions
nbprofiles = 3 # for unlimited possible to have several unlimited ?

# NO: can be usesd ONLY for time ... this makes it a bit tricky => need to first read the number of files
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
dataset.description   = 'LNG data from the AEROCLO-sA campaign'
dataset.instrument_PI = 'Cyrille Flamant'
dataset.history       = 'netcdf file created ' + ttime.ctime(ttime.time()) +  ' from original ASCII files'


latitudes.units  = 'degree_north'
longitudes.units = 'degree_east'
latitudes.long_name = 'Latitude'
longitudes.long_name = 'Longitude'
latitudes.standard_name = 'latitude_north'
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
# precise valid range here ?
# ask Cyrille about units, and "volume_attenuated_backwards_scattering_function_in_air"
# ask Cyrille about altitudes, AMSL or AGL? (for airplane height and lidar data)

ABC_355nm = dataset.createVariable('ABC_355nm', np.float32,
('time','altitude','lat','lon'), fill_value =  -9999.999)
ABC_355nm.long_name = 'Attenuated backscatter coefficient at 355 nm'
ABC_355nm.units     = '???'
# precise valid range here ?

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



ABC_1064nm = str2float( data_all ['1064nm'] )
ABC_532nm  = str2float( data_all ['532nm'] )
ABC_355nm  = str2float( data_all ['355nm'] )
ABC_HRS    = str2float( data_all ['HSR_355nm'] )

# create file

dataset.close()
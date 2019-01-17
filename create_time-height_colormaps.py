''' program to create time-hight 2D plots of a given variable, at a given locatiopn, from MNH outputs'''

loc_lat_WBA = -22.979722
loc_lon_WBA = 14.645278

loc_lat_HB = -22.096247
loc_lon_HB = 14.259694

filelist = [] # list of filenames

# get the dimension of the array for the colormap

# get the data and fill the arrays
i=0
for infile in filelist:
    var_avg, alt, varunits = get_profile_mnh(infile, indir, varname, inres, loc_lat, loc_lon,
                                             nan_val, inunits)

    #

    var2D [i,:] = var_avg
    i+=1

# plot the 2D colormap --- (here or calling a fct if likely usefull elsewhere)



# save

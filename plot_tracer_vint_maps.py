from various_utils import tracer_vert_int
indir = '/mesonh/labl/WA54b/'
exp= "XA54b"

mnh_alt_file= '/mesonh/labl/TEST0/DIA/TEST0_ALT_dia_file.nc'
nbsteps=8

for d in range(13,15):
    dd=str(d).zfill(2)
    for tstep in range(1,nbsteps+1):
        stepstr=str(tstep).zfill(3)
        infilename = exp+'.1.SEP'+dd+'.'+stepstr
        infile = indir + exp + '/' + infilename + '.nc'
        for tracer in range(1,5):
            sttracer = 'SVT'+str(tracer).zfill(3)
            tracer_vert_int(infile, sttracer  , dd , mnh_alt_file, path_out=indir+exp+'/FIG/', 
                            outname=exp+'SEP'+dd+'_'+stepstr+'_'+sttracer)
            print(sttracer + ' traced for SEP'+dd+'_'+stepstr)
            print(indir+exp+'/FIG/')
            print(" -- next --") 


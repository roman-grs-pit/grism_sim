#srun -N 1 -C cpu -t 04:00:00 --qos interactive --account m4943 python scripts/mk_grism_par_fullarea_run1_chunk.py
#srun -N 1 -C cpu -t 04:00:00 --qos interactive --account m4943 python scripts/mk_grism_par_fullarea_run1_chunk.py --dith_ind 1
#srun -N 1 -C cpu -t 04:00:00 --qos interactive --account m4943 python scripts/mk_grism_par_fullarea_run1_chunk.py --dith_ind 0 --dec_ind 1
from astropy.table import Table
from grism_sim import mk_ref_and_grism
import os
from multiprocessing import Pool

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ra_ind",help='index for the ra region; 0 or 1',default=0,type=int)
parser.add_argument("--dec_ind",help='index for the dec region 0,1,2,3',default=0,type=int)
parser.add_argument("--dith_ind",help='index for the dith region',default=0,type=int)


args = parser.parse_args()


stars = Table.read('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus.ecsv')
gals = Table.read('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits')

outdir = '/global/cfs/cdirs/m4943/grismsim/4deg2_run1/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def dosim(args):
    mk_ref_and_grism(args[0],args[1],args[2],args[3],stars,gals,outdir)

inds = []
ndet = 18
pal = [0,5,175,180] 

ra0 = 9.5
dec0 = -.65
decstep = .42
rastep = .82
dithstep = 0.1,0.1 #ra,dec step for dither
ndith = 2
decpa = 0.2 #how much offset there is in declination between 0 and 180 pa
ndec = 4 #number of "rows"
nra = 2 #number of "columns"

pa_off = 60 #new code shifted by 60
pal = [0+pa_off,5+pa_off,175+pa_off,180+pa_off] #position angles

for det in range(0,ndet):
    for pa in pal:
        decoff = decpa/2
        if pa > 100:
            decoff = -decpa/2
        ra = round(ra0+args.ra_ind*rastep+args.dith_ind*dithstep[0],4)
        dec = round(dec0+decstep*args.dec_ind+decoff+args.dith_ind*dithstep[1],4)
        inds.append([ra,dec,pa,det+1])
        

print('will simulate '+str(len(inds))+' grism images')

nproc = 80 #inefficient with 128

with Pool(processes=nproc) as pool:
    res = pool.map(dosim, inds)

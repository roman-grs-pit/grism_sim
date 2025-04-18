#test run with srun -N 1 -C cpu -t 01:00:00 --qos interactive --account m4943 python mk_grism_par.py
from astropy.table import Table
from grism_sim import mk_ref_and_grism
import os
from multiprocessing import Pool

stars = Table.read('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus.ecsv')
gals = Table.read('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits')

outdir = '/global/cfs/cdirs/m4943/grismsim/test/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def dosim(args):
    mk_ref_and_grism(args[0],args[1],args[2],args[3],stars,gals,outdir)

dec = 0
ra = 10
inds = []
ndet = 18
pal = [0,5,175,180] #position angles
decoffl = [-0.1,-0.1,0.1,0.1]
for i in range(0,ndet):
    for pa,decoff in zip(pal,decoffl):
    	inds.append([ra,dec+decoff,pa,i+1])

print('will simulate '+len(inds)+' grism images')
    
with Pool(processes=128) as pool:
    res = pool.map(dosim, inds)

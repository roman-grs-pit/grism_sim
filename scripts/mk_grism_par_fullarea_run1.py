#test run with srun -N 1 -C cpu -t 04:00:00 --qos interactive --account m4943 python mk_grism_par_fullarea_run1.py
from astropy.table import Table
from grism_sim import mk_ref_and_grism
import os
from multiprocessing import Pool

stars = Table.read('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus.ecsv')
gals = Table.read('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits')

outdir = '/global/cfs/cdirs/m4943/grismsim/4deg2_run1/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def dosim(args):
    mk_ref_and_grism(args[0],args[1],args[2],args[3],stars,gals,outdir)

dec = 0
ra = 10
inds = []
ndet = 18
pal = [0,5,175,180] 
decoffl = [-0.1,-0.1,0.1,0.1]

ra0 = 9.5
dec0 = -.65
decstep = .42
rastep = .82
dithstep = 0.1,0.1 #ra,dec step for dither
ndith = 2
decpa = 0.2 #how much offset there is in declination between 0 and 180 pa
ndec = 4 #number of "rows"
nra = 2 #number of "columns"

pal = [0,5,175,180] #position angles

for det in range(0,ndet):
	for pa in pal:
		for dith in range(0,ndith):
			for j in range(0,nra):
				for i in range(0,ndec):
					decoff = decpa/2
					if pa > 100:
						decoff = -decpa/2
					ra = ra0+j*rastep+dith*dithstep[0]
					dec = dec0+decstep*i+decoff+dith*dithstep[1]
					inds.append([ra,dec,pa,i+1])


for i in range(0,ndet):
    for pa,decoff in zip(pal,decoffl):
    	

print('will simulate '+str(len(inds))+' grism images')
    
with Pool(processes=128) as pool:
    res = pool.map(dosim, inds)

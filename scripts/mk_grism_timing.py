from astropy.table import Table
from grism_sim_psf_dependent import mk_grism
import os
import time

stars = Table.read('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus.ecsv')
gals = Table.read('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits')

outdir = os.getenv("SCRATCH")

def dosim(args, **kwargs):
    print(kwargs)
    mk_grism(args[0],args[1],args[2],args[3],stars,gals,outdir,
             confver="03192025_rotAonly.conf",check_psf=True,
             **kwargs)

ra = 10
dec = 0
pa = 60
det_num = 1

args = [ra,dec,pa,det_num,stars,gals,outdir]

start = time.time()
dosim(args, dogal='y')
end = time.time()

print(f"dosim took: {end - start}")
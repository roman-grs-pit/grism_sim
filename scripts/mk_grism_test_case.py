from astropy.table import Table
import os
import sys
import time

github_dir = os.getenv('github_dir')
if github_dir is None:
    print('github_dir environment variable has not been set, will cause problems if not explicitly set in function calss')

outdir = os.getenv("SCRATCH")

from grism_sim_psf_dependent import mk_grism

def dosim(args, **kwargs):
    print(kwargs)
    mk_grism(args[0],args[1],args[2],args[3],stars,gals,outdir, **kwargs)

ra = 9.5
dec = 1
pa = 0
det_num = 1

stars = Table.read(os.path.join(github_dir, "grism_sim/data/test_case_star.txt"), format="ascii")
# stars = Table.read(os.path.join(github_dir, "star_fields/data/sim_star_cat_galacticus.txt"), format="ascii")
gals = Table.read(os.path.join(github_dir, "grism_sim/data/test_case_gal.txt"), format="ascii")

args = [ra,dec,pa,det_num,stars,gals,outdir]

start = time.time()
dosim(args, dogal='y')
end = time.time()

print(f"dosim took: {end - start}")
from astropy.table import Table
import os
import sys

github_dir = sys.argv[1]
os.environ["github_dir"] = github_dir
outdir = os.path.join(github_dir, "bin")

from grism_sim_psf_dependent import mk_grism

def dosim(args, **kwargs):
    print(kwargs)
    mk_grism(args[0],args[1],args[2],args[3],stars,gals,outdir, **kwargs)

ra = 9.5
dec = 1
pa = 0
det_num = 1

stars = Table.read(os.path.join(github_dir, "grism_sim/data/test_case.txt"), format="ascii")
gals = None

args = [ra,dec,pa,det_num,stars,gals,outdir]

dosim(args, dogal='n')
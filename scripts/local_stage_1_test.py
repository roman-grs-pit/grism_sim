from astropy.table import Table
from grism_sim_psf_dependent import mk_grism
import os
from multiprocessing import Pool

outdir = os.getenv("SCRATCH")

# stars = Table.read('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus.ecsv')
# gals = Table.read('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits')
stars = Table.read(os.path.join(outdir, "data/sim_star_cat_galacticus.ecsv"))
gals = None

sel = stars["magnitude"] <= 17
stars = stars[sel]

def dosim(args, **kwargs):
    print(kwargs)
    mk_grism(args[0],args[1],args[2],args[3],args[4],args[5],args[6],
             confver="03192025_rotAonly",check_psf=True, dogal='n',
             conv_gal=False, npsfs=args[7],
             **kwargs)

ra = 10
dec = 0
pa = 60
det_nums = range(1, 19)

dosim([ra,dec,pa,1,stars,gals,outdir,10])
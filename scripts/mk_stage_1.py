from astropy.table import Table
from grism_sim_psf_dependent import mk_grism
import os
from multiprocessing import Pool

outdir = "/global/cfs/cdirs/m4943/grismsim/stage_1/"

stars = Table.read('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus.ecsv')
# gals = Table.read('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits')
gals = None

sel = stars["magnitude"] <= 17
stars = stars[sel]

def dosim(args, **kwargs):
    print(kwargs)
    mk_grism(args[0],args[1],args[2],args[3],args[4],args[5],args[6],
             confver="03192025_rotAonly",check_psf=True, dogal='n',
             conv_gal=False, npsfs=args[7], extra_grism_name=f"_npsfs{args[7]}", extra_ref_name=f"_npsfs{args[7]}",
             **kwargs)

ra = 10
dec = 0
pa = 60
det_nums = range(1, 19)

args = []
for npsfs in [10, 20, 50]:
    for det_num in det_nums:
        args.append([ra,dec,pa,det_num,stars,gals,outdir,npsfs])

print("will simulate"+str(len(args))+" grism images")

with Pool(nproccesses=54) as pool:
    res = pool.map(dosim, args)
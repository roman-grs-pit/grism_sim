import numpy as np
from astropy.io import fits
from collections import defaultdict
import os

def group_grism_files(outdir, all_sim_params):
    """
    Group grism files in a directory by ra, dec, pa, & det_num
    Returns a dict: {base: [file1, file2, ...]}
    """
    bases = set()
    for sim in all_sim_params:
        base = f"grism_ra{sim["tel_ra"]}_dec{sim["tel_dec"]}_pa{sim["tel_pa"]}_detSCA{sim["det_num"]:02}"
        bases.add(base)

    grouped = defaultdict(list)
    for f in os.listdir(outdir):
        for base in bases:
            if f.startswith(base):
                grouped[base].append(os.path.join(outdir, f))
            
    return grouped

def combined_sims(outdir, grouped, seed):

    rng = np.random.default_rng(seed=seed)

    for base, group in grouped.items():
        hdul = fits.open(group[0])
        sci = hdul["SCI"].data
        noiseless_model_0 = hdul["MODEL"].data
        noiseless_model_gather = noiseless_model_0.copy()
        EXPTIME = hdul[0].header["EXPTIME"]
        bg = hdul["SCI"].header["BACKGR"]

        for fn in group[1:]:
            with fits.open(fn) as f:
                noiseless_model = f["MODEL"].data
                sel = noiseless_model < 0
                noiseless_model[sel] = 0
                sci += rng.poisson(noiseless_model * EXPTIME) / EXPTIME
                noiseless_model_gather += noiseless_model
            
        hdul["SCI"].data = sci
        hdul["ERR"].data = np.sqrt((noiseless_model_gather + bg) * EXPTIME) / EXPTIME
        hdul["MODEL"].data = noiseless_model_gather
        hdul.writeto(os.path.join(outdir, base + ".fits"), overwrite=True)
    
    return 0
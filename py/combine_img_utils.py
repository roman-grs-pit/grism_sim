import numpy as np
from astropy.io import fits
from collections import defaultdict
import os

def group_grism_files(outdir, all_sim_params):
    """
    Group grism files in a directory by ra, dec, pa, & det_num
    Returns a dict: {base: [file1, file2, ...]}

    Parameters
    ----------
    outdir: str
        Directory containing files to be grouped.
    all_sim_params: list of dictionaries
        list of dictionaries containing "wfi_cen_ra", "wfi_cen_dec", "wfi_cen_pa", and 
        "det_num" if simulation run to be grouped.
    """
    bases = set()
    for sim in all_sim_params:
        base = f"grism_ra{sim["wfi_cen_ra"]}_dec{sim["wfi_cen_dec"]}_pa{sim["wfi_cen_pa"]}_detSCA{sim["det_num"]:02}"
        bases.add(base)

    grouped = defaultdict(list)
    for f in os.listdir(outdir):
        for base in bases:
            if f.startswith(base):
                grouped[base].append(os.path.join(outdir, f))
            
    return grouped

def combine_sims(outdir, grouped, seed):
    """
    Iterates through grouped fits files to combine partial simulations into full
    sims. Shot noise and ERR are recompouted as combinations occur.

    Parameters
    ----------
    outdir: str
        Directory containing fits files to be combined
    grouped: dict
        Dictionary contianing grouped filenames and base by which they were grouped
    seed: int, optional
        Numpy rng seed for noise. default: 3
    """

    rng = np.random.default_rng(seed=seed)

    for base, group in grouped.items():
        hdul = fits.open(group[0])
        sci = hdul["SCI"].data
        isim_sci = hdul["ISIM_SCI"].data
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
                isim_sci += f["ISIM_SCI"].data
                noiseless_model_gather += noiseless_model
            
        hdul["SCI"].data = sci
        hdul["ERR"].data = np.sqrt((noiseless_model_gather + bg) * EXPTIME) / EXPTIME
        hdul["MODEL"].data = noiseless_model_gather
        hdul["ISIM_SCI"].data = isim_sci
        hdul.writeto(os.path.join(outdir, base + ".fits"), overwrite=True)
    
    return 0

def group_ref_files(outdir, all_sim_params):
    """
    Group reference files in a directory by ra, dec, pa, & det_num
    Returns a dict: {base: [file1, file2, ...]}

    Parameters
    ----------
    outdir: str
        Directory containing files to be grouped.
    all_sim_params: list of dictionaries
        list of dictionaries containing "wfi_cen_ra", "wfi_cen_dec", "wfi_cen_pa", and 
        "det_num" if simulation run to be grouped.
    """
    bases = set()
    for sim in all_sim_params:
        base = f"refimage_ra{sim["wfi_cen_ra"]}_dec{sim["wfi_cen_dec"]}_pa{sim["wfi_cen_pa"]}_detSCA{sim["det_num"]:02}"
        bases.add(base)

    grouped = defaultdict(list)
    for f in os.listdir(outdir):
        for base in bases:
            if f.startswith(base) and not f.endswith("nopad.fits"):
                grouped[base].append(os.path.join(outdir, f))
            
    return grouped

def combine_refs(outdir, grouped):
    """
    Iterates through grouped fits files to combine partial reference images into
    full reference images.

    Parameters
    ----------
    outdir: str
        Directory containing fits files to be combined
    grouped: dict
        Dictionary contianing grouped filenames and base by which they were grouped
    seed: int, optional
        Numpy rng seed for noise. default: 3
    """

    for base, group in grouped.items():

        hdul = fits.open(group[0])
        full_ref = hdul["IMAGE"].data

        for fn in group[1:]:
            with fits.open(fn) as f:
                full_ref += f["IMAGE"].data

        hdul["IMAGE"].data = full_ref
        hdul.writeto(os.path.join(outdir, base + ".fits"), overwrite=True)
    
    return 0
from astropy.io import fits
import numpy as np
import os

def naming_conventions(wfi_cen_ra, wfi_cen_dec, wfi_cen_pa, det, extra_ref_name='', extra_grism_name=''):
    """Gives the filenames for a set of parameters"""

    fn_ref_base = 'refimage_ra%s_dec%s_pa%s_det%s' % (wfi_cen_ra, wfi_cen_dec, wfi_cen_pa, det)
    fn_ref_full = fn_ref_base + extra_ref_name

    fn_grism_base = 'grism_ra%s_dec%s_pa%s_det%s' % (wfi_cen_ra,wfi_cen_dec,wfi_cen_pa,det)
    fn_grism_full = fn_grism_base + extra_grism_name

    filenames = {"fn_ref": fn_ref_full,
                 "fn_grism": fn_grism_full,
                 "fn_ref_base": fn_ref_base,
                 "fn_grism_base": fn_grism_base}

    return filenames

def trim_complete_sims(outdir, all_sims):
    """Check which files were completed, and trim them out of the sims to be run"""

    trimmed_sims = []

    for sim in all_sims:

        # get filenames based on current conventions
        det = "SCA{:02}".format(sim["det_num"])
        names = naming_conventions(sim["wfi_cen_ra"], sim["wfi_cen_dec"], sim["wfi_cen_pa"],
                                   det, sim["extra_ref_name"], sim["extra_grism_name"])
        fn_grism = names["fn_grism"]

        # if the file doesn't exist, add it to trimmed_sims and move on
        if not os.path.exists(fn_path := os.path.join(outdir, fn_grism)):
            trimmed_sims.append(sim)
            continue

        # open the file, and check if it's probably complete
        incomplete_file = False
        with fits.open(fn_path) as hdul:
            if "MODEL" not in hdul:
                incomplete_file = True
            elif not np.any(hdul["MODEL"].data):
                incomplete_file = True

        # If it's incomplete, add its args to trimmed_sim and delete it
        if incomplete_file:
            trimmed_sims.append(sim)
            os.remove(fn_path)

    return trimmed_sims
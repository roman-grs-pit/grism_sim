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

def is_complete(path, is_ref=False):
    """Checks if a Grism Sim file is complete"""

    # see if it's exists
    if not os.path.exists(path):
        return False

    # open it up, and check for data; any data is assumed to be good
    try:
        with fits.open(path) as hdul:
            if is_ref:
                data = hdul["IMAGE"].data
            else:
                data = hdul["MODEL"].data
            if data is None or not np.any(data):
                return False
    # if reading/accessing the data is a problem, throw the file away and start again
    except (TypeError, KeyError, ValueError, OSError, EOFError, IndexError, fits.VerifyError):
        return False

    # if it exists, it has the MODEL HDU, and the MODEL HDU has something, it's probably complete
    return True

def trim_complete_sims(outdir, all_sims):
    """Check which files were completed, and trim them out of the sims to be run"""

    trimmed_sims = []

    for sim in all_sims:

        # get filenames based on current conventions
        det = "SCA{:02}".format(sim["det_num"])
        names = naming_conventions(sim["wfi_cen_ra"], sim["wfi_cen_dec"], sim["wfi_cen_pa"],
                                   det, sim["extra_ref_name"], sim["extra_grism_name"])

        combined_fn = os.path.join(outdir, names["fn_grism_base"] + ".fits")
        partial_fn = os.path.join(outdir, names["fn_grism"] + ".fits")

        # check combined and partial file for completeness.
        if is_complete(combined_fn):
            if is_complete(partial_fn):
                continue
            os.remove(combined_fn)

        else:
            if is_complete(partial_fn):
                continue

        # if it's not complete, add it to the to-do list and delete any file remnants
        trimmed_sims.append(sim)
        if os.path.exists(partial_fn):
            os.remove(partial_fn)

    return trimmed_sims
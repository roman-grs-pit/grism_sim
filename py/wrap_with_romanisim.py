import subprocess
import argparse
import glob
import os

from multiprocessing import Pool
from astropy.io import fits

def fits_to_asdf(fn, outdir, seed=42, static_args=None):

    with fits.open(fn) as f:
        ra, dec = f[0].header["WFICENRA"], f[0].header["WFICENDEC"]
        pa = f[0].header["WFICENPA"]
        det_num = f[0].header["DETNUM"]

    out_fn = os.path.join(outdir, fn.strip(".fits") + "_l2.asdf")

    cmd = [
        out_fn,
        f"--extra-counts {fn} 5",
        f"--radec {str(ra)} {str(dec)} --roll {str(pa)}",
        f"--sca {str(det_num)}",
        f"--rng_seed {seed}",
    ]

    if static_args is not None:
        cmd.append(*static_args)

    subprocess.call(["romanisim-make-image", cmd])

def fits_to_asdf_wrapper(d):
    fits_to_asdf(**d)

def wrap_with_romanisim(outdir):
    """
    Converts grism simulation fits files to asdf files, with realistic noise added, using Roman I-Sim.
    """
    file_glob = glob.glob(os.path.join(outdir, "grism_*_detSCA??.fits"))

    date = "--date 2026-01-01T12:00:00.000"
    ma_table = "--ma_table_number 1036"
    bandpass = "--bandpass GRISM"
    flags = "--usecrds --stpsf"
    nobj = "--nobj 0"
    level = "--level 2"

    static_args = [
        date, ma_table, bandpass, flags, nobj, level
    ]

    args_list = []
    for fn in file_glob:
        args_list.append({"fn": fn, "outdir": outdir, "static_args": static_args})

    with Pool(processes=80) as pool:
        pool.map(fits_to_asdf_wrapper, args_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    wrap_with_romanisim(args.outdir)
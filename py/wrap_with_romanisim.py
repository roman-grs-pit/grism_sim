import subprocess
import argparse
import glob
import os, sys

from multiprocessing import Pool
from astropy.io import fits

def fits_to_asdf(fn, outdir, seed=42, static_args=None):
    """
    Read fits file for romanisim argument info. Actually call romanisim using subprocess.call.

    Parameters
    ----------
    fn: str
        Path to fits file to be converted.
    outdir: str
        Path to output directory
    seed: int, optional
        Random seed. Default: 42
    static_args: list, optional
        Any additional arguments to be passed to romanisim, in their complete string form.
    """

    with fits.open(fn) as f:
        ra, dec = f[0].header["WFICENRA"], f[0].header["WFICENDEC"]
        pa = f[0].header["WFICENPA"] - 60
        det_num = f[0].header["DETNUM"]

    out_fn = os.path.join(outdir, fn.strip(".fits") + "_l2.asdf")

    cmd = [
        out_fn,
        "--extra-counts", fn, "5",
        "--radec", str(ra), str(dec),
        "--roll", str(pa),
        "--sca", str(det_num),
        "--rng_seed", str(seed),
    ]

    if static_args is not None:
        cmd.extend(static_args)

    romanisim_make_image = os.path.join(sys.executable, "../romanisim-make-image") # path to romanisim-make-image call
    romanisim_make_image = os.path.abspath(romanisim_make_image) # clean up path to make it functional

    subprocess.call([romanisim_make_image, *cmd])

def fits_to_asdf_wrapper(d):
    fits_to_asdf(**d)

def wrap_with_romanisim(outdir):
    """
    Converts grism simulation fits files to asdf files, with realistic noise added, using Roman I-Sim.

    Parameters
    ----------
    outdir: str
        Path to output directory
    """
    file_glob = glob.glob(os.path.join(outdir, "grism_*_detSCA??.fits"))

    date = ["--date", "2026-01-01T12:00:00.000"]
    ma_table = ["--ma_table_number", "1036"]
    bandpass = ["--bandpass", "GRISM"]
    flags = ["--usecrds", "--stpsf"]
    nobj = ["--nobj", "0"]
    level = ["--level", "2"]

    static_args = [
        *date, *ma_table, *bandpass, *flags, *nobj, *level
    ]

    args_list = []
    for fn in file_glob:
        args_list.append({"fn": fn, "outdir": outdir, "static_args": static_args})

    with Pool(processes=80) as pool:
        pool.map(fits_to_asdf_wrapper, args_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=str, help="Directory containing fits files to be converted")

    args = parser.parse_args()

    wrap_with_romanisim(args.outdir)
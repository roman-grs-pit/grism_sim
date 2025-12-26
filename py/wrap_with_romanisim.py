import subprocess
import argparse
import glob
import os

from multiprocessing import Pool
from astropy.io import fits

def wrap_with_romanisim(outdir):
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

    seed = 42  # ! fix with standardize non-static seed when agreed upon

    def fits_to_asdf(fn):
        with fits.open(fn) as f:
            print(f["PRIMARY"].header)
            ra, dec = f[0].header["WFICENRA"], f[0].header["WFICENDEC"]
            pa = f[0].header["WFICENPA"]
            det_num = f[0].header["DETNUM"]

        extra_counts = f"--extra-counts {fn} 5"
        radec = f"--radec {str(ra)} {str(dec)} --roll {str(pa)}"
        sca = f"--sca {str(det_num)}"
        rng_seed = f"--rng_seed {seed}"

        out_fn = os.path.join(outdir, fn.strip(".fits") + "_l2.asdf")

        subprocess.call(["romanisim-make-image", out_fn, extra_counts, radec, sca, rng_seed, *static_args])

    with Pool(processes=80) as pool:
        pool.map(fits_to_asdf, file_glob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, help="Output directory")

    args = parser.parse_args()

    wrap_with_romanisim(args.outdir)
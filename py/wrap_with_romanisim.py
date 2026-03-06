import subprocess
import argparse
import glob
import os, sys

from functools import partial
from multiprocessing import Pool
from astropy.io import fits
from pathlib import Path

def fits_to_asdf(fn: str,
                 outdir: str,
                 hdu: int = 5,
                 seed: int = 42,
                 static_args: list[str] | None =None) -> None:
    """
    Read fits file for romanisim argument info. Actually call romanisim using subprocess.call.

    Parameters
    ----------
    fn: str
        Path to fits file to be converted.
    outdir: str
        Path to output directory
    hdu: int, optional
        HDU number for romanisim extra counts (default: 5)
    seed: int, optional
        Random seed. Default: 42
    static_args: list, optional
        Any additional arguments to be passed to romanisim, in their complete string form.
    """

    with fits.open(fn) as f:
        ra, dec = f[0].header["WFICENRA"], f[0].header["WFICENDEC"]
        pa = f[0].header["WFICENPA"] - 60
        det_num = f[0].header["DETNUM"]

    out_fn = os.path.join(outdir, Path(fn).stem + "_l2.asdf")

    cmd = [
        out_fn,
        "--extra-counts", fn, str(hdu),
        "--radec", str(ra), str(dec),
        "--roll", str(pa),
        "--sca", str(det_num),
        "--rng_seed", str(seed),
    ]

    if static_args is not None:
        cmd.extend(static_args)

    romanisim_make_image = os.path.join(sys.executable, "../romanisim-make-image") # path to romanisim-make-image call
    romanisim_make_image = os.path.abspath(romanisim_make_image) # clean up path to make it functional

    subprocess.run([romanisim_make_image, *cmd], check=True)

def wrap_with_romanisim(outdir: str,
                        nprocesses: int = 80,
                        hdu: int = 5,
                        date: str = "2026-01-01T12:00:00.000",
                        ma_table: int = 1036,
                        level: int = 2,
                        extra_static_args: list[str] | None = None) -> None:
    """
    Converts grism simulation fits files to asdf files, with realistic noise added, using Roman I-Sim.

    Parameters
    ----------
    outdir: str
        Path to output directory
    nprocesses: int, optional
        Number of processes to use in python multiprocessing (default: 80)
    hdu: int, optional
        HDU number for romanisim extra counts (default: 5)
    date: str, optional
        Date of pointing used for romanisim background (default: "2026-01-01T12:00:00.000")
    ma_table: int, optional
        MA Table number (default: 1036 for 190s exposure time)
    level: int, optional
        Data product level
    extra_static_args: list, optional
        Any additional arguments to be passed to romanisim. Must be a list of strings, with [flag, value] pairs.
    """

    # set args which are the same for every file wrapped
    date_arg = ["--date", date]
    ma_table_arg = ["--ma_table_number", str(ma_table)]
    bandpass_arg = ["--bandpass", "GRISM"]
    flags_arg = ["--usecrds", "--stpsf"]
    nobj_arg = ["--nobj", "0"]
    level_arg = ["--level", str(level)]

    static_args = [
        *date_arg,
        *ma_table_arg,
        *bandpass_arg,
        *flags_arg,
        *nobj_arg,
        *level_arg
    ]

    if extra_static_args is not None:
        static_args.extend(extra_static_args)

    # partially define fits_to_asdf
    fits_to_asdf_partial = partial(fits_to_asdf, outdir=outdir, hdu=hdu, static_args=static_args)

    # send in all files to be wrapped
    file_glob = glob.glob(os.path.join(outdir, "grism_*_detSCA??.fits"))
    with Pool(processes=nprocesses) as pool:
        pool.map(fits_to_asdf_partial, file_glob)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=str, help="Directory containing fits files to be converted")
    parser.add_argument("--nprocesses", "-p", type=int, default=80,
                        help="Number of processes to use in python multiprocessing (default: 80)")
    parser.add_argument("--hdu", type=int,  default=5,
                        help="HDU number for romanisim extra counts (default: 5)")
    parser.add_argument("--date", type=str, default="2026-01-01T12:00:00.000",
                        help="Observation date to pass to romanisim")
    parser.add_argument("--ma-table", type=int, default=1036,
                        help="MA table number for romanisim (default: 1036)")
    parser.add_argument("--level", type=int, default=2,
                        help="ASDF level (default: 2)")
    parser.add_argument("--extra-static-args", nargs="*", default=None,
                        help="Additional raw flags to append to romanisim call")

    args = parser.parse_args()
    wrap_with_romanisim(
        outdir=args.outdir,
        nprocesses=args.nprocesses,
        hdu=args.hdu,
        date=args.date,
        ma_table=args.ma_table,
        level=args.level,
        extra_static_args=args.extra_static_args,
    )

if __name__ == "__main__":
    main()
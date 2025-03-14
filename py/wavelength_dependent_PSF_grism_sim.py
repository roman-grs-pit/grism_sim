# General imports
from math import ceil
import numpy as np
from scipy import constants
from astropy.io import fits
from astropy.table import Table, join
from grizli.model import GrismFLT
import pysynphot as S
import webbpsf
import webbpsf.roman
import argparse
import os
import yaml

"""
This script is designed to be run either from the command line, which saves a fits with the star dispersion, or
imported and run in a notebook, which returns a GrismFLT object containing the dispersion.

To run from notebook, run create_objects_for_disperse_function with appropriate arguments. This returns a dictionary
with the args for disperse_one_star. 
"""


def interpolate_spectrum(src) -> Table:
    """
    Takes a pysynphot spectrum array. Truncates it, reinterpolates it at every angstrom, and returns it as an 
    astropy table with columns "wave" and "flux".

    Parameters
    ----------

    src: pysynphot.ArraySpectrum
        Pysynphot ArraySpectrum object extending between at least 10000-20000 Angstroms.
    """
    # Three steps in rapid sequence/all mixed together
    # 1) Extract only the relevant part of the spectrum (from 10000 to 20000 angstroms)
    # 2) Interpolate so that each data point is 1 angstrom apart
    # 3) Put the spectrum in an astropy Table
    old_wave = src.wave
    old_flux = src.flux

    new_wave = np.arange(10000, 20000 +1, 1)
    new_flux = np.interp(new_wave, old_wave, old_flux)

    spec = Table([new_wave, new_flux], names=("wave", "flux"), dtype=(np.float64, np.float64))

    return spec

def split_and_apodize(spec, start_wave, end_wave, spectrum_overlap) -> tuple:

    start_wave -= spectrum_overlap * 0.5
    end_wave += spectrum_overlap * 0.5 - 1

    if start_wave < 10000:
        start_wave = 10000
    
    if end_wave > 20000:
        end_wave = 20000

    chunk_spec = chunk(spec, start_wave, end_wave)

    # Setup roll-on/roll-off shape
    window_x = np.linspace(0, np.pi, spectrum_overlap)
    front_y = (1 - np.cos(window_x)) / 2
    back_y = 1 - front_y

    wave = chunk_spec["wave"]
    flux = chunk_spec["flux"]

    # apodize/roll-on, roll-off
    if start_wave != 10000:
        flux[:spectrum_overlap] *= front_y

    if end_wave != 10000:
        flux[-spectrum_overlap:] *= back_y

    return (wave, flux)

def chunk(spec, start_wave, end_wave):
    """
    Returns a chunk of the spectrum array between start and end wavelengths.

    Parameters
    ----------
    spec: astropy.table.Table
        Spectrum the chunk should come from.

    start_wave: int
        Wavelength to begin the cut in the same units as the spectrum.

    end_wave: int
        Wavelength to end the cut in the same units as the spectrum.
    """

    start_idx = np.where(spec["wave"] == start_wave)[0][0]
    end_idx = np.where(spec["wave"] == end_wave)[0][0] + 1

    chunk_spec = np.asarray(spec[start_idx:end_idx])

    return chunk_spec

def psf_bounds(pad, detector_position, half_psf_size) -> tuple:
    """
    Locate the x and y bounds on the detector for placing the psf onto the direct image array.

    Parameters
    ----------
    pad: tuple
        GrismFLT padding (y,x). Set equal GrismFLT.pad (i.e. roman.pad)

    detector_position: tuple
        Detector position of the star (y,x).
    """

    # Pad indicies are stored as (y, x); detector position tuple chosen to match
    x_lower = (pad[1] + detector_position[1]) - half_psf_size
    x_upper = (pad[1] + detector_position[1]) + half_psf_size
    y_lower = (pad[0] + detector_position[0]) - half_psf_size
    y_upper = (pad[0] + detector_position[0]) + half_psf_size

    bounds = (y_lower, y_upper, x_lower, x_upper)

    return bounds

def interp_and_truncate_spec(src) -> Table:
    """
    Takes a pysynphot spectrum array. Truncates it, reinterpolates it at every angstrom, and returns it as an 
    astropy table with columns "wave" and "flux".

    Parameters
    ----------

    src: pysynphot.ArraySpectrum
        Pysynphot ArraySpectrum object extending between at least 10000-20000 Angstroms.
    """
    # Three steps in rapid sequence/all mixed together
    # 1) Extract only the relevant part of the spectrum (from 10000 to 20000 angstroms)
    # 2) Interpolate so that each data point is 1 angstrom apart
    # 3) Put the spectrum in an astropy Table
    start_idx = np.where(src.wave == 10000)[0][0]
    end_idx = np.where(src.wave == 20000)[0][0]

    old_wave = src.wave[start_idx:end_idx]
    old_flux = src.flux[start_idx:end_idx]

    new_wave = np.arange(10000, 20000 +1, 1)
    new_flux = np.interp(new_wave, old_wave, old_flux)

    spec = Table([new_wave, new_flux], names=("wave", "flux"), dtype=(np.float64, np.float64))

    return spec


def disperse_one_star(roman, wfi, src, spectrum_overlap, npsfs, detector_position, psf_thumbnail_size) -> GrismFLT:
    """
    Produce dipsered model. Saves model in the GrismFLT object. Returns that object.

    Parameters
    ----------

    roman: grizli.model.GrismFLT
        GrismFLT object. The pieces of the dispersion will be saved in the GrismFLT object

    spectrum_overlap: int
        The number of data points that overlap as spectrum segments roll-on/off.
    
    npsfs: int
        The number of distinct PSFs to be used. Also, the number of spectrum segments.
    
    detector_position: tuple
        Detector x and y position of the star.
    """
    # Parse args and assertions
    # Enforce int type
    spectrum_overlap = int(spectrum_overlap) # overlap extent; data points
    npsfs = int(npsfs) # number of distinct psfs

    assert psf_thumbnail_size % 2 == 0, "psf_thumbnail_size must be even"
    half_psf_thumb = int(psf_thumbnail_size / 2) # Used for in the for-loop for positioning thumbnail on the detector

    assert type(detector_position)==tuple and len(detector_position)==2 and [type(ii)==int for ii in detector_position], \
           "detector_position arg must be a tuple like (y_position, x_position)"

    # Interpolates spectrum to every angstrom and returns only 10000-20000 angstroms
    spec = interp_and_truncate_spec(src)

    # Setup roll-on/roll-off shape
    window_x = np.linspace(0, np.pi, spectrum_overlap)
    front_y = (1 - np.cos(window_x)) / 2
    back_y = 1 - front_y

    # Determine start wavelength of every bin
    bins = np.linspace(10000, 20000, npsfs + 1)

    # Initialize zero arrays
    full_dispersion = np.zeros_like(roman.model)
    
    # Dipserse each segment one-at-a-time and add each bit to the full_dispersion array
    for ii, start_wave in enumerate(bins[:-1]):
        end_wave = bins[ii+1]

        # Confirmed in the docs; "Wavelengths are always specified in meters"
        psf = wfi.calc_psf(monochromatic=(start_wave * (10**-10)), fov_pixels=182, oversample=1, source=src)[0].data

        # Determine where to place the psf on the detector
        bounds = psf_bounds(pad=roman.pad, detector_position=detector_position, half_psf_size=half_psf_thumb)

        direct = np.zeros_like(roman.model)
        direct[bounds[0]:bounds[1], bounds[2]:bounds[3]] = np.rot90(psf, k=3) # PSF made using non-rotated coordinates

        # Enforce float32 dtype in GrismFLT object and build seg map
        roman.direct.data["SCI"] = direct.astype("float32")
        roman.seg = np.where(roman.direct.data["SCI"], 1, 0).astype("float32") # Segmentation map is made on rotated direct image
        # TODO: is there a notably faster option than np.where?

        start_wave -= spectrum_overlap * 0.5
        end_wave += spectrum_overlap * 0.5 - 1

        # Stay within our spectrum limits (these could be extended or not hardcoded if needed)
        if start_wave < 10000:
            start_wave = 10000
        
        if end_wave > 20000:
            end_wave = 20000

        chunk_spec = chunk(spec, start_wave, end_wave) # extract relevant part of spectrum
        wave = chunk_spec["wave"]
        flux = chunk_spec["flux"]

        # apodize/roll-on, roll-off
        if start_wave != 10000:
            flux[:spectrum_overlap] *= front_y

        if end_wave != 20000:
            flux[-spectrum_overlap:] *= back_y

        # TODO Simulate multiple stars?

        #! Block for in_place sim
        # roman.compute_model_orders(id=1, mag=1, compute_size=False, size=77, is_cgs=True, store=False, 
        #                            in_place=True, spectrum_1d=[wave, flux])

        #? Dispersing in_place causes negatives and casts a shadow?
        #? Adding individual dispersions on my own seems to fix the problem, but why?
        #* If using the manual collection method below, you must comment out the in_place method
        #* Failure to comment this line out with result in negative/incorrect values at the tail end of the dispersion
        #! /block

        #! Block for cumulative sim in my own "net"/capture array
        segment_of_dispersion = roman.compute_model_orders(id=1, mag=1, compute_size=False, size=200, is_cgs=True, store=False, 
                                                          in_place=False, spectrum_1d=[wave, flux])
        
        #* Comment either these two lines or the in_place block to switch collection methods
        full_dispersion += segment_of_dispersion[1]
        #! /block

        # These deletions are unnecessary
        del chunk_spec
        del flux
        del wave

    roman.model = full_dispersion
    return roman

def create_objects_for_disperse_function(empty_fits_dir=None, spectrum_file=None, bandpass_file=None, 
                                         magnitude=6, spectrum_overlap=10, npsfs=20, 
                                         detector_position=(2044, 2044), psf_thumbnail_size=182) -> tuple:
    """
    "If name==main" parses key words. Then calls main.

    If not provided, the main function calls the spectrum lookup function 
    which pulls a star spectrum template from the _______ library. Then, it
    prepares and calls the dispersion_model function. This function produces
    a grism disperion model using a wavelength-dependent PSF. Finally, main 
    saves the model as a fits file, which can later be accessed later to avoid
    needlessly recomputing a given stellar types model. i.e. This trades 
    compute time for storage space.
    """

    # Start dictionary and store args that are not created or modified
    args = {"spectrum_overlap": spectrum_overlap,
            "npsfs": npsfs,
            "detector_position": detector_position,
            "psf_thumbnail_size": psf_thumbnail_size}

    # Read in bandpass array; Will use this to renorm the spectrum
    if bandpass_file:
        df = Table.read(bandpass_file, format='fits')
        bp = S.ArrayBandpass(df["WAVELENGTH"], df["THROUGHPUT"])

    # Read in the spectrum
    if spectrum_file:
        spec = Table.read(spectrum_file, format="ascii")
        src = S.ArraySpectrum(wave=spec["col1"], flux=spec["col2"], waveunits="angstroms", fluxunits="flam")

    # If spectrum was not provided, lookup a spectrum matching the stellar_type in library
    else:
        # TODO Implement lookup function
        None
    
    if bandpass_file:
            src = src.renorm(magnitude, "abmag", bp)
            src.convert("flam")

    args["src"] = src

    if empty_fits_dir:
        empty_direct = os.path.join(empty_fits_dir, "empty_direct.fits")
        empty_seg = os.path.join(empty_fits_dir, "empty_seg.fits")

    else:
        # TODO Implement some make_empty
        None

    # Instantiate WebbPSF object
    wfi = webbpsf.roman.WFI()
    wfi.detector_position = detector_position
    
    args["wfi"] = wfi

    # Instantiate GrismFLT; Fill in with empty direct image and segmentation map
    lower_limit = min(detector_position) - ceil(psf_thumbnail_size / 2)
    upper_limit = max(detector_position) - ceil(psf_thumbnail_size / 2)

    # Pad is defined squarely for simplicity; this could be changed if warranted
    if lower_limit < 0:
        pad = abs(lower_limit)
    elif upper_limit > 4088:
        pad = upper_limit
    else:
        pad = 0
    pad = max(pad, 200) # pad must be as large or larger than the size cutout to avoid failures at the edges

    roman = GrismFLT(direct_file=empty_direct, seg_file=empty_seg, pad=pad)

    args["roman"] = roman

    return args

def main() -> None:
    """
    Parse args and call disperse_one_star, save result in fits file
    """
    # Initialize arg parser to assist with command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--empty_fits_dir", type=str, default=None,
                        help="An optional filepath pointing to a directory with empty direct and segmentation fits files.")
    
    parser.add_argument("--spectrum_file", type=str, default=None,
                        help="An optional filepath to the spectrum file.")

    parser.add_argument("--bandpass_file", type=str, default=None,
                        help="An optional filepath to a bandpass file.")

    parser.add_argument("--stellar_type", type=str, default=None,
                        help="If spectrum_file is not provided, a required argument indicating stellar type to lookup.")
                    
    parser.add_argument("--magnitude", type=float, default=6,
                        help="An optional argument: Magnitude of the star in the bandpass provided.")

    parser.add_argument("--spectrum_overlap", type=int, default=10,
                        help="An optional argument: amount of wavelength in angstroms in the overlapping regions of the spectrum segments.")

    parser.add_argument("--npsfs", type=int, default=20,
                        help="The number of distinct PSFs to be used. Also, the number of spectrum segments.")

    parser.add_argument("--save_file", type=str, default="PSF_dependent_dispersion.fits",
                        help="Name used when saving the results to a fits file.")

    parser.add_argument("--detector_position", type=tuple, default=(2044, 2044),
                        help="Specify star detector position in a tuple (y, x). Default=(2044, 2044)")
        
    parser.add_argument("--pdf_thumbnail_size", type=int, default=182,
                        help="Size of the PSF thumbnail.")

    args = parser.parse_args()

    args_for_disperse = create_objects_for_disperse_function(empty_fits_dir=args.empty_fits_dir, 
                                                             spectrum_file=args.spectrum_file, 
                                                             bandpass_file=args.bandpass_file, 
                                                             magnitude=args.magnitude,
                                                             spectrum_overlap=args.spectrum_overlap, 
                                                             npsfs=args.npsfs, 
                                                             detector_position=args.detector_position,
                                                             psf_thumbnail_size=args.psf_thumbnail_size)

    roman = disperse_one_star(args_for_disperse)

    # Save results
    pad = roman.pad
    upright_img = np.rot90(roman.model[pad:-pad,pad:-pad])
    ImageHDU = fits.ImageHDU(data=upright_img, name="SCI")
    ImageHDU.writeto(args.save_file, overwrite=True)
        
    return None

if __name__ == "__main__":

    main()
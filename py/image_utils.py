import numpy as np
from astropy.io import fits

try:
    import stpsf
except:
    import webbpsf as stpsf

wfi = stpsf.roman.WFI()
wfi.filter = "GRISM0" #eventually make this detector specific

def star_postage_grid(psf_grid, mag, detx=2044, dety=2044, half_fov_pixels=182):
    """
    Evaluate the PSF at a specific point using a given GriddedPSFModel object.

    Parameters
    ----------
    psf_grid: GriddedPSFModel
        Photutils PSF Grid
    mag: float
        Magnitude of the object. Converted to flux using mag2flux.
    detx: float, optional
        x postition of the object on the detector in science coordinates. default: 2044
    dety: float, optional
        y postition of the object on the detector in science coordinates. default: 2044
    half_fov_pixels: int, optional
        Half of the fov_pixels value used to generate the psf_grid. default: 182
    """

    flux = mag2flux(mag)

    x_0 = int(detx)
    y_0 = int(dety)
    y, x = np.mgrid[y_0-half_fov_pixels:y_0+half_fov_pixels, x_0-half_fov_pixels:x_0+half_fov_pixels]

    psf = psf_grid.evaluate(x=x, y=y, x_0=detx, y_0=dety, flux=flux).astype(np.float32)
    return psf

def gal_postage_grid(psf_grid, detx=2044, dety=2044, half_fov_pixels=364, flux=1):
    """
    Evaluate the PSF at a specific point using a given GriddedPSFModel object.

    Parameters
    ----------
    psf_grid: GriddedPSFModel
        Photutils PSF Grid
    detx: float
        x postition of the object on the detector in science coordinates.
    dety: float
        y postition of the object on the detector in science coordinates.
    half_fov_pixels: int
        Half of the fov_pixels value used to generate the psf_grid.
    """

    x_0 = int(detx)
    y_0 = int(dety)
    y, x = np.mgrid[y_0-half_fov_pixels:y_0+half_fov_pixels, x_0-half_fov_pixels:x_0+half_fov_pixels]

    psf = psf_grid.evaluate(x=x, y=y, x_0=detx, y_0=dety, flux=flux) #? What's the right flux for this?
    return psf

def create_psf_grid(wavelength=1.5e-6, fov_pixels=364, det="SCA01"):
    """
    Generate new monochromatic GriddedPSFModel object.

    Parameters
    ----------
    wavelength: float, optional
        Monochromatic wavelength at which the psf is evaluated. default: 1.5e-6
    fov_pixels: int, optional
        Size of the psf thumbnails in pixels. default: 364
    det: str, optional
        Name of the detector. default: "SCA01"
    """

    wfi.detector = det
    grid = wfi.psf_grid(all_detectors=False, use_detsampled_psf=True, monochromatic=wavelength, fov_pixels=fov_pixels)
    return grid

#fiducial zero point set based on 2022 sim below
def mag2flux(mag,zp=26.5):
    """
    Convert magnitude to flux.

    Parameters
    ----------
    mag: float
        Apparent Magnitude to convert to flux
    zp: float, optional
        zeropoint; default: 26.5 - Wang, Y., 2022, ApJ, 928, 110. doi: 10.3847/1538-4357/ac4973
    """

    f0 = 10**(0.4*zp)
    flux = f0*10**(-0.4*mag) #mag = 26.5 - 2.5*np.log10(sumflux)
    return flux
    
def star_postage(mag,detx=2044,dety=2044,offx=0,offy=0,wavelength = 1.5e-6, fov_pixels=364, oversample=4,arcperpixel = 0.11, det="SCA01"):
    """
    Evaluate monochromatic PSF for a given detector at a given detector coordinate. 
    Total thumbnail size in pixel is fov_pixels * oversample.

    Parameters
    ----------
    mag: float
        Magnitude of the object. Converted to flux using mag2flux.
    detx: float, optional
        x postition of the object on the detector in science coordinates. 
        default: 2044
    dety: float, optional
        y postition of the object on the detector in science coordinates. 
        default: 2044
    offx: float, optional
        x pixel offset between PSF thumbnail center and object center. default: 0
    offy: float, optional
        y pixel offset between PSF thumbnail center and object center. default: 0
    wavelength: float, optional
        Monochromatic wavelength at which the psf is evaluated. default: 1.5e-6
    fov_pixels: int, optional
        Size of the thumnail to be returned in pixels. default: 364
    oversample: int, optional
        Factor by which to oversample the PSF. i.e. number of simulated pixels per 
        detector pixel in thumbnail. default: 4
    arcperpixel: float, optional
        arcseconds per pixel on detector; used for source offset. default: 0.11
    det: str, optional
        Name of the detector. default: "SCA01"
    """

    wfi.options['source_offset_x'] = offx*arcperpixel
    wfi.options['source_offset_y'] = offy*arcperpixel
    wfi.detector = det
    wfi.detector_position = (detx, dety)
    psf = wfi.calc_psf(monochromatic=wavelength, fov_pixels=fov_pixels, oversample=oversample)
    flux = mag2flux(mag)
    return psf[0].data*flux

def get_psf(wavelength = 1.5e-6, fov_pixels=364, oversample=4,detx=2044,dety=2044, det="SCA01"):
    """
    Evaluate monochromatic PSF for a given detector at a given detector coordinate. 
    Total thumbnail size in pixel is fov_pixels * oversample. This function does
    not support source offset and returns PSF normalized such that it sums to 1, 
    i.e. not adjusted to an object's flux.

    Parameters
    ----------
    wavelength: float, optional
        Monochromatic wavelength at which the psf is evaluated. default: 1.5e-6
    fov_pixels: int, optional
        Size of the thumnail to be returned in pixels. default: 364
    oversample: int, optional
        Factor by which to oversample the PSF. i.e. number of simulated pixels per 
        detector pixel in thumbnail. default: 4
    detx: float, optional
        x postition of the object on the detector in science coordinates. 
        default: 2044
    dety: float, optional
        y postition of the object on the detector in science coordinates. 
        default: 2044
    det: str, optional
        Name of the detector. default: "SCA01"
    """

    wfi.detector = det
    wfi.detector_position = (detx, dety) #fiducial case is at the center
    psf = wfi.calc_psf(monochromatic=wavelength, fov_pixels=fov_pixels, oversample=oversample)
    return psf

#wavelength = 1.5e-6, fov_pixels=364, oversample=2
def star_postage_inpsf(mag,psf):
    """
    Adjust a fiducial PSF to given magnitude.

    Parameters
    ----------
    mag: float
        Magnitude of the object. Converted to flux using mag2flux.
    psf: numpy.ndarray
        PSF thumbnail, normalized to sum to 1.
    """
    flux = mag2flux(mag)
    return psf[0].data*flux


def add_Roman_header(file_name,fout_name=None,roman_base_dir='',det='1'):
    """
    DEPRECATED. This function is not longer used in current scripts.

    Add Roman header and config file info to fits files for Grizli GrismFLT instantiation.

    Parameters
    ----------
    file_name: str
        path to file to which the header will be added
    fout_name: str, optional
        path to which the modified file will be saved. If none, file_name will be  
        overwritten. default: None
    roman_base_dir: str, optional
        roman_base_dir+"configuration" should point to aXeSim style config files.
        default: ''
    det: str, optional
        detector number to be simulated using this fits file. default: '1'
    """

    if fout_name is None:
        fout_name = file_name
    file = fits.open(file_name) 
    file[0].header["INSTRUME"] = "ROMAN"
    file[0].header["FILTER"] = "d1_"
    file[1].header["CONFFILE"] = roman_base_dir+"configuration/Roman.det"+det+".07242020.conf"
    file.writeto(fout_name, overwrite=True)
    file.close()

def fake_header_wcs(crval1, crval2, crpix2=2044,crpix1=2044, cdelt1=0.11, cdelt2=0.11,
                crota2=0.0,naxis1=4088,naxis2=4088):
    """
    Returns an otherwise empty PrimaryHDU containing only WCS information.

    Parameters
    ----------
    crval1: float
    crval2: float
    """

    # NOTE: Pysiaf is used for WCS now. These functions should only be used for fits header info
    #make empty hdu header and add wcs 
    
    hdu = fits.PrimaryHDU()
    
    return add_wcs(hdu,crval1, crval2, crpix2,crpix1, cdelt1, cdelt2, crota2,naxis1,naxis2)


def add_wcs(hdu,crval1, crval2, crpix2=2044,crpix1=2044, cdelt1=0.11, cdelt2=0.11,
                crota2=0.0,naxis1=4088,naxis2=4088):
    """
    Adds WCS information to given HDU header

    Parameters
    ----------
    hdu: HDU
    crval1: float
    crval2: float
    """
    # NOTE: Pysiaf is used for WCS now. These functions should only be used for fits header info
    
    #add wcs to existing header
    #maintain consistency with https://github.com/roman-grs-pit/observing-program/blob/main/py/footprintutils.py, at some point make both use same function
    # crota2 - degree
    # cdelt1 - arcsec
    # cdelt2 - arcsec

    #hdu = fits.PrimaryHDU()
    #hdu.header

    # http://stsdas.stsci.edu/documents/SUG/UG_21.html

    theta = crota2*np.pi/180. # radians
    cdelt1 /= 3600. # deg
    cdelt2 /= 3600. # deg

    R = np.array([
        [-1*np.cos(theta), 1*np.sin(theta)],
        [1*np.sin(theta), np.cos(theta)],
    ])


    cd1_1 = cdelt1*R[0,0]
    cd1_2 = cdelt2*R[0,1]
    cd2_1 = cdelt1*R[1,0]
    cd2_2 = cdelt2*R[1,1]
                    
    hdu.header.set('NAXIS',2) # pixels
    hdu.header.set('NAXIS1',naxis1) # pixels
    hdu.header.set('NAXIS2',naxis2) # pixels

    hdu.header.set('WCSAXES',2) # pixels
    hdu.header.set('CTYPE1','RA---TAN')
    hdu.header.set('CTYPE2','DEC--TAN')
    hdu.header.set('CRVAL1',crval1) # deg
    hdu.header.set('CRVAL2',crval2) # deg
    hdu.header.set('CRPIX1',crpix1) # pixels
    hdu.header.set('CRPIX2',crpix2) # pixels
    #hdu.header.set('CDELT1',cdelt1)
    #hdu.header.set('CDELT2',cdelt2)
    #hdu.header.set('CROTA2',crota2)
    hdu.header.set("CD1_1",cd1_1)
    hdu.header.set("CD1_2",cd1_2)
    hdu.header.set("CD2_1",cd2_1)
    hdu.header.set("CD2_2",cd2_2)

    return hdu.header

import numpy as np
import webbpsf
from astropy.io import fits

wfi = webbpsf.roman.WFI()
wfi.filter = "GRISM0" #eventually make this detector specific


#fiducial zero point set based on 2022 sim below
def mag2flux(mag,zp=26.5):
    f0 = 10**(0.4*zp)
    flux = f0*10**(-0.4*mag) #mag = 26.5 - 2.5*np.log10(sumflux)
    return flux
    
def star_postage(mag,detx=2044,dety=2044,offx=0,offy=0,wavelength = 1.5e-6, fov_pixels=364, oversample=2,arcperpixel = 0.11):
    wfi.options['source_offset_x'] = offx*arcperpixel
    wfi.options['source_offset_y'] = offy*arcperpixel
    wfi.detector_position = (detx, dety)
    psf = wfi.calc_psf(monochromatic=wavelength, fov_pixels=fov_pixels, oversample=oversample)
    flux = mag2flux(mag)
    return psf[0].data*flux

def get_psf(wavelength = 1.5e-6, fov_pixels=364, oversample=2,detx=2044,dety=2044):
    wfi.detector_position = (detx, dety) #fiducial case is at the center
    psf = wfi.calc_psf(monochromatic=wavelength, fov_pixels=fov_pixels, oversample=oversample)
    return psf

#wavelength = 1.5e-6, fov_pixels=364, oversample=2
def star_postage_inpsf(mag,psf):
    flux = mag2flux(mag)
    return psf[0].data*flux


def add_Roman_header(file_name,fout_name=None,roman_base_dir='',det='1'):
    if fout_name is None:
        fout_name = file_name
    file = fits.open(file_name) 
    file[0].header["INSTRUME"] = "ROMAN"
    file[0].header["FILTER"] = "d1_"
    file[1].header["CONFFILE"] = roman_base_dir+"configuration/Roman.det"+det+".07242020.conf"
    file.writeto(fout_name, overwrite=True)
    file.close()

def add_wcs(hdu,crval1, crval2, crpix2=2044,crpix1=2044, cdelt1=0.11, cdelt2=0.11,
                crota2=0.0,naxis1=4088,naxis2=4088):

    # crota2 - degree
    # cdelt1 - arcsec
    # cdelt2 - arcsec

    #hdu = fits.PrimaryHDU()
    #hdu.header

    # http://stsdas.stsci.edu/documents/SUG/UG_21.html

    theta = crota2*np.pi/180. # radians
    cdelt1 /= 3600. # deg
    cdelt2 /= 3600. # deg

    cd1_1 = -cdelt1*np.cos(theta)
    cd1_2 = -cdelt2*np.sin(theta)
    cd2_1 = -cdelt1*np.sin(theta)
    cd2_2 = cdelt2*np.cos(theta)

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
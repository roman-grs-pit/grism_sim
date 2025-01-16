import numpy as np
import webbpsf
from astropy.io import fits

wfi = webbpsf.roman.WFI()
wfi.filter = "GRISM0"

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

def add_Roman_header(file_name,fout_name=None,det=1,roman_base_dir='',det='1'):
    if fout_name is None:
        fout_name = file_name
    file = fits.open(file_name) 
    file[0].header["INSTRUME"] = "ROMAN"
    file[0].header["FILTER"] = "d1_"
    file[1].header["CONFFILE"] = roman_base_dir+"configuration/Roman.det"+det+".07242020.conf"
    file.writeto(fout_name, overwrite=True)
    file.close()
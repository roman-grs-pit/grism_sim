'''
environment to set on NERSC machines before running this script:
export github_dir=/global/common/software/m4943/grizli0/
export psf_grid_data_read=/dvs_ro/cfs/cdirs/m4943/grismsim/psf_grid_data
export PYTHONPATH=$PYTHONPATH:$github_dir//grism_sim/py/:$github_dir/optical_model_tools/py/:$github_dir/psf_grids/py/:$github_dir/galacticus_sed_calculator/

'''
import psf_grid_utils as pgu  # needs to come after setting $psf_grid_data_read
import pysynphot as S
from galacticus_sed_calculator import SEDCalculator as sed
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from grizli import fake_image
from scipy import signal
from astropy.modeling.models import Sersic2D
import image_utils as iu
from optical_model_tools.v0_8 import optical_model
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table, join, vstack
import os
import sys
from astropy.io import fits
import h5py
import glob
from grizli.model import GrismFLT
import grizli
import yaml
import tqdm
import logging
import time
logname = 'grism_sim'
logger = logging.getLogger(logname)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


# os.environ['github_dir'] = '/global/common/software/m4943/grizli0/'
# sys.path.append(os.environ['github_dir']+'/grism_sim/py')
# sys.path.append(os.environ['github_dir']+'/psf_grids/py')
# os.environ['psf_grid_data_read'] = "/dvs_ro/cfs/cdirs/m4943/grismsim/psf_grid_data"
output_dir = os.environ['SCRATCH']+'/roman_test/nopysiaf/'

optmod = optical_model.RomanOpticalModel()

conf_file = os.path.join(
    os.environ['github_dir'], "grism_sim/data/grizli_config.yaml")
with open(conf_file) as f:
    grizli_conf = yaml.safe_load(f)


def get_galacticus_catinfo(fname):
    fn = h5py.File(fname)
    try:
        mags = fn['Lightcone']['Output1']['nodeData']['apparentMagnitudeRomanWFI:F158'][:]
        ras, decs = fn['Lightcone']['Output1']['nodeData']['rightAscension'][:
                                                                             ], fn['Lightcone']['Output1']['nodeData']['declination'][:]
        galt = Table()
        galt['mag'] = mags
        galt['RA'] = ras
        galt['DEC'] = decs
        return galt
    except:
        return None


# compile all of the galacticus info
gall = []
gal_fns = glob.glob(
    '/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock/romanUNIT*')
for fname in gal_fns:
    gali = get_galacticus_catinfo(fname)
    if gali is not None:
        gall.append(gali)
    else:
        print(fname + ' failed')
input_gals = vstack(gall)
print('compiled input gals, length; '+str(len(input_gals)))

# parameters for simulation
magmax = 25
det_num = 1
tel_ra = 10.065
tel_dec = 0.04
pa = 0

# get ra,dec at center of detector for wcs
ra, dec = optmod.coords.convert_sca_to_sky(
    2043, 2043, tel_ra, tel_dec, pa, det_num)
ra = ra[0]
dec = dec[0]
print('center of detector ra, dec: ', ra, dec)
psf_cutout_size = 365
det = "SCA{:02}".format(det_num)

confdir = grizli_conf["confdir"]
conf = grizli_conf["conf"][det]
if confdir is not None:
    conf = os.path.join(confdir, conf)
psf_cutout_size = 365
pad = psf_cutout_size
gpad = grizli_conf["pad"]
background = grizli_conf["grism_background"]
EXPTIME = 301
NEXP = 1


gal_fpa = optmod.coords.calculate_fpa_pos(
    input_gals['RA'], input_gals['DEC'], tel_ra, tel_dec, pa)
gal_xy_raw = optmod.coords.convert_fpa_to_sca(
    gal_fpa[0], gal_fpa[1], sca=det_num)
gal_xy = (gal_xy_raw[0] + gpad, gal_xy_raw[1] + gpad)
sel_ondet = gal_xy[0] > 0
sel_ondet &= gal_xy[0] < 4088 + 2*(gpad)
sel_ondet &= gal_xy[1] > 0
sel_ondet &= gal_xy[1] < 4088 + 2*(gpad)
gals = input_gals[sel_ondet]
gals['Xpos'] = gal_xy[0][sel_ondet]
gals['Ypos'] = gal_xy[1][sel_ondet]

sel_mag = gals['mag'] < magmax
gals = gals[sel_mag]
ngal = len(gals)
print('number of galaxies within detector padded region with magnitude < ' +
      str(magmax) + ' is '+str(ngal))


# get PSF
half_fov_pixels = grizli_conf["fov_pixels"] // 2
# {instrument}_{filter}_{fovp}_wave{wavelength}_{det}.fits
psf_filename = f"wfi_grism0_fovp{half_fov_pixels * 2}_wave{15000:.0f}_{det}.fits".lower()
psf_grid = pgu.load_psf_grid(psf_filename)
fid_psf = iu.psf_grid_evaluate_fast(psf_grid, 2044, 2044, None)

# galaxy profile
r_eff = 2.5  # radius for profile in pixels
x, y = np.meshgrid(np.arange(-15, 15), np.arange(-15, 15)
                   )  # 30x30 grid of pixels
round_exp = Sersic2D(amplitude=1, r_eff=r_eff, n=1)  # round exponential
# ust something that is not a pointsource, this should get much better
testprof = round_exp(x, y)
conv_prof_fixed = signal.fftconvolve(fid_psf, testprof, mode='same')
testprof /= np.sum(testprof)  # normalize the profile

# make reference image
full_image = np.zeros((4088+2*(gpad+pad), 4088+2*(gpad+pad)))
full_seg = np.zeros((4088+2*(gpad+pad), 4088+2*(gpad+pad)), dtype=int)
thresh = 0.01  # threshold flux for segmentation map
N = 0
print('adding galaxies to reference image')
for i in range(0, ngal):
    row = gals[i]
    mag = row['mag']
    imflux = iu.mag2flux(mag)
    xpos = row['Xpos']
    ypos = row['Ypos']
    if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
        print(xpos, ypos, 'out of bounds position')
    xp = int(xpos)
    yp = int(ypos)
    xoff = 0  # xpos-xp
    yoff = 0  # ypos-yp
    sp = imflux*conv_prof_fixed
    fov_pixels = (pad-1) // 2
    full_image[yp+pad-fov_pixels:yp+pad+fov_pixels,
               xp+pad-fov_pixels:xp+pad+fov_pixels] += sp
print('added galaxies to the reference image')
# rotates roman.direct.data["REF"] and seg map for stars; galaxy seg map rotated later
full_image = np.rot90(full_image, k=3)
full_seg = np.rot90(full_seg, k=3)

# write some fits image files

fn_root = 'refimage_ra%s_dec%s_pa%s_det%s' % (tel_ra, tel_dec, pa, det)
direct_fits_out = os.path.join(output_dir, fn_root+'.fits')
direct_fits_out_nopad = os.path.join(output_dir, fn_root+'_nopad.fits')
nopad_seg = os.path.join(output_dir, fn_root + "_seg_nopad.fits")
pad_seg = os.path.join(output_dir, fn_root + "seg_wpad.fits")

# rotates roman.direct.data["REF"] and seg map for stars; galaxy seg map rotated later
full_image = np.rot90(full_image, k=3)
full_seg = np.rot90(full_seg, k=3)

hdu = fits.PrimaryHDU(data=full_seg[pad:-pad, pad:-pad])
hdul = fits.HDUList([hdu])
hdu.writeto(nopad_seg, overwrite=True)
hdu = fits.PrimaryHDU(data=full_seg)
hdul = fits.HDUList([hdu])
hdu.writeto(pad_seg, overwrite=True)

cut_image = full_image[pad:-pad, pad:-pad]
phdu = fits.PrimaryHDU(data=cut_image)
phdu.header["INSTRUME"] = 'ROMAN   '
phdu.header["FILTER"] = "f140w"
phdu.header["EXPTIME"] = 141
shp = cut_image.shape
print('cut image shape: ', shp)
print(shp[0]/2, shp[1]/2)
phdu.header = iu.add_wcs(phdu, ra, dec, crpix2=int(shp[1]/2), crpix1=int(shp[0]/2),
                         crota2=pa, naxis1=int(shp[0]), naxis2=int(shp[1]))

err = np.random.poisson(10, cut_image.shape)*0.001  # np.zeros(cut_image.shape)
ihdu = fits.ImageHDU(data=cut_image, name='SCI', header=phdu.header)
ehdu = fits.ImageHDU(data=err, name='ERR', header=phdu.header)
dhdu = fits.ImageHDU(data=np.zeros(cut_image.shape),
                     name='DQ', header=phdu.header)
hdul = fits.HDUList([phdu, ihdu, ehdu, dhdu])
hdul.writeto(direct_fits_out_nopad, overwrite=True)

fn_root_grism = 'grism_ra%s_dec%s_pa%s_det%s' % (tel_ra, tel_dec, pa, det)

empty_grism = os.path.join(output_dir, 'empty_'+fn_root_grism+'.fits')
h, wcs = fake_image.roman_header(
    ra=ra, dec=dec, pa_aper=pa, naxis=(4088, 4088))
head = wcs.to_header()
fake_image.make_fake_image(h, output=empty_grism,
                           exptime=EXPTIME, nexp=NEXP, background=background)
file = fits.open(empty_grism)
file[1].header["CONFFILE"] = os.path.join(
    os.getenv('github_dir'), "grism_sim/data", conf)
file.writeto(empty_grism, overwrite=True)
file.close()

size = grizli_conf["size"][det]

# initialize grizli
roman = GrismFLT(grism_file=empty_grism,
                 ref_file=direct_fits_out_nopad, seg_file=None, pad=gpad)
masked_seg = fits.open(nopad_seg)[0].data
# this segmentation map should have the area of the padded grism image, but not have the padding added because of the PSF size
roman.seg = masked_seg.astype("float32")

# simulate dispersed spectra
# sys.path.append(os.environ['github_dir']+'/galacticus_sed_calculator')
# setup
unit = FlatLambdaCDM(H0=67.74, Om0=0.3089)
sedTemplateFilename = '/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
sedCalc = sed(sedTemplateFilename, cosmology=unit)
# wavelengths to produce sed over
obs_wavelengths = np.linspace(0.5, 2.5, 1000)*u.micron
high_res_wavelengths = np.linspace(1e4, 2e4, 5000)*u.angstrom
photid = 0

gal_fns = glob.glob(
    '/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock/romanUNIT*')
for fname in gal_fns:
    t0 = time.time()
    galt = get_galacticus_catinfo(fname)
    if galt is not None:
        gal_fpa = optmod.coords.calculate_fpa_pos(
            galt['RA'], galt['DEC'], tel_ra, tel_dec, pa)
        gal_xy_raw = optmod.coords.convert_fpa_to_sca(
            gal_fpa[0], gal_fpa[1], sca=det_num)
        gal_xy = (gal_xy_raw[0] + gpad, gal_xy_raw[1] + gpad)

        sel_ondet = gal_xy[0] > 0
        sel_ondet &= gal_xy[0] < 4088 + 2*(gpad)
        sel_ondet &= gal_xy[1] > 0
        sel_ondet &= gal_xy[1] < 4088 + 2*(gpad)
        # gals = galt[sel_ondet]
        galt['Xpos'] = gal_xy[0]  # [sel_ondet]
        galt['Ypos'] = gal_xy[1]  # [sel_ondet]
        # gals.rename_column(gal_mag_col, 'mag')
        sel_mag = galt['mag'] < magmax
        ids = np.arange(0, len(galt))
        ids = ids[sel_ondet & sel_mag]
        gals = galt[sel_ondet & sel_mag]
        # for i in tqdm.tqdm(range(0,len(gals))):
        for i in range(0, len(gals)):

            photid += 1
            row = gals[i]
            mag = row['mag']
            idi = ids[i]
            imflux = iu.mag2flux(mag)  # imflux = row['flux']
            # make image, put it in seg
            full_image = np.zeros((4088+2*(gpad+pad), 4088+2*(gpad+pad)))
            full_seg = np.zeros(
                (4088+2*(gpad+pad), 4088+2*(gpad+pad)), dtype=int)
            thresh = 0.01  # threshold flux for segmentation map
            N = 0
            # if args.fast_direct == 'y':
            # signal.convolve2d(fid_psf[0].data,testprof,mode='same')
            conv_prof = conv_prof_fixed
            xpos = row['Xpos']
            ypos = row['Ypos']
            xp = int(xpos)
            yp = int(ypos)
            xoff = 0  # xpos-xp
            yoff = 0  # ypos-yp
            sp = imflux*conv_prof
            fov_pixels = (pad-1) // 2
            full_image[yp+pad-fov_pixels:yp+pad+fov_pixels,
                       xp+pad-fov_pixels:xp+pad+fov_pixels] += sp
            masked_im = full_image[pad:-pad, pad:-pad]

            selseg = sp > thresh
            full_seg[yp+pad-fov_pixels:yp+pad+fov_pixels, xp +
                     pad-fov_pixels:xp+pad+fov_pixels][selseg] = photid
            masked_seg = full_seg[pad:-pad, pad:-pad]
            roman.seg = np.rot90(np.asarray(masked_seg, dtype=np.float32), k=3)
            total_flux = sedCalc.evaluate_total_spectrum(
                fname, idi, obs_wavelengths=obs_wavelengths, use_synphot=False)
            # this gets flux interpolating for high res
            flux = total_flux(high_res_wavelengths, flux_unit='FLAM')

            # roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=True, store=False,is_cgs=True, spectrum_1d=[high_res_wavelengths.value, flux.value])
            roman.compute_model_orders(id=photid, mag=mag, compute_size=False, in_place=True,
                                       store=False, is_cgs=True, spectrum_1d=[high_res_wavelengths.value, flux.value])
        tf = time.time()
        logger.info(fname + ' finished; processed ' +
                    str(len(gals))+' in '+str(tf-t0)+' seconds')
    else:
        logger.info(fname+' failed to load')

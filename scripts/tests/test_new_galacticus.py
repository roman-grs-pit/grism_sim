import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table, join, vstack
import os,sys
from astropy.io import fits
import h5py
import glob
from grizli.model import GrismFLT
import grizli
import yaml
import pysiaf
import tqdm
os.environ['github_dir']='/global/common/software/m4943/grizli0/'
sys.path.append(os.environ['github_dir']+'/grism_sim/py')
sys.path.append(os.environ['github_dir']+'/psf_grids/py')
os.environ['psf_grid_data_read'] = "/dvs_ro/cfs/cdirs/m4943/grismsim/psf_grid_data"
output_dir = os.environ['SCRATCH']+'/roman_test/'
import psf_grid_utils as pgu #needs to come after setting $psf_grid_data_read

conf_file = os.path.join(os.environ['github_dir'], "grism_sim/data/grizli_config.yaml")
with open(conf_file) as f:
    grizli_conf = yaml.safe_load(f)

def get_galacticus_catinfo(fname):
    fn = h5py.File(fname)
    try:
        mags = fn['Lightcone']['Output1']['nodeData']['apparentMagnitudeRomanWFI:F158'][:]
        ras,decs = fn['Lightcone']['Output1']['nodeData']['rightAscension'][:],fn['Lightcone']['Output1']['nodeData']['declination'][:]
        galt = Table()
        galt['mag'] = mags
        galt['RA'] = ras
        galt['DEC'] = decs
        return galt
    except:
        return None

#compile all of the galacticus info
gall = []
gal_fns = glob.glob('/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock/romanUNIT*')
for fname in gal_fns:
    gali = get_galacticus_catinfo(fname)
    if gali is not None:
        gall.append(gali)
    else:
        print(fname + ' failed')
input_gals = vstack(gall)
print('compiled input gals, length; '+str(len(input_gals)))

#parameters for simulation
magmax = 25
det_num = 1
tel_ra=10.065
tel_dec=0.04
pa = 0
psf_cutout_size=365 
det = "SCA{:02}".format(det_num)
        
confdir = grizli_conf["confdir"]
conf = grizli_conf["conf"][det]
if confdir is not None:
    conf = os.path.join(confdir, conf)
psf_cutout_size=365    
pad = psf_cutout_size
gpad = grizli_conf["pad"]
background = grizli_conf["grism_background"]
EXPTIME = 301 
NEXP = 1     


#get coordinate transformation information
siaf = pysiaf.Siaf("roman")
wfi_siaf = siaf["WFI{:02}_FULL".format(det_num)]
# Use WFI_CEN for aiming
v2ref = siaf["WFI_CEN"].V2Ref
v3ref = siaf["WFI_CEN"].V3Ref
attmat = pysiaf.utils.rotations.attitude_matrix(v2ref, v3ref, tel_ra, tel_dec, pa) # pysiaf pa is 60 more than image_utils pa (i.e. siaf_pa = iu_pa + 60)
wfi_siaf.set_attitude_matrix(attmat)

#test to find out where center of detector will be
ra, dec = wfi_siaf.det_to_sky(2043, 2043) # I believe pysiaf uses 0-index for origin pixel; thus, center pixel is 2043 not 2044
print('center of detector will be at ',ra,dec)

gal_xy_siaf = wfi_siaf.sky_to_sci(input_gals["RA"], input_gals["DEC"])
gal_xy = (gal_xy_siaf[0] + gpad, gal_xy_siaf[1] + gpad)
sel_ondet = gal_xy[0] > 0
sel_ondet &= gal_xy[0] < 4088 + 2*( gpad)
sel_ondet &= gal_xy[1] > 0
sel_ondet &= gal_xy[1] < 4088 + 2*( gpad)
gals = input_gals[sel_ondet]
gals['Xpos'] = gal_xy[0][sel_ondet]
gals['Ypos'] = gal_xy[1][sel_ondet]

sel_mag = gals['mag'] < magmax
gals = gals[sel_mag]
ngal = len(gals)
print('number of galaxies within detector padded region with magnitude < '+str(magmax)+ ' is '+str(ngal))


#get PSF
half_fov_pixels = grizli_conf["fov_pixels"] // 2
psf_filename = f"wfi_grism0_fovp{half_fov_pixels * 2}_wave{15000:.0f}_{det}.fits".lower() # {instrument}_{filter}_{fovp}_wave{wavelength}_{det}.fits
psf_grid = pgu.load_psf_grid(psf_filename)
import image_utils as iu
fid_psf = iu.psf_grid_evaluate_fast(psf_grid,2044,2044,None)

#galaxy profile
r_eff = 2.5 #radius for profile in pixels
x, y = np.meshgrid(np.arange(-15,15), np.arange(-15,15)) #30x30 grid of pixels
from astropy.modeling.models import Sersic2D
round_exp = Sersic2D(amplitude=1, r_eff=r_eff,n=1) #round exponential 
testprof = round_exp(x,y) #ust something that is not a pointsource, this should get much better
from scipy import signal
conv_prof_fixed = signal.fftconvolve(fid_psf,testprof,mode='same')
testprof /= np.sum(testprof) #normalize the profile

#make reference image
full_image = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)))
full_seg = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)),dtype=int)
thresh = 0.01 #threshold flux for segmentation map
N = 0
print('adding galaxies to reference image')
for i in range(0,ngal):
    row = gals[i]
    mag = row['mag']
    imflux = iu.mag2flux(mag)
    xpos = row['Xpos']
    ypos = row['Ypos']
    if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
        print(xpos,ypos,'out of bounds position')
    xp = int(xpos)
    yp = int(ypos)
    xoff = 0#xpos-xp
    yoff = 0#ypos-yp
    sp = imflux*conv_prof_fixed
    fov_pixels = (pad-1) // 2
    full_image[yp+pad-fov_pixels:yp+pad+fov_pixels,xp+pad-fov_pixels:xp+pad+fov_pixels] += sp
print('added galaxies to the reference image')
# rotates roman.direct.data["REF"] and seg map for stars; galaxy seg map rotated later
full_image = np.rot90(full_image, k=3)
full_seg = np.rot90(full_seg, k=3)

#write some fits image files

fn_root = 'refimage_ra%s_dec%s_pa%s_det%s' % (tel_ra,tel_dec,pa,det)
direct_fits_out = os.path.join(output_dir,fn_root+'.fits' )
direct_fits_out_nopad = os.path.join(output_dir,fn_root+'_nopad.fits')
nopad_seg = os.path.join(output_dir,fn_root+ "_seg_nopad.fits")
pad_seg = os.path.join(output_dir,fn_root+ "seg_wpad.fits")

# rotates roman.direct.data["REF"] and seg map for stars; galaxy seg map rotated later
full_image = np.rot90(full_image, k=3)
full_seg = np.rot90(full_seg, k=3)

hdu = fits.PrimaryHDU(data=full_seg[pad:-pad,pad:-pad])
hdul = fits.HDUList([hdu])
hdu.writeto(nopad_seg, overwrite=True)
hdu = fits.PrimaryHDU(data=full_seg)
hdul = fits.HDUList([hdu])
hdu.writeto(pad_seg, overwrite=True)

cut_image = full_image[pad:-pad,pad:-pad]
phdu = fits.PrimaryHDU(data=cut_image)
phdu.header["INSTRUME"] = 'ROMAN   '
phdu.header["FILTER"] = "f140w"
phdu.header["EXPTIME"] = 141
shp = cut_image.shape
phdu.header = iu.add_wcs(phdu,ra, dec, crpix2=shp[1]/2,crpix1=shp[0]/2,
                         crota2=pa,naxis1=shp[0],naxis2=shp[1])

err = np.random.poisson(10,cut_image.shape)*0.001 #np.zeros(cut_image.shape)
ihdu = fits.ImageHDU(data=cut_image,name='SCI',header=phdu.header)
ehdu = fits.ImageHDU(data=err,name='ERR',header=phdu.header)
dhdu = fits.ImageHDU(data=np.zeros(cut_image.shape),name='DQ',header=phdu.header)
hdul = fits.HDUList([phdu,ihdu,ehdu,dhdu])
hdul.writeto(direct_fits_out_nopad, overwrite=True)

from grizli import fake_image
fn_root_grism = 'grism_ra%s_dec%s_pa%s_det%s' % (tel_ra,tel_dec,pa,det)
 
empty_grism = os.path.join(output_dir, 'empty_'+fn_root_grism+'.fits')
h, wcs = fake_image.roman_header(ra=ra, dec=dec, pa_aper=pa, naxis=(4088,4088))
head = wcs.to_header()
fake_image.make_fake_image(h, output=empty_grism, exptime=EXPTIME, nexp=NEXP, background=background)
file = fits.open(empty_grism)
file[1].header["CONFFILE"] = os.path.join(os.getenv('github_dir'), "grism_sim/data", conf)
file.writeto(empty_grism, overwrite=True)
file.close()

size = grizli_conf["size"][det]

#initialize grizli
roman = GrismFLT(grism_file=empty_grism,ref_file=direct_fits_out_nopad, seg_file=None, pad=gpad)
masked_seg = fits.open(nopad_seg)[0].data       
roman.seg = masked_seg.astype("float32") #this segmentation map should have the area of the padded grism image, but not have the padding added because of the PSF size

#simulate dispersed spectra
sys.path.append(os.environ['github_dir']+'/galacticus_sed_calculator')
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import SEDfromSFH as sed
import pysynphot as S
#setup
unit = FlatLambdaCDM(H0=67.74, Om0=0.3089)
sedTemplateFilename='/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
sedCalc = sed.sed_calculator(sedTemplateFilename, cosmology=unit)
obs_wavelengths = np.linspace(0.5, 2.5, 1000)*u.micron #wavelengths to produce sed over
high_res_wavelengths = np.linspace(1e4, 2e4, 5000)*u.angstrom
photid = 0  

gal_fns = glob.glob('/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock/romanUNIT*')
for fname in gal_fns:
    galt = get_galacticus_catinfo(fname)
    if galt is not None:
        gal_xy_siaf = wfi_siaf.sky_to_sci(galt["RA"], galt["DEC"])
        gal_xy = (gal_xy_siaf[0] + gpad, gal_xy_siaf[1] + gpad)

        sel_ondet = gal_xy[0] > 0
        sel_ondet &= gal_xy[0] < 4088 + 2*( gpad)
        sel_ondet &= gal_xy[1] > 0
        sel_ondet &= gal_xy[1] < 4088 + 2*( gpad)
        #gals = galt[sel_ondet]
        galt['Xpos'] = gal_xy[0]#[sel_ondet]
        galt['Ypos'] = gal_xy[1]#[sel_ondet]
        #gals.rename_column(gal_mag_col, 'mag')
        sel_mag = galt['mag'] < magmax
        ids = np.arange(0,len(mags))
        ids = ids[sel_ondet&sel_mag]
        gals = galt[sel_ondet&sel_mag]
        for i in tqdm.tqdm(range(0,len(gals))):
            photid += 1
            row = gals[i]
            mag = row['mag']
            idi = ids[i]
            imflux = iu.mag2flux(mag)#imflux = row['flux']
            #make image, put it in seg
            full_image = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)))
            full_seg = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)),dtype=int)
            thresh = 0.01 #threshold flux for segmentation map
            N = 0
            #if args.fast_direct == 'y':
            conv_prof = conv_prof_fixed#signal.convolve2d(fid_psf[0].data,testprof,mode='same')
            xpos = row['Xpos']
            ypos = row['Ypos']
            xp = int(xpos)
            yp = int(ypos)
            xoff = 0#xpos-xp
            yoff = 0#ypos-yp
            sp = imflux*conv_prof
            fov_pixels = (pad-1) // 2
            full_image[yp+pad-fov_pixels:yp+pad+fov_pixels,xp+pad-fov_pixels:xp+pad+fov_pixels] += sp
            masked_im = full_image[pad:-pad,pad:-pad]
            
            selseg = sp > thresh
            full_seg[yp+pad-fov_pixels:yp+pad+fov_pixels,xp+pad-fov_pixels:xp+pad+fov_pixels][selseg] = photid
            masked_seg = full_seg[pad:-pad,pad:-pad]
            roman.seg = np.rot90(np.asarray(masked_seg,dtype=np.float32), k=3)
            total_flux = sedCalc.evaluate_total_spectrum(fname, idi, obs_wavelengths=obs_wavelengths,use_synphot=False)
            flux = total_flux(high_res_wavelengths, flux_unit='FLAM') #this gets flux interpolating for high res
            
            roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=True, store=False,is_cgs=True, spectrum_1d=[high_res_wavelengths.value, flux.value])
            print(fname +' finished')
    else:
        print(fname+' failed to load')

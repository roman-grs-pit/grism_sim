'''
#Usage: create direct image of star field and then simulate a grism image for it
#Example call: python $github_dir/grism_sim/scripts/mkstar_image.py
#Requirements: grizli (and all of its dependencies); A clone of the roman-grs-pit star_fields repo in the same relative path
#Will produce a plot and save the corresponding direct and grism images
#A successful run should demonstrate that basic Roman GRS simulation capabilities have been installed properly and give one an idea on how to produce more/better simulations



## Note that when running mkstar_image.py the following error is produced that does not hinder running the script:
UserWarning: Extinction files not found in /Users/ave_astro/pysyn_cdbs/extinction
 warnings.warn('Extinction files not found in %s' % (extdir, ))
UserWarning: No graph or component tables found; functionality will be SEVERELY crippled. No files found for /Users/ave_astro/pysyn_cdbs/mtab/*_tmg.fits
 warnings.warn('No graph or component tables found; '
UserWarning: No thermal tables found, no thermal calculations can be performed. No files found for /Users/ave_astro/pysyn_cdbs/mtab/*_tmt.fits
 warnings.warn('No thermal tables found, '.
## The extinction and mtab folder can be found here: https://pysynphot.readthedocs.io/en/stable/ but do not seem to be necessary at the moment.

'''
import numpy as np
from scipy import signal
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import os, sys
import matplotlib.pyplot as plt
# Spectra tools
import pysynphot as S
import webbpsf

from grizli.model import GrismFLT

import image_utils as iu

import yaml

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mkdirect", help="whether to make the direct image or not", default='y')
parser.add_argument("--fast_direct", help="if y, a single PSF is used over the whole detector when making the direct image", default='y')
parser.add_argument("--checkseg", help="check whether segmentation maps lines up properly", default='n')

parser.add_argument("--github_dir", help="path to directory where Roman GRS PIT github repos have been cloned; assumes it is the same for all", default=os.getenv('github_dir'))
parser.add_argument("--star_image_dir", help="directory to save star image and grism files", default=os.getenv('star_image_dir'))
parser.add_argument("--out_fn", help="output file name, written to star_image_dir", default='grism_test.fits')
# might need to make this a prefix if we want to propagate the SCA informations
parser.add_argument("--pad", help="padding in pixels to add to image", default=365, type=int)
parser.add_argument("--det", help="detector to simulate", default=1, type=int)
parser.add_argument("--center",help="telescope boresight (tel) or center of detector (det)",default='det')
parser.add_argument("--ngal",help="number of galaxies to simulate; all if None",default=None)
#These were used at first but should not be necessary, keeping for future debugging
#parser.add_argument("--input_star_fn", help="full path to file containing info on stars to simulate",default=os.getenv('github_dir')+'star_fields/py/stars_radec00.ecsv')
#parser.add_argument("--roman_base_dir", help="base directory for roman calibration files",default=os.getenv('roman_base_dir'))
#parser.add_argument("--roman_2022sim_dir", help="base directory for products from the 2022 simulations",default=os.getenv('roman_2022sim_dir'))


args = parser.parse_args()

#roman_base_dir = args.roman_base_dir
star_image_dir = args.star_image_dir
github_dir = args.github_dir
det_num = args.det

det = "SCA{:02}".format(det_num)

if not github_dir:
    print()
    print("MISSING github_dir! The github_dir needs to be set as an environmental variable or with the argument --github_dir")
    print()
    parser.print_help(sys.stderr)
    sys.exit(1)

if not star_image_dir:
    print()
    print("MISSING star_image_dir! The github_dir needs to be set as an environmental variable or with the argument --star_image_dir")
    print()
    parser.print_help(sys.stderr)
    sys.exit(1)


conf_file = os.path.join(github_dir, "grism_sim/data/grizli_config.yaml")
with open(conf_file) as f:
    grizli_conf = yaml.safe_load(f)

input_star_fn = os.path.join(github_dir, 'star_fields/py/stars_radec00.ecsv') #this was produced by the script in star_fields

if args.ngal is not None:
    ngal = int(args.ngal)

mockdir = '/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/'
#if ngal != 0:
input_gal_fn = mockdir+'Euclid_Roman_4deg2_radec.fits' #this is only on NERSC for now


pad = args.pad #user supplied padding, corresponding to the size in pixels for the psf and thus the size of each object in the direct image
gpad = grizli_conf["pad"] #grism padding needed based on configuration yaml
#Taking this out and instead using combination//we will make padding equal to the greater of the two
#if pad > gpad:
#    gpad = pad
#else:
#    pad = gpad
#print('the padding used throughout is '+str(pad))



#direct_fits_out = os.path.join(star_image_dir, 'ra0dec0_SCA1.fits')
#direct_fits_out_nopad = os.path.join(star_image_dir ,'ra0dec0_SCA1_nopad.fits')
direct_fits_out = os.path.join(star_image_dir, 'ra0dec0_%s.fits' % (det))
direct_fits_out_nopad = os.path.join(star_image_dir ,'ra0dec0_%s_nopad.fits' % (det))
nopad_seg = os.path.join(star_image_dir, "seg_nopad.fits")
pad_seg = os.path.join(star_image_dir, "seg_wpad.fits")
#example_direct = args.roman_2022sim_dir + 'products/FOV0/roll_0/dither_0x_0y/SCA1/GRS_FOV0_roll0_dx0_dy0_SCA1_direct_final.fits'

#this ends up setting the background noise and defines the WCS
import grizli.fake_image
ra, dec = 0, 0
pa_aper = 128.589
background = grizli_conf["grism_background"]
EXPTIME = 301 
NEXP = 1     

if args.center == 'tel':
    sys.path.append(github_dir+'/observing-program/py')
    import roman_coords_transform as ctrans
    code_data_dir = github_dir+'/observing-program/data/'
    rctrans = ctrans.RomanCoordsTransform(file_path=code_data_dir)
    dfoot = rctrans.wfi_sky_pointing(ra, dec, pa_aper, ds9=False)
    ra = dfoot[0][int(det_num)]['ra_cen']
    dec = dfoot[0][int(det_num)]['dec_cen']

print('will be centered on '+str(ra),str(dec))
#get WCS to use for double padded direct reference image

tot_im_size = 4088+2*(gpad+pad)

im_head = iu.fake_header_wcs(ra, dec, crpix2=tot_im_size/2,crpix1=tot_im_size/2, cdelt1=0.11, cdelt2=0.11,crota2=pa_aper,naxis1=tot_im_size,naxis2=tot_im_size)
im_wcs = WCS(im_head)

stars00 = Table.read(input_star_fn)

star_coords = coords_test = SkyCoord(ra=stars00['RA']*u.degree,dec=stars00['DEC']*u.degree, frame='icrs')
star_xy = im_wcs.world_to_pixel(star_coords)
print('range of x y values in input star catalog:')
print(min(star_xy[0]),max(star_xy[0]),min(star_xy[1]),max(star_xy[1]))

sel_ondet = star_xy[0] > 0#stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
sel_ondet &= star_xy[0] < 4088 + 2*( gpad)
sel_ondet &= star_xy[1] > 0#stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
sel_ondet &= star_xy[1] < 4088 + 2*( gpad)

print('cutting stars to be on detector + padded area')
stars00 = stars00[sel_ondet]
print(stars00['Xpos'].shape)
stars00['Xpos'] = star_xy[0][sel_ondet]
stars00['Ypos'] = star_xy[1][sel_ondet]
Ntot= len(stars00)

print(star_xy[0][sel_ondet].shape)

#if ngal != 0:
h=0.6774
Mpc = 3.08568025E24 # cm
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.3089, Tcmb0=2.725)


gals = fits.open(input_gal_fn)[1].data

gal_coords = SkyCoord(ra=gals['RA']*u.degree,dec=gals['DEC']*u.degree, frame='icrs')
gal_xy = im_wcs.world_to_pixel(gal_coords)
print('range of x y values in input galaxy catalog:')
print(min(gal_xy[0]),max(gal_xy[0]),min(gal_xy[1]),max(gal_xy[1]))

sel_ondet = gal_xy[0] > 0#stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
sel_ondet &= gal_xy[0] < 4088 + 2*( gpad)
sel_ondet &= gal_xy[1] > 0#stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
sel_ondet &= gal_xy[1] < 4088 + 2*( gpad)
gals = Table(gals[sel_ondet])
gals['Xpos'] = gal_xy[0][sel_ondet]
gals['Ypos'] = gal_xy[1][sel_ondet]
lum_distance = cosmo.luminosity_distance(gals['Z']).value
lum_distance_cm = lum_distance*Mpc # cm
flux = gals['tot_Lum_F158_Av1.6523']/(4.0*np.pi*lum_distance_cm**2.0)
gals['flux'] = flux
mag = -2.5*np.log10(flux)+26.5
gals['mag'] = mag
#gal_xy = gal_xy[sel_ondet]
if ngal is None:
    ngal = len(gals)
print('number of galaxies within detector padded region is '+str(ngal))



#empty_grism = roman_base_dir+'roman_empty_starfield_test.fits'
empty_grism = os.path.join(star_image_dir, 'roman_empty_starfield_test.fits')
h, wcs = grizli.fake_image.roman_header(ra=ra, dec=dec, pa_aper=pa_aper, naxis=(4088,4088))
head = wcs.to_header()
grizli.fake_image.make_fake_image(h, output=empty_grism, exptime=EXPTIME, nexp=NEXP, background=background)
file = fits.open(empty_grism)
#file[1].header["CONFFILE"] = os.path.join(github_dir, "grism_sim/data/Roman.det1.07242020.conf") #roman_base_dir+"configuration/Roman.det1.07242020.conf" # This had to be a path, not just a filename; otherwise, grizli can't find the sensitivity fits
file[1].header["CONFFILE"] = os.path.join(github_dir, "grism_sim/data/Roman.det%i.07242020.conf" % (det_num))
file.writeto(empty_grism, overwrite=True)
file.close()


if args.mkdirect == 'y':
    #This takes ~6 minutes and is by far the greatest processing time
    #The time is dominated by calculating the PSF for each object...should switch to using  a grid: https://webbpsf.readthedocs.io/en/latest/psf_grids.html
    #If you see obvious ways to speed it up, please test their implementation and make PR!
    full_image = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)))
    full_seg = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)),dtype=int)
    thresh = 0.01 #threshold flux for segmentation map
    N = 0
    if args.fast_direct == 'y':
        fid_psf = iu.get_psf(fov_pixels=pad-1, det=det)
    #stars00 = Table.read(args.input_star_fn)
    for i in range(0,len(stars00)):
        xpos = stars00[i]['Xpos']
        ypos = stars00[i]['Ypos']
        #print(xpos,ypos)
        mag = stars00[i]['magnitude']
        if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
            print(xpos,ypos,'out of bounds position')
        xp = int(xpos)
        yp = int(ypos)
        xoff = 0#xpos-xp
        yoff = 0#ypos-yp
        if args.fast_direct == 'y':
            sp = iu.star_postage_inpsf(mag,fid_psf)
        else:
            sp = iu.star_postage(mag,xpos,ypos,xoff,yoff,fov_pixels=pad-1, det=det)
        fov_pixels = pad-1
        full_image[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels] += sp
        selseg = sp > thresh
        #seg = np.zeros((len(sp),len(sp)),dtype=int)
        #seg[selseg] = i+1
        #set instead of add; any blends end up being last added source
        #but, then go back and return the values that got set to zero back to their original values
        #full_segold = np.copy(full_seg)
        full_seg[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels][selseg] = i+1#seg 
        #sel = full_seg != i+1
        #full_seg[sel] = full_segold[sel]
        N += 1
        if N//10 == N/10:
            print(N,Ntot,len(np.unique(full_seg)),i+1)
    
    phdu = fits.PrimaryHDU()
    ihdu = fits.ImageHDU(data=full_image,name='SCI')
    err_array = np.zeros(full_image.shape) #this gets over-written for cut image currently, 0s are bad for grizli if it actually gets used
    ehdu = fits.ImageHDU(data=err_array,name='ERR')
    hdul = fits.HDUList([phdu,ihdu,ehdu])
    hdul[0].header["INSTRUME"] = "ROMAN"
    #hdul[0].header["FILTER"] = "f140w"
    
    hdul.writeto(direct_fits_out, overwrite=True)
    
    hdu = fits.PrimaryHDU(data=full_seg[pad:-pad,pad:-pad])
    hdul = fits.HDUList([hdu])
    hdu.writeto(nopad_seg, overwrite=True)
    hdu = fits.PrimaryHDU(data=full_seg)
    hdul = fits.HDUList([hdu])
    hdu.writeto(pad_seg, overwrite=True)

file = fits.open(direct_fits_out)
cut_image = file[1].data[pad:-pad,pad:-pad]
phdu = fits.PrimaryHDU(data=cut_image)
phdu.header["INSTRUME"] = 'ROMAN   '
phdu.header["FILTER"] = "f140w"
phdu.header["EXPTIME"] = 141
shp = cut_image.shape
phdu.header = iu.add_wcs(phdu,ra, dec, crpix2=shp[1]/2,crpix1=shp[0]/2, cdelt1=0.11, cdelt2=0.11,
                crota2=pa_aper,naxis1=shp[0],naxis2=shp[1])

#print(phdu.header)

#plt.imshow(np.log(cut_image+.01))
#plt.show()
err = np.random.poisson(10,cut_image.shape)*0.001 #np.zeros(cut_image.shape)
ihdu = fits.ImageHDU(data=cut_image,name='SCI',header=phdu.header)
ehdu = fits.ImageHDU(data=err,name='ERR',header=phdu.header)
dhdu = fits.ImageHDU(data=np.zeros(cut_image.shape),name='DQ',header=phdu.header)
hdul = fits.HDUList([phdu,ihdu,ehdu,dhdu])
#hdul = fits.HDUList([ihdu,ehdu,dhdu])

hdul.writeto(direct_fits_out_nopad, overwrite=True)
#hdul.writeto(direct_fits_out, overwrite=True)

#filetest = fits.open(direct_fits_out_nopad)
#for i in range(0,len(filetest)):
#    print(filetest[i].header.keys)

size = grizli_conf["size"][det]

#roman = GrismFLT(grism_file=empty_grism,direct_file=direct_fits_out_nopad, seg_file=None, pad=gpad)
roman = GrismFLT(grism_file=empty_grism,ref_file=direct_fits_out_nopad, seg_file=None, pad=gpad)
testf = fits.open(nopad_seg)
#roman = GrismFLT(grism_file=empty_grism,direct_file=direct_fits_out_nopad, seg_file=None, pad=gpad)
#testf = fits.open(pad_seg)


masked_seg = testf[0].data
#print('number of unique values in segmentation map '+str(len(np.unique(masked_seg))),np.min(masked_seg),np.max(masked_seg))
if args.checkseg == 'y':
    for i in range(0,len(stars00)):
        #sel_row = stars00['index'] == photid
        photid = i+1
        xpos = int(stars00[i]['Xpos'])
        ypos = int(stars00[i]['Ypos'])
        print(photid,masked_seg[xpos][ypos],cut_image[xpos][ypos])

#padded_masked_seg = np.pad(masked_seg, [gpad, gpad], mode='constant')
#roman.seg = padded_masked_seg.astype("float32")

roman.seg = masked_seg.astype("float32") #this segmentation map should have the area of the padded grism image, but not have the padding added because of the PSF size

print(masked_seg.shape,cut_image.shape)
#print('added segmentation map')

#SED_dir = roman_base_dir+"FOV0/SEDs/" # Change to your path to directory containing SEDs

# Create F158 Filter Bandpass object

#df = Table.read(os.path.join(SED_dir, "wfirst_wfi_f158_001_syn.fits"), format='fits') #close to H-band
df = Table.read(os.path.join(github_dir, 'grism_sim/data/wfirst_wfi_f158_001_syn.fits'), format='fits') #close to H-band
bp = S.ArrayBandpass(df["WAVELENGTH"], df["THROUGHPUT"])

minlam = grizli_conf["minlam"]
maxlam = grizli_conf["maxlam"]

tempdir = os.path.join(github_dir, 'star_fields/data/SEDtemplates/')
templates = open(os.path.join(github_dir, 'star_fields/data/SEDtemplates/input_spectral_STARS.lis')).readlines()
temp_inds = stars00['star_template_index'] - 58*(stars00['star_template_index']//58)

count = 0
print('about to simulate grism')
for i in range(0,len(stars00)):
    photid = i+1
    row = stars00[i]
    mag = row["magnitude"]
    temp_ind = int(temp_inds[i])
    #print(temp_ind)
    star_type = templates[temp_ind].strip('\n')
    temp = np.loadtxt(os.path.join(tempdir, star_type)).transpose()
    wave = temp[0]
    sel = wave > minlam
    sel &= wave < maxlam
    wave = wave[sel]
    flux = temp[1]
    flux = flux[sel]
    star_spec = S.ArraySpectrum(wave=wave, flux=flux, waveunits="angstroms", fluxunits="flam")
    spec = star_spec.renorm(mag, "abmag", bp)
    spec.convert("flam")

    #print('made it to grism step')
    # size is read in from grizli_config.yaml above
    #print(row)
    roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=True, store=False,
                               is_cgs=True, spectrum_1d=[spec.wave, spec.flux])
    count += 1
    #print(count)

testprof = np.zeros((4,4)) #just something that is not a pointsource, this should get much better
wave = np.linspace(2000, 40000, 19001) #wavelength grid for simulation
sel_wave = wave > minlam
sel_wave &= wave < maxlam
wave = wave[sel_wave]
    
for i in range(0,ngal):
    photid += 1
    row = gals[i]
    mag = row['mag']
    imflux = row['flux']
    #make image, put it in reference
    full_image = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)))
    full_seg = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)),dtype=int)
    thresh = 0.01 #threshold flux for segmentation map
    N = 0
    if args.fast_direct == 'y':
        conv_prof = signal.convolve2d(fid_psf,testprof,mode='same')
    else:
        print('need to write something for non-fixed psf')
        break
    xpos = row['Xpos']
    ypos = row['Ypos']
    if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
        print(xpos,ypos,'out of bounds position')
    xp = int(xpos)
    yp = int(ypos)
    xoff = 0#xpos-xp
    yoff = 0#ypos-yp
    sp = imflux*conv_prof
    fov_pixels = pad-1
    full_image[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels] += sp
    masked_im = full_image[pad:-pad,pad:-pad]
    #copying from process_ref_file in grizli
    roman.direct.data['REF'] = masked_im
    roman.direct.data['REF'] *= roman.direct.ref_photflam
    
    selseg = sp > thresh
    full_seg[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels][selseg] = photid
    masked_seg = full_seg[pad:-pad,pad:-pad]
    roman.seg = masked_seg.astype("float32")
    
    #get sed and convert to spectrum
    sim_fn = mockdir+'galacticus_FOV_EVERY100_sub_'+str(row['SIM'])+'.hdf5'
    sim = h5py.File(sim_fn, 'r')
    sed = sim['Outputs']['SED:observed:dust:Av1.6523'][row['IDX']]
    flux = sed[sel_wave]
    gal_spec = S.ArraySpectrum(wave=wave, flux=flux, waveunits="angstroms", fluxunits="flam")
    spec = gal_spec.renorm(mag, "abmag", bp)
    spec.convert("flam")
    
    roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=True, store=False,
                               is_cgs=True, spectrum_1d=[spec.wave, spec.flux])

print(roman.model.shape)

plt.imshow(roman.model, vmax=0.2, cmap="hot")
plt.colorbar()
plt.title('with padding')
plt.show()


if gpad != 0:
    plt.imshow(roman.model[gpad:-gpad, gpad:-gpad], vmax=0.2, cmap="hot")
    plt.title('cut to 4088x4088')
    plt.colorbar()
    plt.show()



#save grism model image + noise
out_fn = os.path.join(args.star_image_dir, args.out_fn)
hdu_list = fits.open(empty_grism)
if gpad != 0:
    hdu_list.append(fits.ImageHDU(data=roman.model[gpad:-gpad, gpad:-gpad],name='MODEL'))
    #hdu_list.append(fits.ImageHDU(data=roman.grism.data['SCI'][gpad:-gpad, gpad:-gpad],name='ERR'))
    hdu_list['ERR'].data = roman.grism.data['SCI'][gpad:-gpad, gpad:-gpad]
else:
    hdu_list.append(fits.ImageHDU(data=roman.model,name='MODEL'))
    #hdu_list.append(fits.ImageHDU(data=roman.grism.data['SCI']),name='ERR')
    hdu_list['ERR'].data = roman.grism.data['SCI']

hdu_list.writeto(out_fn, overwrite=True)
hdu_list.close()
print('wrote to '+out_fn)

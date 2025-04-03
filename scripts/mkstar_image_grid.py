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
from time import time

start = time() #! TIMING

import numpy as np
from astropy.io import fits
from astropy.table import Table
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

pad = args.pad #user supplied padding, corresponding to the size in pixels for the psf and thus the size of each object in the direct image
gpad = grizli_conf["pad"] #grism padding needed based on configuration yaml
#Taking this out and instead using combination//we will make padding equal to the greater of the two
#if pad > gpad:
#    gpad = pad
#else:
#    pad = gpad
#print('the padding used throughout is '+str(pad))


stars00 = Table.read(input_star_fn)
sel_ondet = stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
sel_ondet &= stars00['Ypos'] < 4088 + 2*( gpad)
print('check no negative x,y positions:')
print(np.min(stars00['Xpos'] ),np.min(stars00['Ypos']))
stars00 = stars00[sel_ondet]
Ntot= len(stars00)

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

start_direct = time() #! TIMING

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
    else:
        psf_grid = iu.create_psf_grid(fov_pixels=pad-1)
    #stars00 = Table.read(args.input_star_fn)

    start_postage = time() #! TIMING
    cumulative_psf = 0
    cumulative_seg = 0

    for i in range(0,len(stars00)):
        xpos = stars00[i]['Xpos']
        ypos = stars00[i]['Ypos']
        mag = stars00[i]['magnitude']
        if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
            print(xpos,ypos,'out of bounds position')
        xp = int(xpos)
        yp = int(ypos)
        xoff = 0#xpos-xp
        yoff = 0#ypos-yp

        start_psf = time() #! TIMING
        if args.fast_direct == 'y':
            sp = iu.star_postage_inpsf(mag,fid_psf)
        else:
            sp = iu.star_postage_grid(psf_grid,mag,xpos,ypos,fov_pixels=pad-1)
        end_psf = time() #! TIMING
        fov_pixels = pad-1
        full_image[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels] += sp
        start_seg = time() #! TIMING
        selseg = sp > thresh
        full_seg[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels][selseg] = i+1
        end_seg = time() #! TIMING
        N += 1
        if N//10 == N/10:
            print(N,Ntot,len(np.unique(full_seg)),i+1)

        cumulative_psf += (end_psf - start_psf) #! TIMING
        cumulative_seg += (end_seg - start_seg)
    
    end_postage = time() #! TIMING

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

end_direct = time() #! TIMING

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

start_grism = time() #! TIMING

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

end_grism = time() #! TIMING

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

end = time() #! TIMING

print(
	"""
	Timings:
	Overall runtime: %f
	Direct time:     %f
	Postage time:    %f
	Grism time:      %f
	
	Cumulative PSF:  %f
	Cumulative Seg:  %f
	
	"""%((end - start), (end_direct - start_direct), (end_postage - start_postage), (end_grism - start_grism), cumulative_psf, cumulative_seg)
)
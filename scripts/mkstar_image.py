import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
import matplotlib.pyplot as plt
# Spectra tools
import pysynphot as S
import webbpsf

from grizli.model import GrismFLT

import image_utils as iu

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mkdirect", help="whether to make the direct image or not",default='y')
parser.add_argument("--checkseg", help="check whether segmentation maps lines up properly",default='n')

parser.add_argument("--roman_base_dir", help="base directory for roman calibration files",default=os.getenv('roman_base_dir'))
parser.add_argument("--roman_2022sim_dir", help="base directory for products from the 2022 simulations",default=os.getenv('roman_2022sim_dir'))
parser.add_argument("--star_image_dir", help="directory to save star image files",default=os.getenv('star_image_dir'))
parser.add_argument("--input_star_fn", help="full path to file containing info on stars to simulate",default=os.getenv('github_dir')+'star_fields/py/stars_radec00.ecsv')
parser.add_argument("--pad", help="padding in pixels to add to image",default=365,type=int)

args = parser.parse_args()

roman_base_dir = args.roman_base_dir
star_image_dir = args.star_image_dir
pad = args.pad

stars00 = Table.read(args.input_star_fn)
sel_ondet = stars00['Xpos'] < 4088
sel_ondet &= stars00['Ypos'] < 4088
print('check no negative x,y positions:')
print(np.min(stars00['Xpos'] ),np.min(stars00['Ypos']))
stars00 = stars00[sel_ondet]
Ntot= len(stars00)

direct_fits_out = star_image_dir + 'ra0dec0_SCA1.fits'
direct_fits_out_nopad = star_image_dir + 'ra0dec0_SCA1_nopad.fits'
nopad_seg = star_image_dir +"seg_nopad.fits"
example_direct = args.roman_2022sim_dir + 'products/FOV0/roll_0/dither_0x_0y/SCA1/GRS_FOV0_roll0_dx0_dy0_SCA1_direct_final.fits'

import grizli.fake_image
ra, dec = 0, 0
pa_aper = 128.589
background = 0.57
EXPTIME = 301 # 10 ks ~ 4 HST orbits
NEXP = 1     # divided between 10 exposures

empty_grism = roman_base_dir+'roman_empty_starfield_test.fits'
h, wcs = grizli.fake_image.roman_header(ra=ra, dec=dec, pa_aper=pa_aper, naxis=(4088,4088))
head = wcs.to_header()
grizli.fake_image.make_fake_image(h, output=empty_grism, exptime=EXPTIME, nexp=NEXP,background=background)
file = fits.open(empty_grism)
file[1].header["CONFFILE"] = roman_base_dir+"configuration/Roman.det1.07242020.conf" # This had to be a path, not just a filename; otherwise, grizli can't find the sensitivity fits
file.writeto(empty_grism, overwrite=True)
file.close()


if args.mkdirect == 'y':

	full_image = np.zeros((4088+2*pad,4088+2*pad))
	full_seg = np.zeros((4088+2*pad,4088+2*pad),dtype=int)
	thresh = 0.01 #threshold flux for segmentation map
	N = 0
	#stars00 = Table.read(args.input_star_fn)
	for i in range(0,len(stars00)):
		xpos = stars00[i]['Xpos']
		ypos = stars00[i]['Ypos']
		mag = stars00[i]['magnitude']
		if xpos > 4088 or ypos > 4088:
		    print(xpos,ypos,'out of bounds position')
		xp = int(xpos)
		yp = int(ypos)
		xoff = 0#xpos-xp
		yoff = 0#ypos-yp
		sp = iu.star_postage(mag,xpos,ypos,xoff,yoff,fov_pixels=pad-1)
		fov_pixels = pad-1
		full_image[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels] += sp
		selseg = sp > thresh
		seg = np.zeros((len(sp),len(sp)),dtype=int)
		seg[selseg] = i+1
		#set instead of add; any blends end up being last added source
		#but, then go back and return the values that got set to zero back to their original values
		full_segold = np.copy(full_seg)
		full_seg[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels] = seg 
		sel = full_seg != i+1
		full_seg[sel] = full_segold[sel]
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

file = fits.open(direct_fits_out)
phdu = fits.PrimaryHDU()
cut_image = file[1].data[pad:-pad,pad:-pad]
#plt.imshow(np.log(cut_image+.01))
#plt.show()
err = np.random.poisson(10,cut_image.shape)*0.001 #np.zeros(cut_image.shape)
ihdu = fits.ImageHDU(data=cut_image,name='SCI',header=head)
ehdu = fits.ImageHDU(data=err,name='ERR',header=head)
dhdu = fits.ImageHDU(data=np.zeros(cut_image.shape),name='DQ')
hdul = fits.HDUList([phdu,ihdu,ehdu,dhdu])
#example_file = fits.open(example_direct)
#for ii in range(0,len(hdul)):
#    hdul[ii].header = example_file[ii].header #eventually, we will want real varying WCS

hdul[0].header["INSTRUME"] = 'ROMAN   '
hdul[0].header["FILTER"] = "f140w"
hdul[0].header["EXPTIME"] = 141
	#hdul[0].header["FILTER"] = "f140w"
	
hdul.writeto(direct_fits_out_nopad, overwrite=True)

	#file[1].data = file[1].data[pad:-pad,pad:-pad]
	#print(file[1].data.shape)
	#file[2].data = file[2].data[pad:-pad,pad:-pad]
	#file.writeto(direct_fits_out_nopad, overwrite=True)
	#file.close()


#roman = GrismFLT(grism_file=roman_base_dir+'roman_empty.fits',direct_file=direct_fits_out_nopad,ref_file=ref_fits_out, seg_file=None, pad=pad)
#Get WCS info correct for ref file and it might work; make padding smaller
gpad = 100


roman = GrismFLT(grism_file=empty_grism,direct_file=direct_fits_out_nopad, seg_file=None, pad=gpad)
testf = fits.open(nopad_seg)
masked_seg = testf[0].data
print('number of unique values in segmentation map '+str(len(np.unique(masked_seg))),np.min(masked_seg),np.max(masked_seg))
if args.checkseg == 'y':
	for i in range(0,len(stars00)):
		#sel_row = stars00['index'] == photid
		photid = i+1
		xpos = int(stars00[i]['Xpos'])
		ypos = int(stars00[i]['Ypos'])
		print(photid,masked_seg[xpos][ypos],cut_image[xpos][ypos])

padded_masked_seg = np.pad(masked_seg, [gpad, gpad], mode='constant')
roman.seg = padded_masked_seg.astype("float32")

print('added segmentation map')

SED_dir = roman_base_dir+"FOV0/SEDs/" # Change to your path to directory containing SEDs

# Create F158 Filter Bandpass object
df = Table.read(os.path.join(SED_dir, "wfirst_wfi_f158_001_syn.fits"), format='fits') #close to H-band
bp = S.ArrayBandpass(df["WAVELENGTH"], df["THROUGHPUT"])

minlam = 1e4
maxlam=2e4
github_dir=os.getenv('github_dir')
tempdir = github_dir+'star_fields/data/SEDtemplates/'
templates = open(github_dir+'star_fields/data/SEDtemplates/input_spectral_STARS.lis').readlines()
temp_inds = stars00['star_template_index'] - 58*(stars00['star_template_index']//58)

count = 0
#for (segid,photid) in zip(found_segids,found_ids):
print('about to simulate grism')
for i in range(0,len(stars00)):
    #sel_row = stars00['index'] == photid
    photid = i+1
    row = stars00[i]#stars00[sel_row]
    mag = row["magnitude"]#[0]
    temp_ind = int(temp_inds[i])#[sel_row])[0])#int(row['star_template_index'][0])
    #print(temp_ind)
    star_type = templates[temp_ind].strip('\n')
    temp = np.loadtxt(tempdir+star_type).transpose()
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
    # By default, grizli trys to compute a cutout size. This cutout size is not large enough for the roman grism.
    # In 4) FOV0_sims/notebooks/dy-by-optimize.ipynb, I estimate the maximum needed size to be 77 for detector 1.
    # See that notebook for more details
    #segid changed to photid with new seg map
    roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=77, in_place=True, store=False,
                               is_cgs=True, spectrum_1d=[spec.wave, spec.flux])
    count += 1
    #print(count)
if gpad != 0:
	plt.imshow(roman.model[gpad:-gpad, gpad:-gpad], vmax=0.2, cmap="hot")
else:
	plt.imshow(roman.model, vmax=0.2, cmap="hot")
plt.colorbar()
plt.show()
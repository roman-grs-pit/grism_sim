from tqdm import tqdm
import h5py
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
import grizli.fake_image

import image_utils as iu

import yaml

github_dir_env=os.getenv('github_dir')
if github_dir_env is None:
    print('github_dir environment variable has not been set, will cause problems if not explicitly set in function calss')

def mk_ref_image(tel_ra,tel_dec,pa,det_num,star_input,gal_input,output_dir,psf_cutout_size=365,github_dir=github_dir_env,gal_mag_col='mag_F158_Av1.6523',dogal='y',magmax=25):
    #tel_ra,tel_dec correspond to the coordinates (in degrees) of the middle of the field (not the detector center)
    #pa is the position angle (in degrees)
    #det_num is an integer corresponding to the detector number
    #star_input is a table or array with columns 'RA', 'DEC', 'magnitude', 'star_template_index'
    #gal_input is a table or array with columns...
    #output_dir is the directory for output
    #psf_cutout_size is the size in pixels used to determine the psf and then all of the image stamps that get added together
    det = "SCA{:02}".format(det_num)
    conf_file = os.path.join(github_dir, "grism_sim/data/grizli_config.yaml")
    with open(conf_file) as f:
        grizli_conf = yaml.safe_load(f)
    
    pad = psf_cutout_size
    gpad = grizli_conf["pad"]
    fn_root = 'refimage_ra%s_dec%s_pa%s_det%s' % (tel_ra,tel_dec,pa,det)
    direct_fits_out = os.path.join(output_dir,fn_root+'.fits' )
    direct_fits_out_nopad = os.path.join(output_dir,fn_root+'_nopad.fits')
    nopad_seg = os.path.join(output_dir,fn_root+ "_seg_nopad.fits")
    pad_seg = os.path.join(output_dir,fn_root+ "seg_wpad.fits")
    #example_direct = args.roman_2022sim_dir + 'products/FOV0/roll_0/dither_0x_0y/SCA1/GRS_FOV0_roll0_dx0_dy0_SCA1_direct_final.fits'
    
    #this ends up setting the background noise and defines the WCS
        
    background = grizli_conf["grism_background"]
    EXPTIME = 301 
    NEXP = 1     

    sys.path.append(github_dir+'/observing-program/py')
    import roman_coords_transform as ctrans
    code_data_dir = github_dir+'/observing-program/data/'
    rctrans = ctrans.RomanCoordsTransform(file_path=code_data_dir)
    dfoot = rctrans.wfi_sky_pointing(tel_ra, tel_dec, pa, ds9=False)
    ra = dfoot[0][int(det_num)]['ra_cen']
    dec = dfoot[0][int(det_num)]['dec_cen']

    tot_im_size = 4088+2*(gpad+pad)

    im_head = iu.fake_header_wcs(ra, dec, crpix2=tot_im_size/2,crpix1=tot_im_size/2, cdelt1=0.11, cdelt2=0.11,crota2=pa,naxis1=tot_im_size,naxis2=tot_im_size)
    im_wcs = WCS(im_head)

    star_coords = SkyCoord(ra=star_input['RA']*u.degree,dec=star_input['DEC']*u.degree, frame='icrs')
    star_xy = im_wcs.world_to_pixel(star_coords)
    
    sel_ondet = star_xy[0] > 0#stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
    sel_ondet &= star_xy[0] < 4088 + 2*( gpad)
    sel_ondet &= star_xy[1] > 0#stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
    sel_ondet &= star_xy[1] < 4088 + 2*( gpad)
    
    print('cutting stars to be on detector + padded area')
    stars = star_input[sel_ondet]
    #print(stars00['Xpos'].shape)
    stars['Xpos'] = star_xy[0][sel_ondet]
    stars['Ypos'] = star_xy[1][sel_ondet]
    Ntot= len(stars)
    ngal = 0
    fid_psf = iu.get_psf(fov_pixels=pad-1, det=det)
    if dogal == 'y':
        gal_coords = SkyCoord(ra=(gal_input['RA'])*u.degree,dec=gal_input['DEC']*u.degree, frame='icrs')
        gal_xy = im_wcs.world_to_pixel(gal_coords)
    
        sel_ondet = gal_xy[0] > 0
        sel_ondet &= gal_xy[0] < 4088 + 2*( gpad)
        sel_ondet &= gal_xy[1] > 0
        sel_ondet &= gal_xy[1] < 4088 + 2*( gpad)
        gals = Table(gal_input[sel_ondet])
        gals['Xpos'] = gal_xy[0][sel_ondet]
        gals['Ypos'] = gal_xy[1][sel_ondet]
        gals.rename_column(gal_mag_col, 'mag')
        sel_mag = gals['mag'] < magmax
        gals = gals[sel_mag]
        ngal = len(gals)
        print('number of galaxies within detector padded region is '+str(ngal))

        #fiducial galaxy profile
        
        r_eff = 2.5 #radius for profile in pixels
        x, y = np.meshgrid(np.arange(-15,15), np.arange(-15,15)) #30x30 grid of pixels
        from astropy.modeling.models import Sersic2D
        round_exp = Sersic2D(amplitude=1, r_eff=r_eff,n=1) #round exponential 
        testprof = round_exp(x,y) #np.ones((4,4)) #just something that is not a pointsource, this should get much better
        testprof /= np.sum(testprof) #normalize the profile
        conv_prof_fixed = signal.convolve2d(fid_psf[0].data,testprof,mode='same')

    full_image = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)))
    full_seg = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)),dtype=int)
    thresh = 0.01 #threshold flux for segmentation map
    N = 0
    for i in range(0,len(stars)):
        xpos = stars[i]['Xpos']
        ypos = stars[i]['Ypos']
        mag = stars[i]['magnitude']
        if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
            print(xpos,ypos,'out of bounds position')
        xp = int(xpos)
        yp = int(ypos)
        xoff = 0#xpos-xp
        yoff = 0#ypos-yp
        
        sp = iu.star_postage_inpsf(mag,fid_psf)
        #can code something here to not use constant psf
        #else:
        #    sp = iu.star_postage(mag,xpos,ypos,xoff,yoff,fov_pixels=pad-1, det=det)
        fov_pixels = pad-1
        full_image[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels] += sp
        selseg = sp > thresh
        full_seg[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels][selseg] = i+1#seg 

    if ngal > 0:
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
            fov_pixels = pad-1
            full_image[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels] += sp

    
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
    phdu.header = iu.add_wcs(phdu,ra, dec, crpix2=shp[1]/2,crpix1=shp[0]/2, cdelt1=0.11, cdelt2=0.11,
                crota2=pa,naxis1=shp[0],naxis2=shp[1])

    err = np.random.poisson(10,cut_image.shape)*0.001 #np.zeros(cut_image.shape)
    ihdu = fits.ImageHDU(data=cut_image,name='SCI',header=phdu.header)
    ehdu = fits.ImageHDU(data=err,name='ERR',header=phdu.header)
    dhdu = fits.ImageHDU(data=np.zeros(cut_image.shape),name='DQ',header=phdu.header)
    hdul = fits.HDUList([phdu,ihdu,ehdu,dhdu])
    hdul.writeto(direct_fits_out_nopad, overwrite=True)
    
    return True


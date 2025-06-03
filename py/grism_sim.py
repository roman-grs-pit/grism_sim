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

from grizli.model import GrismFLT
import grizli.fake_image

import image_utils as iu

import yaml

github_dir_env=os.getenv('github_dir')
if github_dir_env is None:
    print('github_dir environment variable has not been set, will cause problems if not explicitly set in function calss')


def mk_ref_and_grism(tel_ra,tel_dec,pa,det_num,star_input,gal_input,output_dir,confver='07242020',psf_cutout_size=365,extra_grism_name='',github_dir=github_dir_env,gal_mag_col='mag_F158_Av1.6523',dogal='y',magmax=25,mockdir='/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/'):
    #tel_ra,tel_dec correspond to the coordinates (in degrees) of the middle of the field (not the detector center)
    #pa is the position angle (in degrees), relative to lines of ra=constant; note, requires +60 on pa for wfi_sky_pointing
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
    dfoot = rctrans.wfi_sky_pointing(tel_ra, tel_dec, pa+60, ds9=False)
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
        fn_galout = os.path.join(output_dir,'gals_'+fn_root+ ".fits")
        gals.write(fn_galout,overwrite=True)
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
    phdu.header = iu.add_wcs(phdu,ra, dec, crpix2=shp[1]/2,crpix1=shp[0]/2, cdelt1=0.11, cdelt2=0.11,
                crota2=pa,naxis1=shp[0],naxis2=shp[1])

    err = np.random.poisson(10,cut_image.shape)*0.001 #np.zeros(cut_image.shape)
    ihdu = fits.ImageHDU(data=cut_image,name='SCI',header=phdu.header)
    ehdu = fits.ImageHDU(data=err,name='ERR',header=phdu.header)
    dhdu = fits.ImageHDU(data=np.zeros(cut_image.shape),name='DQ',header=phdu.header)
    hdul = fits.HDUList([phdu,ihdu,ehdu,dhdu])
    hdul.writeto(direct_fits_out_nopad, overwrite=True)

    fn_root_grism = 'grism_ra%s_dec%s_pa%s_det%s' % (tel_ra,tel_dec,pa,det)
    fn_root_grism += extra_grism_name 
    empty_grism = os.path.join(output_dir, 'empty_'+fn_root_grism+'.fits')
    h, wcs = grizli.fake_image.roman_header(ra=ra, dec=dec, pa_aper=pa, naxis=(4088,4088))
    head = wcs.to_header()
    grizli.fake_image.make_fake_image(h, output=empty_grism, exptime=EXPTIME, nexp=NEXP, background=background)
    file = fits.open(empty_grism)
    file[1].header["CONFFILE"] = os.path.join(github_dir, "grism_sim/data/Roman.det"+str(det_num)+"."+confver+".conf") #% (det_num,confver))
    file.writeto(empty_grism, overwrite=True)
    file.close()

    size = grizli_conf["size"][det]
    
    
    roman = GrismFLT(grism_file=empty_grism,ref_file=direct_fits_out_nopad, seg_file=None, pad=gpad)
    masked_seg = fits.open(nopad_seg)[0].data       
    roman.seg = masked_seg.astype("float32") #this segmentation map should have the area of the padded grism image, but not have the padding added because of the PSF size
    
    df = Table.read(os.path.join(github_dir, 'grism_sim/data/wfirst_wfi_f158_001_syn.fits'), format='fits') #close to H-band
    bp = S.ArrayBandpass(df["WAVELENGTH"], df["THROUGHPUT"])
    
    minlam = grizli_conf["minlam"]
    maxlam = grizli_conf["maxlam"]
    
    tempdir = os.path.join(github_dir, 'star_fields/data/SEDtemplates/')
    templates = open(os.path.join(github_dir, 'star_fields/data/SEDtemplates/input_spectral_STARS.lis')).readlines()
    temp_inds = stars['star_template_index'] - 58*(stars['star_template_index']//58)
    
    count = 0
    print('about to simulate grism')
    for i in tqdm(range(0,len(stars))):
        photid = i+1
        row = stars[i]
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
    
    wave = np.linspace(2000, 40000, 19001) #wavelength grid for simulation
    sel_wave = wave > minlam
    sel_wave &= wave < maxlam
    wave = wave[sel_wave]
    
        
    for i in tqdm(range(0,ngal)):
        photid += 1
        row = gals[i]
        mag = row['mag']
        imflux = iu.mag2flux(mag)#imflux = row['flux']
        #make image, put it in seg
        full_image = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)))
        full_seg = np.zeros((4088+2*(gpad+pad),4088+2*(gpad+pad)),dtype=int)
        thresh = 0.01 #threshold flux for segmentation map
        N = 0
        #if args.fast_direct == 'y':
        conv_prof = conv_prof_fixed#signal.convolve2d(fid_psf[0].data,testprof,mode='same')
        #else:
        #   print('need to write something for non-fixed psf')
        #   break
        xpos = row['Xpos']
        ypos = row['Ypos']
        #if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
        #    print(xpos,ypos,'out of bounds position')
        xp = int(xpos)
        yp = int(ypos)
        xoff = 0#xpos-xp
        yoff = 0#ypos-yp
        sp = imflux*conv_prof
        fov_pixels = pad-1
        full_image[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels] += sp
        masked_im = full_image[pad:-pad,pad:-pad]
        #copying from process_ref_file in grizli
        #roman.direct.data['REF'] = np.asarray(masked_im,dtype=np.float32)
        #roman.direct.data['REF'] *= roman.direct.ref_photflam
        
        selseg = sp > thresh
        full_seg[xp+pad-fov_pixels:xp+pad+fov_pixels,yp+pad-fov_pixels:yp+pad+fov_pixels][selseg] = photid
        masked_seg = full_seg[pad:-pad,pad:-pad]
        roman.seg = np.rot90(np.asarray(masked_seg,dtype=np.float32), k=3)
        # galaxy seg map rotation
        # seg map is built on unrotated full_image and must be rotated before dispersion
        
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
    
    
    # rotate model back to correct orientation
    roman.model = np.rot90(roman.model)
    roman.grism.data['SCI'] = np.rot90(roman.grism.data['SCI'])
    
    #save grism model image + noise
    
    hdu_list = fits.open(empty_grism)
    if gpad != 0:
        hdu_list.append(fits.ImageHDU(data=roman.model[gpad:-gpad, gpad:-gpad],name='MODEL'))
        #hdu_list.append(fits.ImageHDU(data=roman.grism.data['SCI'][gpad:-gpad, gpad:-gpad],name='ERR'))
        hdu_list['ERR'].data = roman.grism.data['SCI'][gpad:-gpad, gpad:-gpad]
    else:
        hdu_list.append(fits.ImageHDU(data=roman.model,name='MODEL'))
        #hdu_list.append(fits.ImageHDU(data=roman.grism.data['SCI']),name='ERR')
        hdu_list['ERR'].data = roman.grism.data['SCI']
    
    out_fn = os.path.join(output_dir, fn_root_grism+'.fits')
    hdu_list.writeto(out_fn, overwrite=True)
    hdu_list.close()
    print('wrote to '+out_fn)



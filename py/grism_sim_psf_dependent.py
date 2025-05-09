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

def mk_grism(tel_ra,tel_dec,pa,det_num,star_input,gal_input,output_dir,confver='07242020',psf_cutout_size=365,extra_grism_name='',github_dir=github_dir_env,gal_mag_col='mag_F158_Av1.6523',dogal='y',magmax=25,mockdir='/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/'):
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

    empty_direct_fits_out_nopad = os.path.join(output_dir,fn_root+'_nopad.fits')
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

    tot_im_size = 4088+2*gpad #? I removed the pad from this size. I don't use it below. What impact does this have on the ra/dec placements?

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

    # Cuts galaxy catalog and preps convolution info?
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

    # Save an empty direct fits with appropriate header info and WCS
    full_model = np.zeros((tot_im_size, tot_im_size))
    phdu = fits.PrimaryHDU(data=full_model)
    phdu.header["INSTRUME"] = 'ROMAN   '
    phdu.header["FILTER"] = "f140w"
    phdu.header["EXPTIME"] = 141
    shp = full_model.shape
    phdu.header = iu.add_wcs(phdu,ra, dec, crpix2=shp[1]/2,crpix1=shp[0]/2, cdelt1=0.11, cdelt2=0.11,
                crota2=pa,naxis1=shp[0],naxis2=shp[1])

    err = np.random.poisson(10,full_model.shape)*0.001 #np.zeros(full_model.shape)
    ihdu = fits.ImageHDU(data=full_model,name='SCI',header=phdu.header)
    ehdu = fits.ImageHDU(data=err,name='ERR',header=phdu.header)
    dhdu = fits.ImageHDU(data=np.zeros(full_model.shape),name='DQ',header=phdu.header)
    hdul = fits.HDUList([phdu,ihdu,ehdu,dhdu])
    hdul.writeto(empty_direct_fits_out_nopad, overwrite=True)

    # Save empty grism fits
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
    
    # Instantiate Grizli GrizliFLT
    roman = GrismFLT(grism_file=empty_grism,ref_file=empty_direct_fits_out_nopad, seg_file=None, pad=gpad) 
    roman.seg = np.zeros((tot_im_size,tot_im_size), dtype=np.float32) #this segmentation map should have the area of the padded grism image, but not have the padding added because of the PSF size
    
    df = Table.read(os.path.join(github_dir, 'grism_sim/data/wfirst_wfi_f158_001_syn.fits'), format='fits') #close to H-band
    bp = S.ArrayBandpass(df["WAVELENGTH"], df["THROUGHPUT"])
    
    minlam = grizli_conf["minlam"]
    maxlam = grizli_conf["maxlam"]
    
    tempdir = os.path.join(github_dir, 'star_fields/data/SEDtemplates/')
    templates = open(os.path.join(github_dir, 'star_fields/data/SEDtemplates/input_spectral_STARS.lis')).readlines()
    temp_inds = stars['star_template_index'] - 58*(stars['star_template_index']//58)

    # Setup roll-on/roll-off shape
    npsfs = grizli_conf["npsfs"]
    spectrum_overlap = grizli_conf["spectrum_overlap"]
    window_x = np.linspace(0, np.pi, spectrum_overlap)
    front_y = (1 - np.cos(window_x)) / 2
    back_y = 1 - front_y
    fov_pixels = pad-1
    # full_model = np.zeros((4088, 4088)) # This is already created above and remains unchanged by this point. We continue to use it below

    bins = np.linspace(minlam, maxlam, npsfs + 1)

    # START sim here
    for ii, start_wave in enumerate(bins[:-1]):
        print(f"starting at {start_wave}")
        end_wave = bins[ii+1]

        start_wave -= spectrum_overlap * 0.5
        end_wave += spectrum_overlap * 0.5 - 1

        # Stay within our spectrum limits (these could be extended or not hardcoded if needed)
        if start_wave < 10000:
            start_wave = 10000
        
        if end_wave > 20000:
            end_wave = 20000

        # fid_psf = iu.get_psf(fov_pixels=pad-1, det=det) # Fiducial psf generation
        psf_grid = iu.create_psf_grid(wavelength=start_wave*10e-11, fov_pixels=fov_pixels, det=det) # PSF Grid generation

        thresh = 0.01 #threshold flux for segmentation map
        det_with_pad = grizli_conf["detector_size"] + 2*gpad

        # STAR SIM
        print("adding stars to model")
        for i in tqdm(range(0,len(stars))):
            photid = i+1

            # STAR DIRECT
            xpos = stars[i]['Xpos']
            ypos = stars[i]['Ypos']
            mag = stars[i]['magnitude']
            if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
                print(xpos,ypos,'out of bounds position')
            xp = int(xpos)
            yp = int(ypos)
            xoff = 0#xpos-xp
            yoff = 0#ypos-yp

            # sp = iu.star_postage_inpsf(mag,fid_psf) # PSF from fiducial
            sp = iu.star_postage_grid(psf_grid,mag,xp,yp,fov_pixels=fov_pixels) # PSF from grid
            #? Do we need to subtract the pad to xp, yp to get the actual detector position for STPSF?

            # sp limits are needed to keep only what fits on the detector (plus pad)
            sp_lims = [max(0,-(xp+gpad-fov_pixels)), min(fov_pixels*2,fov_pixels*2-(xp+gpad+fov_pixels-det_with_pad)),
            max(0,-(yp+gpad-fov_pixels)), min(fov_pixels*2,fov_pixels*2-(yp+gpad+fov_pixels-det_with_pad))]

            # Set direct image equal to sp; don't add
            roman.direct.data["REF"][xp+gpad-fov_pixels:xp+gpad+fov_pixels,yp+gpad-fov_pixels:yp+gpad+fov_pixels] = sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]]
            roman.direct.data['REF'] *= roman.direct.ref_photflam #? Copied from below. This is used by grizli to setup the ref? Do we need to use it here? It was commneted out before, but seems important to include?      

            # Define selseg from original sp
            selseg = sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]] > thresh
            # set seg; use unique ids to make other spaces irrelevant (no need to reset between stars)
            roman.seg[xp+gpad-fov_pixels:xp+gpad+fov_pixels,yp+gpad-fov_pixels:yp+gpad+fov_pixels][selseg] = photid 

            # STAR GRISM
            row = stars[i]
            mag = row["magnitude"]
            temp_ind = int(temp_inds[i])
            #print(temp_ind)
            star_type = templates[temp_ind].strip('\n')
            temp = np.loadtxt(os.path.join(tempdir, star_type)).transpose()
            wave = temp[0]
            flux = temp[1]

            # renormalization has to occur before picking out the spectrum segment to avoid a DisjointError
            star_spec = S.ArraySpectrum(wave=wave, flux=flux, waveunits="angstroms", fluxunits="flam")
            spec = star_spec.renorm(mag, "abmag", bp)
            spec.convert("flam")

            # pick out segment of spectrum
            sel = spec.wave > start_wave
            sel &= spec.wave < end_wave
            wave = spec.wave[sel]
            flux = spec.flux[sel]

            # apodize/roll-on, roll-off
            if start_wave != 10000:
                flux[:spectrum_overlap] *= front_y
            if end_wave != 20000:
                flux[-spectrum_overlap:] *= back_y            
        
            # roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=True, store=False,
            #                         is_cgs=True, spectrum_1d=[spec.wave, spec.flux])
            
            #? Is this catcher process necessary? Could we utilize in_place? That used to "cast a shadow." Does it still do that?
            segment_of_dispersion = roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=False, store=False,
                                    is_cgs=True, spectrum_1d=[wave, flux])
            
            full_model += segment_of_dispersion[1]

        if ngal > 0:
            print('adding galaxies to model')
            for i in tqdm(range(0,ngal)):
                photid += 1
                row = gals[i]
                mag = row['mag']
                imflux = iu.mag2flux(mag)#imflux = row['flux']
                #make image, put it in seg
                thresh = 0.01 #threshold flux for segmentation map

                gal_psf = iu.gal_postage_grid(psf_grid,xp,yp,fov_pixels=fov_pixels)
                conv_prof = signal.convolve2d(gal_psf,testprof,mode='same') #! Setup PSF convolution

                xpos = row['Xpos']
                ypos = row['Ypos']
                #if xpos > 4088+2*gpad or ypos > 4088+2*gpad:
                #    print(xpos,ypos,'out of bounds position')
                xp = int(xpos)
                yp = int(ypos)
                xoff = 0#xpos-xp
                yoff = 0#ypos-yp
                sp = imflux*conv_prof

                # sp limits are needed to keep only what fits on the detector (plus pad)
                sp_lims = [max(0,-(xp+gpad-fov_pixels)), min(fov_pixels*2,fov_pixels*2-(xp+gpad+fov_pixels-det_with_pad)),
                max(0,-(yp+gpad-fov_pixels)), min(fov_pixels*2,fov_pixels*2-(yp+gpad+fov_pixels-det_with_pad))]

                # Set direct image equal to sp; don't add
                roman.direct.data["REF"][xp+gpad-fov_pixels:xp+gpad+fov_pixels,yp+gpad-fov_pixels:yp+gpad+fov_pixels] = sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]]
                roman.direct.data['REF'] *= roman.direct.ref_photflam #? Do I need to process the ref image when I'm setting it directly in grizli? It was commneted out before, but seems important to include?
                
                selseg = sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]] > thresh
                roman.seg[xp+gpad-fov_pixels:xp+gpad+fov_pixels,yp+gpad-fov_pixels:yp+gpad+fov_pixels][selseg] = photid
                
                #get sed and convert to spectrum
                sim_fn = mockdir+'galacticus_FOV_EVERY100_sub_'+str(row['SIM'])+'.hdf5'
                sim = h5py.File(sim_fn, 'r')
                sed_flux = sim['Outputs']['SED:observed:dust:Av1.6523'][row['IDX']]

                # initial cut to avoid nan values
                wave = np.linspace(2000, 40000, 19001) #wavelength grid for simulation
                sel_wave = wave > 10000
                sel_wave &= wave <20000
                wave = wave[sel_wave]
                flux = sed_flux[sel_wave]
                
                gal_spec = S.ArraySpectrum(wave=wave, flux=flux, waveunits="angstroms", fluxunits="flam")
                spec = gal_spec.renorm(mag, "abmag", bp) # renorm and convert units
                spec.convert("flam") 

                # pick out segment of spectrum
                sel_wave = wave > start_wave
                sel_wave &= wave < end_wave
                wave = spec.wave[sel_wave]
                flux = spec.flux[sel_wave]

                # apodize/roll-on, roll-off
                if start_wave != 10000:
                    flux[:spectrum_overlap] *= front_y
                if end_wave != 20000:
                    flux[-spectrum_overlap:] *= back_y    

                segment_of_dispersion = roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=False, store=False,
                                        is_cgs=True, spectrum_1d=[wave, flux])[1]
                
                full_model += segment_of_dispersion
                
    # There is no direct image or segmentation file to save. The only savable/non-intermediate product is the grism model. 
    # We'd need to run a seperate direct image loop using the H Band filter and an effective PSF to get a realistic direct image.

    #save grism model image + noise
    roman.model = full_model
    
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
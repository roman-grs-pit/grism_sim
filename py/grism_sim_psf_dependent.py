from tqdm import tqdm
import time
import h5py
import numpy as np
from scipy import signal
from astropy.io import fits
from astropy.table import Table#, join
# from astropy.wcs import WCS
# from astropy import units as u
# from astropy.coordinates import SkyCoord
import os
# import matplotlib.pyplot as plt
# Spectra tools
import pysynphot as S

from grizli.model import GrismFLT
import grizli.fake_image

import pysiaf # use for WCS instead of iu functions
import image_utils as iu
import psf_grid_utils as pgu

import yaml

github_dir_env=os.getenv('github_dir')
if github_dir_env is None:
    print('github_dir environment variable has not been set, will cause problems if not explicitly set in function calss')

psf_grid_data_write=os.getenv("psf_grid_data_write")
if psf_grid_data_write is None:
    print("psf_grid_data_write variable has not been set. This will cause problems if psf_grid fits do not already exist.")

def try_wait_loop(func, *args, max_attempts=3, wait=5, **kwargs):
    """
    Attempt to call func using args and kwargs. Upon a fail, wait som time and try
    again. Upon {max_attempts} number of fails, raise Exception. Used when reading
    files recently written on NERSC.

    Parameters
    ----------
    func: callable
        Function or other callable to attempt calling
    max_attempts: int, optional
        Maximum number of attempts before raising exception. default: 3
    wait: float, optional
        Seconds to wait upon failure before retrying. default: 5
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            res = func(*args, **kwargs)
            break
        except Exception as e:
            attempt += 1 
            if attempt < max_attempts:
                print(f"{func.__name__} failed. Waiting {wait} and retrying: {attempt}/{max_attempts}")
                time.sleep(wait)
            else:
                print(f"{func.__name__} failed. Maximum retries exceeded: {max_attempts}")
                raise e
    
    return res

def mk_grism(tel_ra,tel_dec,tel_pa,det_num,star_input,gal_input,output_dir,
             extra_grism_name='',extra_ref_name='',github_dir=github_dir_env,
             gal_mag_col='mag_F158_Av1.6523',dogal='y',magmax=25,
             mockdir='/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/',
             check_psf=False,conv_gal=True,use_tqdm=False,seed=3,**kwargs):
    """
    Simulated Roman WFI Grism Image of objects in catalogs. Includes background and
    shot noise. Final fits file contains: PrimaryHDU, SCI, ERR, DQ, Noiseless Model

    Process Outline:
    Load default config -> Prepare WCS -> Save empty fits files for Grizli --
    --> Cut catalogs to on-detector objects -> Instantiate grizli.GrismFLT --
    --> Loop over wavelength bins: ( Load PSF Grid -> Loop over objects: (
    Compute monochromatic direct image -> Write direct & segmentation image
    to GrismFLT -> Cut out relevant part of spectrum -> Compute piece of sim ))
    --> Save full simulation result and first computed monochromatic direct
    image as fits files.

    Parameters
    ----------
    tel_ra: float
        WFI Center pointing, RA in degrees
    tel_dec: float
        WFI Center pointing, Dec in degrees
    tel_pa: float
        WFI Center pointing, PA in degrees
    det_num: int
        Detector Number
    star_input: astropy.table.Table
        Table with columns 'RA', 'DEC', 'magnitude', 'star_template_index'
    gal_input: astropy.table.Table
        Table with columns 
    output_dir: str
        Path to directory for output
    extra_grism_name: str, optional
        Appended to grism fits file
    extra_ref_name: str, optional
        Appended to empty reference fits file
    github_dir: str, optional
        Path to directory containing all GRS PIT Github repos. 
        default: reads environment variable
    gal_mag_col: str, optional
        Name of the column in the gal_input catalog wih magnitude information
        default: "mag_F158_Av1.6523"
    dogal: str, optional
        Simulates galaxies if set to 'y'
        default: 'y'
    magmax: float, optional
        Brightest galaxy magnitude to simulate; Fainter galaxies are skipped
        default: 25
    mockdir: float, optional
        Directory containing galaxy SEDs. default: NERSC path
    check_psf: bool, optional
        Determines whether to check psf_grid fits files match given kwargs. 
        default: False
    conv_gal: bool, optional
        Determines whether to calculate and convolve PSF with galaxies.
        default: True
    use_tqdm: bool, optional
        Show tqdm progress bar during simulation. Reccomend set to False when 
        running sbatch jobs on NERSC. default: False
    seed: int, optional
        Numpy rng seed for noise. default: 3
    """
    #tel_ra,tel_dec correspond to the coordinates (in degrees) of the middle of the field (not the detector center)
    #tel_pa is the position angle (in degrees), relative to lines of ra=constant; note, requires +60 on tel_pa for wfi_sky_pointing
    #det_num is an integer corresponding to the detector number
    #star_input is a table or array with columns 'RA', 'DEC', 'magnitude', 'star_template_index'
    #gal_input is a table or array with columns...
    #output_dir is the directory for output
    #psf_cutout_size is the size in pixels used to determine the psf and then all of the image stamps that get added together
    
    if gal_input is None:
        dogal = 'n'

    timings = {}
    timings["checkpoint_0"] = time.time()
    print("checkpoint_0")
    # * Read config
    conf_file = os.path.join(github_dir, "grism_sim/data/grizli_config.yaml")
    with open(conf_file) as f:
        grizli_conf = yaml.safe_load(f)
    
    kwkeys = list(kwargs.keys())
    for arg in kwkeys:
        if arg in grizli_conf:
            grizli_conf[arg] = kwargs.pop(arg)
    
    det = "SCA{:02}".format(det_num)
    half_fov_pixels = int(grizli_conf["fov_pixels"] / 2) # size of star thumbnails
    thresh = grizli_conf["thresh"] # threshhold pixel value to be dispersed
    size = grizli_conf["size"][det] + 364
    gpad = grizli_conf["pad"] # padding added in order to catch off-detector objects that disperse on-detector
    tot_im_size = grizli_conf["detector_size"] + 2*gpad 
    confdir = grizli_conf["confdir"]
    conf = grizli_conf["conf"][det]
    if confdir is not None:
        conf = os.path.join(confdir, conf)
    
    #this ends up setting the background noise and defines the WCS
    background = grizli_conf["grism_background"]
    EXPTIME = grizli_conf["GEXPTIME"] 
    NEXP = 1     

    timings["checkpoint_1"] = time.time()
    print("checkpoint_1")
    # * Setup WCS
    siaf = pysiaf.Siaf("roman")
    wfi_siaf = siaf["WFI{:02}_FULL".format(det_num)]
    
    # Use WFI_CEN for aiming
    v2ref = siaf["WFI_CEN"].V2Ref
    v3ref = siaf["WFI_CEN"].V3Ref

    attmat = pysiaf.utils.rotations.attitude_matrix(v2ref, v3ref, tel_ra, tel_dec, tel_pa) # pysiaf tel_pa is 60 more than image_utils tel_pa (i.e. siaf_pa = iu_pa + 60)

    wfi_siaf.set_attitude_matrix(attmat)
    ra, dec = wfi_siaf.det_to_sky(2043, 2043) # I believe pysiaf uses 0-index for origin pixel; thus, center pix is 2043 not 2044

    timings["checkpoint_2"] = time.time()
    print("checkpoint_2")
    # * Save helper empty fits files
    # Save an empty direct fits with appropriate header info and WCS
    full_model_noiseless = np.zeros((tot_im_size, tot_im_size))
    full_ref = np.zeros((tot_im_size, tot_im_size))

    fn_root = 'refimage_ra%s_dec%s_pa%s_det%s' % (tel_ra,tel_dec,tel_pa,det)
    fn_root += extra_ref_name
    empty_direct_fits_out_nopad = os.path.join(output_dir,fn_root+'_nopad.fits')

    phdu = fits.PrimaryHDU(data=full_model_noiseless)
    phdu.header["INSTRUME"] = 'ROMAN   '
    phdu.header["FILTER"] = "f140w"
    phdu.header["EXPTIME"] = grizli_conf["DIREXPTIME"] # direct exptime
    shp = full_model_noiseless.shape
    phdu.header = iu.add_wcs(phdu,ra, dec, crpix2=shp[1]/2,crpix1=shp[0]/2,
                             crota2=tel_pa,naxis1=shp[0],naxis2=shp[1])

    err = np.random.poisson(10,full_model_noiseless.shape)*0.001 #np.zeros(full_model_noiseless.shape)
    ihdu = fits.ImageHDU(data=full_model_noiseless,name='SCI',header=phdu.header)
    ehdu = fits.ImageHDU(data=err,name='ERR',header=phdu.header)
    dhdu = fits.ImageHDU(data=np.zeros(full_model_noiseless.shape),name='DQ',header=phdu.header)
    hdul = fits.HDUList([phdu,ihdu,ehdu,dhdu])
    hdul.writeto(empty_direct_fits_out_nopad, overwrite=True)

    # Save empty grism fits
    fn_root_grism = 'grism_ra%s_dec%s_pa%s_det%s' % (tel_ra,tel_dec,tel_pa,det)
    fn_root_grism += extra_grism_name 
    empty_grism = os.path.join(output_dir, 'empty_'+fn_root_grism+'.fits')
    h, _ = grizli.fake_image.roman_header(ra=ra, dec=dec, pa_aper=tel_pa, naxis=(4088,4088))
    grizli.fake_image.make_fake_image(h, output=empty_grism, exptime=EXPTIME, nexp=NEXP, background=background)
    file = try_wait_loop(fits.open, empty_grism)
    file[1].header["CONFFILE"] = os.path.join(github_dir, "grism_sim/data", conf)
    file.writeto(empty_grism, overwrite=True)
    file.close()

    timings["checkpoint_3"] = time.time()
    print("checkpoint_3")
    # * Use WCS to Prepare object Catalogs
    star_xy_siaf = wfi_siaf.sky_to_sci(star_input["RA"], star_input["DEC"])
    star_xy = (star_xy_siaf[0] + gpad, star_xy_siaf[1] + gpad)
    
    sel_ondet = star_xy[0] > 0#stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
    sel_ondet &= star_xy[0] < tot_im_size
    sel_ondet &= star_xy[1] > 0#stars00['Xpos'] < 4088 + 2*( gpad) #we want everything within padded area around grism
    sel_ondet &= star_xy[1] < tot_im_size
    
    print('cutting stars to be on detector + padded area')
    stars = star_input[sel_ondet]
    #print(stars00['Xpos'].shape)
    stars['Xpos'] = star_xy[0][sel_ondet]
    stars['Ypos'] = star_xy[1][sel_ondet]
    nstar = len(stars)
    ngal = 0

    # Cuts galaxy catalog and preps convolution info?
    if dogal == 'y':
        gal_xy_siaf = wfi_siaf.sky_to_sci(gal_input["RA"], gal_input["DEC"])
        gal_xy = (gal_xy_siaf[0] + gpad, gal_xy_siaf[1] + gpad)
    
        sel_ondet = gal_xy[0] > 0
        sel_ondet &= gal_xy[0] < tot_im_size
        sel_ondet &= gal_xy[1] > 0
        sel_ondet &= gal_xy[1] < tot_im_size
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

        if not conv_gal:
            testprof = np.pad(testprof, 698, mode="constant", constant_values=0)

    timings["checkpoint_4"] = time.time()
    print("checkpoint_4")
    # * Read bandpass file, and setup apodization
    df = Table.read(os.path.join(github_dir, 'grism_sim/data/wfirst_wfi_f158_001_syn.fits'), format='fits') #close to H-band
    bp = S.ArrayBandpass(df["WAVELENGTH"], df["THROUGHPUT"])
    
    minlam = grizli_conf["minlam"]
    maxlam = grizli_conf["maxlam"]
    spectrum_overlap = grizli_conf["spectrum_overlap"]
    npsfs = grizli_conf["npsfs"]
    
    tempdir = os.path.join(github_dir, 'star_fields/data/SEDtemplates/')
    temp_inds = stars['star_template_index'] - 58*(stars['star_template_index']//58)
    with open(os.path.join(github_dir, 'star_fields/data/SEDtemplates/input_spectral_STARS.lis')) as f:
        templates = f.readlines()

    # Setup roll-on/roll-off shape
    window_x = np.linspace(0, np.pi, spectrum_overlap)
    front_y = (1 - np.cos(window_x)) / 2
    back_y = 1 - front_y

    bins = np.linspace(minlam, maxlam, npsfs + 1)

    timings["checkpoint_5"] = time.time()
    print("checkpoint_5")
    # * Instantiate Grizli GrismFLT
    roman = try_wait_loop(GrismFLT, grism_file=empty_grism,ref_file=empty_direct_fits_out_nopad, seg_file=None, pad=gpad)
    roman.seg = np.zeros((tot_im_size,tot_im_size), dtype=np.float32) #this segmentation map should have the area of the padded grism image, but not have the padding added because of the PSF size

    print("checkpoint_6")
    timings["checkpoint_6"] = time.time()
    timings["PSF_grid_load"] = 0
    timings["star_PSF_eval"] = 0
    timings["star_placement"] = 0
    timings["star_spec_prep"] = 0
    timings["star_grism_sim"] = 0
    if dogal:
        if conv_gal:
            timings["gal_PSF_eval"] = 0
            timings["gal_PSF_conv"] = 0
        else:
            timings["gal_flux_step"] = 0
        timings["gal_placement"] = 0
        timings["gal_spec_prep"] = 0
        timings["gal_grism_sim"] = 0

    # * START sim here
    for ii, start_wave in enumerate(bins[:-1]):
        end_wave = bins[ii+1]
        print(f"starting at {start_wave}")

        start = time.time()
        # * read in/check psf_grid
        psf_filename = f"wfi_grism0_fovp{half_fov_pixels * 2}_wave{start_wave:.0f}_{det}.fits".lower() # {instrument}_{filter}_{fovp}_wave{wavelength}_{det}.fits
        try:
            psf_grid = pgu.load_psf_grid(psf_filename)
        except OSError as e:
            print("creating new PSF Grid")
            pgu.save_one_grid(det_num, start_wave, psf_grid_data_write, half_fov_pixels=half_fov_pixels, **kwargs)
            psf_grid = try_wait_loop(pgu.load_psf_grid, psf_filename)
        
        if check_psf:
                if kwargs is not None:
                    pgu.check_version(psf_filename, **kwargs)
                else:
                    pgu.check_version(psf_filename)
                
        end = time.time()
        timings["PSF_grid_load"] += (end - start)

        # * STAR SIM
        print("adding stars to model")
        if use_tqdm:
            star_iter = tqdm(range(0,len(stars)))
        else:
            star_iter = range(0,len(stars))
        for jj in star_iter:
            photid = jj+1

            # STAR DIRECT
            # direct read of characteristics
            xpos = stars[jj]['Xpos']
            ypos = stars[jj]['Ypos']
            mag = stars[jj]['magnitude']

            # cleaned up characteristisc
            xp = int(xpos) 
            yp = int(ypos)
            xtrue = xpos - gpad
            ytrue = ypos - gpad

            start = time.time()
            sp = iu.star_postage_grid(psf_grid,mag,xtrue,ytrue,half_fov_pixels=half_fov_pixels) # PSF from grid

            end = time.time()
            timings["star_PSF_eval"] += (end - start)

            start = time.time()
            # sp limits are needed to keep only what fits on the detector (plus pad)
            sp_lims = [max(0,-(yp-half_fov_pixels)), min(half_fov_pixels*2,half_fov_pixels*2-(yp+half_fov_pixels-tot_im_size)),
                        max(0,-(xp-half_fov_pixels)), min(half_fov_pixels*2,half_fov_pixels*2-(xp+half_fov_pixels-tot_im_size))]

            roman_lims = [max(0, yp-half_fov_pixels), min(tot_im_size, yp+half_fov_pixels), 
                            max(0, xp-half_fov_pixels), min(tot_im_size, xp+half_fov_pixels)]

            # Set direct image equal to sp; don't add
            roman.direct.data["REF"][roman_lims[0]:roman_lims[1], roman_lims[2]:roman_lims[3]] = sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]]
            roman.direct.data['REF'] *= roman.direct.ref_photflam 

            if start_wave==minlam:
                full_ref[roman_lims[0]:roman_lims[1], roman_lims[2]:roman_lims[3]] += sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]]

            # Define selseg from original sp
            selseg = sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]] > thresh
            # set seg; use unique ids to make other spaces irrelevant (no need to reset between stars)
            roman.seg[roman_lims[0]:roman_lims[1], roman_lims[2]:roman_lims[3]][selseg] = photid 

            # Rotations after placement on detector
            roman.direct.data["REF"] = np.rot90(roman.direct.data["REF"], k=3)
            roman.seg = np.rot90(roman.seg, k=3)

            end = time.time()
            timings["star_placement"] += (end - start)

            # STAR GRISM
            row = stars[jj]
            mag = row["magnitude"]
            temp_ind = int(temp_inds[jj])
            star_type = templates[temp_ind].strip('\n')
            temp = np.loadtxt(os.path.join(tempdir, star_type)).transpose()

            start = time.time()
            wave = np.arange(minlam, maxlam, 5)
            flux = np.interp(wave, temp[0], temp[1]) #interp avoids indexing errors by normalizing sed shape/length
            # renormalization has to occur before picking out the spectrum segment to avoid a DisjointError
            star_spec = S.ArraySpectrum(wave=wave, flux=flux, waveunits="angstroms", fluxunits="flam")
            spec = star_spec.renorm(mag, "abmag", bp)
            spec.convert("flam")

            # if-elses enforce minlam/maxlam bounds on the spectrum (and avoid issues with negative indicies)
            if start_wave != minlam:
                # Adjust start_wave to include overlap region
                start_wave_index = np.searchsorted(spec.wave, start_wave, side="left")
                start_index_w_overlap = start_wave_index - int(spectrum_overlap * 0.5)
                sel = wave >= spec.wave[start_index_w_overlap]
            else:
                # Set lower limit on sel_wave
                sel = wave >= start_wave

            if end_wave != maxlam:
                # Adjust end_wave to include overlap region
                end_wave_index = np.searchsorted(spec.wave, end_wave, side="right")
                end_index_w_overlap = end_wave_index + int(spectrum_overlap * 0.5 - 1) 
                sel &= wave < spec.wave[end_index_w_overlap]
            else:
                # Set upper limit on sel_wave
                sel &= wave <= end_wave
            
            # pick out segment of spectrum
            wave = spec.wave[sel]
            flux = spec.flux[sel]

            # apodize/roll-on, roll-off
            if start_wave != minlam:
                flux[:spectrum_overlap] *= front_y
            if end_wave != maxlam:
                flux[-spectrum_overlap:] *= back_y    

            end = time.time()
            timings["star_spec_prep"] += (end - start)
            
            start = time.time()
            segment_of_dispersion = roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=False, store=False,
                                    is_cgs=True, spectrum_1d=[wave, flux])
            
            # compute_model_orders returns a boolean IF the dispersion would not land on the detector
            try:
                full_model_noiseless += segment_of_dispersion[1]
            except TypeError: # catch "cannot index bool" error (star would not disperse onto detector)
                continue

            end = time.time()
            timings["star_grism_sim"] += (end - start)

        # * GAL SIM
        if ngal > 0:
            print('adding galaxies to model')
            if use_tqdm:
                gal_iter = tqdm(range(0,ngal))
            else:
                gal_iter = range(0,ngal)
            for jj in gal_iter:
                # This if statements allows galaxies without stars; else, photid is not set and an OSError is raised
                if "photid" not in locals():
                    photid = 0
                photid += 1
                row = gals[jj]

                # GAL DIRECT
                # direct read of characteristics
                xpos = row['Xpos']
                ypos = row['Ypos']
                mag = row['mag']

                # cleaned up characteristisc
                xp = int(xpos) 
                yp = int(ypos)
                xtrue = xpos - gpad
                ytrue = ypos - gpad

                start = time.time()
                # convolve direct thumbnail with psf
                imflux = iu.mag2flux(mag)#imflux = row['flux']
                if conv_gal:
                    gal_psf = iu.gal_postage_grid(psf_grid,xtrue,ytrue,half_fov_pixels=half_fov_pixels)
                    
                    end = time.time()
                    timings["gal_PSF_eval"] += (end - start)

                    start = time.time()
                    conv_prof = signal.fftconvolve(gal_psf,testprof,mode='same') 
                    sp = imflux*conv_prof

                    end = time.time()
                    timings["gal_PSF_conv"] += (end - start)
                else:
                    sp = imflux * testprof
                    end = time.time()
                    timings["gal_flux_step"] += (end - start)

                start = time.time()
                # sp limits are needed to keep only what fits on the detector (plus pad)
                sp_lims = [max(0,-(yp-half_fov_pixels)), min(half_fov_pixels*2,half_fov_pixels*2-(yp+half_fov_pixels-tot_im_size)),
                            max(0,-(xp-half_fov_pixels)), min(half_fov_pixels*2,half_fov_pixels*2-(xp+half_fov_pixels-tot_im_size))]

                roman_lims = [max(0, yp-half_fov_pixels), min(tot_im_size, yp+half_fov_pixels), 
                                max(0, xp-half_fov_pixels), min(tot_im_size, xp+half_fov_pixels)]

                # Set direct image equal to sp; don't add
                roman.direct.data["REF"][roman_lims[0]:roman_lims[1], roman_lims[2]:roman_lims[3]] = sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]]
                roman.direct.data['REF'] *= roman.direct.ref_photflam

                # GAL GRISM
                if start_wave==minlam:
                    full_ref[roman_lims[0]:roman_lims[1], roman_lims[2]:roman_lims[3]] += sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]]
                
                selseg = sp[sp_lims[0]:sp_lims[1],sp_lims[2]:sp_lims[3]] > thresh
                roman.seg[roman_lims[0]:roman_lims[1], roman_lims[2]:roman_lims[3]][selseg] = photid

                # Rotations after placement on detector
                roman.direct.data["REF"] = np.rot90(roman.direct.data["REF"], k=3)
                roman.seg = np.rot90(roman.seg, k=3)

                end = time.time()
                timings["gal_placement"] += (end - start)
                
                start = time.time()
                #get sed and convert to spectrum
                sim_fn = os.path.join(mockdir, 'galacticus_FOV_EVERY100_sub_'+str(row['SIM'])+'.hdf5')
                sim = h5py.File(sim_fn, 'r')
                sed_flux = sim['Outputs']['SED:observed:dust:Av1.6523'][row['IDX']]

                # initial cut to avoid errors from nan values
                wave = np.linspace(2000, 40000, 19001) #wavelength grid for simulation
                sel_wave = wave > minlam
                sel_wave &= wave < maxlam
                wave = wave[sel_wave]
                flux = sed_flux[sel_wave]
                
                gal_spec = S.ArraySpectrum(wave=wave, flux=flux, waveunits="angstroms", fluxunits="flam")
                spec = gal_spec.renorm(mag, "abmag", bp) # renorm and convert units
                spec.convert("flam") 

                # if-elses enforce minlam/maxlam bounds on the spectrum (and avoid issues with negative indicies)
                if start_wave != minlam:
                    # Adjust start_wave to include overlap region
                    start_wave_index = np.searchsorted(spec.wave, start_wave, side="left")
                    start_index_w_overlap = start_wave_index - int(spectrum_overlap * 0.5)
                    sel_wave = spec.wave >= spec.wave[start_index_w_overlap]
                else:
                    # Set lower limit on sel_wave
                    sel_wave = spec.wave >= start_wave

                if end_wave != maxlam:
                    # Adjust end_wave to include overlap region
                    end_wave_index = np.searchsorted(spec.wave, end_wave, side="right")
                    end_index_w_overlap = end_wave_index + int(spectrum_overlap * 0.5 - 1) 
                    sel_wave &= spec.wave < spec.wave[end_index_w_overlap]
                else:
                    # Set upper limit on sel_wave
                    sel_wave &= spec.wave <= end_wave

                # pick out segment of spectrum
                wave = spec.wave[sel_wave]
                flux = spec.flux[sel_wave]

                # apodize/roll-on, roll-off
                if start_wave != minlam:
                    flux[:spectrum_overlap] *= front_y
                if end_wave != maxlam:
                    flux[-spectrum_overlap:] *= back_y    

                end = time.time()
                timings["gal_spec_prep"] += (end - start)

                start = time.time()
                segment_of_dispersion = roman.compute_model_orders(id=photid, mag=mag, compute_size=False, size=size, in_place=False, store=False,
                                        is_cgs=True, spectrum_1d=[wave, flux])
                
                # compute_model_orders returns a boolean IF the dispersion would not land on the detector
                try:
                    full_model_noiseless += segment_of_dispersion[1] # catch "cannot index bool" error (gal would not disperse onto detector)
                except TypeError:
                    continue

                end = time.time()
                timings["gal_grism_sim"] += (end - start)

    timings["checkpoint_7"] = time.time()
    print("checkpoint_7")
    # * save grism model image + noise
    true_noiseless = np.copy(full_model_noiseless)
    # Noise
    rng = np.random.default_rng(seed=seed)
    sel = full_model_noiseless < 0
    full_model_noiseless[sel] = 0
    full_model_poisson = rng.poisson(full_model_noiseless * EXPTIME) / EXPTIME
    
    bg_noise = background + roman.grism.data["SCI"]
    full_model_final = full_model_poisson + bg_noise

    # Final model rotation
    full_model_final = np.rot90(full_model_final, k=1)
    full_model_noiseless = np.rot90(full_model_noiseless, k=1)
    true_noiseless = np.rot90(true_noiseless, k=1)

    # Save model
    hdu_list = fits.open(empty_grism)
    if gpad != 0:
        hdu_list.append(fits.ImageHDU(data=true_noiseless[gpad:-gpad, gpad:-gpad], name='MODEL'))
        #hdu_list.append(fits.ImageHDU(data=roman.grism.data['SCI'][gpad:-gpad, gpad:-gpad],name='ERR'))
        hdu_list['ERR'].data = np.sqrt((full_model_noiseless[gpad:-gpad, gpad:-gpad] + background) * EXPTIME) / EXPTIME
        hdu_list["SCI"].data = full_model_final[gpad:-gpad, gpad:-gpad]
    else:
        hdu_list.append(fits.ImageHDU(data=true_noiseless, name='MODEL'))
        #hdu_list.append(fits.ImageHDU(data=roman.grism.data['SCI']),name='ERR')
        hdu_list['ERR'].data = np.sqrt((full_model_noiseless + background) * EXPTIME) / EXPTIME
        hdu_list["SCI"].data = full_model_final
    
    out_fn = os.path.join(output_dir, fn_root_grism+'.fits')
    hdu_list.writeto(out_fn, overwrite=True)
    hdu_list.close()
    print('wrote to '+out_fn)

    # * save monochromatic direct image
    hdu_list = fits.open(empty_direct_fits_out_nopad)
    if gpad != 0:
        hdu_list.append(fits.ImageHDU(data=full_ref[gpad:-gpad, gpad:-gpad],name='IMAGE'))
    else:
        hdu_list.append(fits.ImageHDU(data=full_ref,name='IMAGE'))
    out_fn = os.path.join(output_dir, fn_root+'.fits')
    hdu_list.writeto(out_fn, overwrite=True)
    hdu_list.close()
    print('wrote to '+out_fn)

    timings["checkpoint_8"] = time.time()
    print("checkpoint_8")

    # * print timings
    for key in timings.keys():
        if "checkpoint" not in key:
            print(key, timings[key])
    for ii in range(0, 8):
        print(f"Split {ii}-{ii+1}: ", (timings[f"checkpoint_{ii+1}"] - timings[f"checkpoint_{ii}"]))

    timing_file = os.path.join(output_dir, f"timings_for_{fn_root_grism}.txt")
    with open(timing_file, "w") as f:
        f.write(f"NStars: {nstar} \n")
        f.write(f"NGals: {ngal} \n")
        f.write(f"NPSFs: {npsfs} \n")
        f.write(f"\nCheckpoints\n")
        f.write(f"-----------\n")
        for ii in range(0, 8):
            s = f"Split {ii}-{ii+1}: {timings[f'checkpoint_{ii+1}'] - timings[f'checkpoint_{ii}']} \n"
            f.write(s)  
        
        f.write(f"\nFor-loops (6-7 split)\n")
        f.write(f"---------------------\n")
        f.write("")
        for key in timings.keys():
            if "checkpoint" not in key:
                s = f"{key} - {timings[key]} \n"
                f.write(s)

    return full_model_noiseless
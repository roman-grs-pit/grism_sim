import numpy as np
from astropy.table import Table, join, vstack
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import os,sys
import h5py
import glob
import tqdm

def calc_line_flux(fname, line, z_vals, cosmo):
    '''
    Helper function to calculate true line flux of an emission line as sum of AGN+Disk+Spheroid
    '''
    fn = h5py.File(fname)
    lum_AGN = fn['Lightcone']['Output1']['nodeData']['luminosityEmissionLineAGN:'+line][:]
    lum_disk = fn['Lightcone']['Output1']['nodeData']['luminosityEmissionLineDisk:'+line][:]
    lum_spheroid = fn['Lightcone']['Output1']['nodeData']['luminosityEmissionLineSpheroid:'+line][:]
    line_lum = np.asarray(lum_AGN) + np.asarray(lum_disk) + np.asarray(lum_spheroid)
    lum_distance = np.asarray(cosmo.luminosity_distance(z_vals).value)
    line_flux = line_lum / (4*np.pi*lum_distance**2)
    return line_flux

def get_galacticus_catinfo(fname):
    fn = h5py.File(fname)
    try:
        ras,decs = fn['Lightcone']['Output1']['nodeData']['rightAscension'][:],fn['Lightcone']['Output1']['nodeData']['declination'][:]
        app_mags_F158 = fn['Lightcone']['Output1']['nodeData']['apparentMagnitudeRomanWFI:F158'][:]
        app_mags_F184 = fn['Lightcone']['Output1']['nodeData']['apparentMagnitudeRomanWFI:F184'][:]        
        redshifts = fn['Lightcone']['Output1']['nodeData']['redshift'][:]
        disk_radii = fn['Lightcone']['Output1']['nodeData']['diskRadius'][:]
        spheroid_radii = fn['Lightcone']['Output1']['nodeData']['spheroidRadius'][:]
        
        galt = Table()
        galt['RA'] = ras
        galt['DEC'] = decs
        galt['Z'] = redshifts
        
        skip_start = len('/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock/romanUNIT-d')
        skip_end = 12
        sims = str(fname[skip_start:-skip_end]) # Picks out hdf5 file number from filename
        galt['SIM'] = sims
        idxs = np.arange(0,len(ras)) 
        galt['IDX'] = idxs
        galt['unique_ID'] = [str(idxs[ii])+'_'+sims for ii in range(len(ras))]
        
        galt['mag_F158_app'] = app_mags_F158
        galt['mag_F184_app'] = app_mags_F184
        cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
        dist = np.asarray(cosmo.comoving_distance(redshifts))    # Need to compute absolute magnitude
        abs_mag_F158 = np.asarray(app_mags_F158) - 5*np.log10(dist) + 5
        abs_mag_F184 = np.asarray(app_mags_F184) - 5*np.log10(dist) + 5
        galt['mag_F158_abs'] = abs_mag_F158
        galt['mag_F184_abs'] = abs_mag_F184
        
        galt['diskRadius'] = disk_radii
        galt['spheroidRadius'] = spheroid_radii

        # Emission lines luminosities and converting to fluxes
        flux_halpha = calc_line_flux(fname, 'balmerAlpha6565', redshifts, cosmo)
        flux_oiii4933 = calc_line_flux(fname, 'oxygenIII4933', redshifts, cosmo)
        flux_oiii4960 = calc_line_flux(fname, 'oxygenIII4960', redshifts, cosmo)
        flux_oiii5008 = calc_line_flux(fname, 'oxygenIII5008', redshifts, cosmo)
        galt['flux_halpha'] = flux_halpha
        galt['flux_oiii4933'] = flux_oiii4933
        galt['flux_oiii4960'] = flux_oiii4960
        galt['flux_oiii5008'] = flux_oiii5008
        
        return galt
    except:
        return None

# Compile all of the galacticus info from mock folder into single catalog table
mock_path = '/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock'
gall = []
gal_fns = glob.glob('/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/paper1Calibration_allUNIT_0.2degMock/romanUNIT*') # Grab all filenames from mock_path starting with 'romanUNIT'
for fname in gal_fns:
    gali = get_galacticus_catinfo(fname)
    if gali is not None:
        gall.append(gali)
    else:
        print(fname + ' failed')
galaxy_catalog = vstack(gall)

output_path = '/global/cfs/cdirs/m4943/grismsim/galacticus_02deg2_mock/'
output_filename = output_path+'Roman_02deg2_catalog.fits'
galaxy_catalog.write(output_filename, format='fits', overwrite=True)
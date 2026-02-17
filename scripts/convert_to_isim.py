#convert catalogs to format needed for romanism
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table, join, vstack
import os,sys
from astropy.io import fits


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mkstars", help="whether to make the star file",  action='store_true')
parser.add_argument("--mkgal", help="",  action='store_true')
args = parser.parse_args()

def mag2flux(mag):
    return 10**(mag/-2.5)

if args.mkgal:
    gals = Table.read('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits')
    t = Table()
    t['ra'] = gals['RA']
    t['dec'] = gals['DEC']
    t['n'] = np.ones(len(gals))
    t['type'] = 'SER'
    magl = gals['mag_F158_Av1.6523']
    fluxl = mag2flux(np.array(magl))
    t['F158'] = fluxl
    t['kron_f158_abmag'] = magl
    mel = [0.1,0.1]
    t['kron_f158_abmag_err'] = 0.1#
    
    t['half_light_radius'] = 2.5*.11
    axisratio  = np.ones(len(gals))
    t['ba'] = axisratio
    ol = np.zeros(len(gals))
    t['pa'] = ol
    tgal = t
    del t
    #t.write('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec_4isim.ecsv',overwrite=True)

if args.mkstars:
    stars = Table.read('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus.ecsv')
    t = Table()
    t['ra'] = stars['RA']
    #ral = [10.,10.01]
    #t['ra'] = ral
    #decl = [1.,1.02]
    t['dec'] = stars['DEC']#decl
    nl = [4,4]
    t['n'] = -1*np.ones(len(stars))#nl
    #tl = ['SER','SER']
    t['type'] = 'PSF'#tl
    magl = stars['magnitude']#[12.4,12.2]
    fluxl = mag2flux(np.array(magl))
    t['F158'] = fluxl
    t['kron_f158_abmag'] = magl
    mel = [0.1,0.1]
    t['kron_f158_abmag_err'] = 0.1#mel
    rl = np.zeros(len(stars))#[4.35,0.5]
    t['half_light_radius'] = rl
    axisratio  = np.ones(len(stars))#[1,1]
    t['ba'] = axisratio
    ol = np.zeros(len(stars))#[0,0]
    t['pa'] = ol
    #t.write('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus_4isim.ecsv',overwrite=True)

if args.mkstars and args.mkgal:
    t = vstack([tgal,t])
    t.write('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec_wstars_4isim.ecsv',overwrite=True)
elif args.mkgal:
    tgal.write('/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec_4isim.ecsv',overwrite=True)
elif args.mkstars:
    t.write('/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus_4isim.ecsv',overwrite=True)

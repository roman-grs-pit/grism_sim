#!/usr/bin/env python

import argparse
import yaml

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
import galsim
import romanisim
from romanisim import log, wcs, persistence, parameters, l3, bandpass, util
from romanisim import ris_make_utils as ris
from copy import deepcopy
import math
import asdf
import os.path

github_dir ='/global/common/software/m4943/grizli0/'
star_file = '/global/cfs/cdirs/m4943/grismsim/stars/sim_star_cat_galacticus_4isim.ecsv'
npix = 5000
pixscalefrac = 0.5
pixscale = pixscalefrac * parameters.pixel_scale
midpoint = (npix - 1) / 2
exptime = 100
nexp = 1

def mkL3(args):
    r, d = args[0],args[1]
    center = util.celestialcoord(SkyCoord(ra=r * u.deg, dec=d * u.deg))
    twcs = wcs.create_tangent_plane_gwcs(
        (midpoint, midpoint), pixscale, center)

    metadata = deepcopy(parameters.default_mosaic_parameters_dictionary)
    metadata['filename'] = os.path.basename(args[2])

    cat = ris.create_catalog(metadata=metadata, catalog_name=star_file,
                            bandpasses=['F158'], coord=center, radius=npix)

    im, extras = l3.simulate(
        (npix, npix), twcs, exptime, 'F158',
        cat, effreadnoise=None, nexposures=nexp,
        metadata=metadata)

    # Create metadata for simulation parameter
    #romanisimdict = deepcopy(vars(args))
    #if 'filename' in romanisimdict:
    #    romanisimdict['filename'] = str(romanisimdict['filename'])
    #romanisimdict.update(**extras)
    #romanisimdict['version'] = romanisim.__version__

    af = asdf.AsdfFile()
    af.tree = {'roman': im, 'romanisim': romanisimdict}
    af.write_to(open(args.filename, 'wb'))

output_dir = '/global/cfs/cdirs/m4943/grismsim/skycell_mosaics/'
out_base_name = '_stars.asdf'
    
if __name__ == '__main__':
    from multiprocessing import Pool
    skycell_data = Table.read(github_dir+'/grism_sim/data/skycells_subset.ecsv')
    inds = []
    for i in range(0,len(skycell_data)):
        inds.append(skycell_data[i]['ra_center'],skycell_data[i]['dec_center'],output_dir+skycell_data[i]['name']+out_base_name)
    with Pool(processes=nproc) as pool:
        res = pool.map(mkL3, inds)
    

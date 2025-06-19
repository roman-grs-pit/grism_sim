import numpy as np
import yaml
import os
from astropy.table import Table
from grism_sim_psf_dependent import mk_grism
import pysiaf
from astropy.io import fits
import matplotlib.pyplot as plt

github_dir = os.getenv("github_dir")
conf_file = os.path.join(github_dir, "grism_sim/data/grizli_config.yaml")
with open(conf_file) as f:
    grizli_conf = yaml.safe_load(f)

tel_ra = 0
tel_dec = 0
tel_pa = 0
det_num = 1

gpad = grizli_conf["pad"]
det_size = grizli_conf["detector_size"]

cen = int(det_size/2)+gpad

max = det_size + 2*gpad
min = 0

center = [cen, cen]
edges = [[cen, max], [cen, min], [max, cen], [min, cen]]
corners = [[min, min], [min, max], [max, min], [max, max]]

coords = [center, *edges, *corners]

ind = [ii for ii in range(len(coords))]
tmp = [154.0 for _ in range(len(coords))]
mag = [1.0 for _ in range(len(coords))]

siaf = pysiaf.Siaf("roman")
aper = siaf["WFI01_FULL"]
v2, v3 = siaf["WFI_CEN"].V2Ref, siaf["WFI_CEN"].V3Ref
attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, tel_ra, tel_dec, tel_pa)
aper.set_attitude_matrix(attmat)
ra, dec = aper.sci_to_sky(np.asarray([ii[0] for ii in coords]), np.asarray([ii[1] for ii in coords]))

star_input = Table([ind, tmp, mag, ra, dec],
               names=("index", "star_template_index", "magnitude", "RA", "DEC"))

scratch = os.getenv("SCRATCH")

mk_grism(tel_ra,tel_dec,tel_pa,det_num,star_input,None,output_dir=scratch, dogal='n')
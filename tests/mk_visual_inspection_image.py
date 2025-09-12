from grism_sim_psf_dependent import mk_grism
from matplotlib import patches
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib as mpl
import pysiaf
import numpy as np
import os
import subprocess

# outdir = os.path.join(os.getenv("SCRATCH"), "tests")
outdir = os.path.join("/global/cfs/cdirs/m4943/grismsim/visual_inspection")

os.chdir(os.path.join(os.getenv("github_dir"), "grism_sim"))
tag = subprocess.check_output("git describe --tags", shell=True).decode().strip()

tel_ra = 10
tel_dec = 10
tel_pa = 60
det_num = [1, 18]
star_input = None
gal_input = None
output_dir = None
extra_grism_name = "_" + tag
extra_ref_name = "_" + tag

star_ra, star_dec = np.mgrid[0:360:1, 0:360:1]
app_ra, app_dec = np.mgrid[tel_ra-0.5:tel_ra+0.5:0.125, tel_dec-0.5:tel_dec+0.5:0.125]

star_ra = np.append(star_ra.ravel(), app_ra)
star_dec = np.append(star_dec.ravel(), app_dec)

N = star_ra.size
idx = np.arange(N)
star_temp_idx = np.full(N, 154.)
mag = np.full(N, 15)

star_input = Table([idx, star_temp_idx, mag, star_ra, star_dec],
                   names = ["index", "star_template_index", "magnitude", "RA", "DEC"])

gal_ra, gal_dec = np.mgrid[0.5:360:1, 0.5:360:1]
app_ra, app_dec = np.mgrid[tel_ra-0.5+0.0625:tel_ra+0.5+0.0625:0.125, tel_dec-0.5+0.0625:tel_dec+0.5+0.0625:0.125]

gal_ra = np.append(gal_ra.ravel(), app_ra)
gal_dec = np.append(gal_dec.ravel(), app_dec)

N = gal_ra.size
sim = np.full(N, 23)    # For SED selection
idx = np.full(N, 0)     # For SED selection
z = np.full(N, 23)
mag = np.full(N, 20)
unique_ID = np.arange(N)

gal_input = Table([gal_ra, gal_dec, sim, idx, z, mag, unique_ID],
                  names = ["RA","DEC","SIM","IDX","Z","mag_F158_Av1.6523","unique_ID"])

siaf = pysiaf.Siaf("roman")
wfi_cen = siaf["WFI_CEN"]
v2, v3 = wfi_cen.V2Ref, wfi_cen.V3Ref

attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, tel_ra, tel_dec, tel_pa)

apertureList = [siaf[f"WFI{det_num:02}_FULL"] for det_num in range(1, 19)]

verticies = []
for aperture in apertureList:
    aperture.set_attitude_matrix(attmat)
    verticies.append((aperture.AperName,
                      aperture.idl_to_sky([aperture.XIdlVert1, aperture.XIdlVert2, aperture.XIdlVert3, aperture.XIdlVert4], 
                                          [aperture.YIdlVert1, aperture.YIdlVert2, aperture.YIdlVert3, aperture.YIdlVert4])))
    
fig, ax = plt.subplots()
fig.set_size_inches(10,10)
ax.set_title(f"{tag}: Detectors vs Stars")
ax.scatter(star_ra, star_dec, color='r', label="Stars", s=0.25)
ax.scatter(gal_ra, gal_dec, color='g', label="Gals", s=0.25)

cmap = mpl.colormaps["tab20"]

for ii, (apername, coords) in enumerate(verticies):
    poly_points = np.column_stack((coords[0], coords[1]))
    poly_patch = patches.Polygon(poly_points, closed=True, color=cmap(ii), fill=False, label=apername)
    ax.add_patch(poly_patch)

ax.legend()

ax.set_xlim(tel_ra - 1, tel_ra + 1)
ax.set_ylim(tel_ra - 1, tel_ra + 1)

plt.savefig(os.path.join(outdir, f"{tag}_footprint.png"), bbox_inches="tight")

for det_num in range(1, 19):
    mk_grism(tel_ra, tel_dec, tel_pa, det_num, star_input, gal_input, outdir, 
             extra_grism_name=extra_grism_name, extra_ref_name=extra_ref_name)
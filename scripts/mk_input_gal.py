import numpy as np
from astropy.table import Table

import h5py
import glob

local_path = "/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/"

all_sims = glob.glob(local_path + "galacticus_FOV_EVERY100_sub_*.hdf5")

flux_cor = [sim for sim in all_sims if "_flux.hdf5" in sim]
sims = [sim for sim in all_sims if "_flux.hdf5" not in sim]

print(all_sims)
print(len(all_sims))

prefix = "Euclid_Roman_4deg2_radec"


all_RA  = np.array([])
all_DEC = np.array([])
all_sim = np.array([])
all_ind = np.array([])
all_Z = np.array([])
all_lum184 = np.array([])
all_lum158 = np.array([])

for sim_file in sims:
    sim = h5py.File(sim_file, 'r')
    RA = sim['Outputs']['RA'][:]
    DEC = sim['Outputs']['DEC'][:]
    z = sim['Outputs']['ObservedRedshift'][:]
    lum158 = sim['Outputs']['total_Luminosity_Roman_F158_Av1.6523'][:]
    lum184 = sim['Outputs']['total_Luminosity_Roman_F184_Av1.6523'][:]
    
    all_RA  = np.concatenate([all_RA, RA])
    all_DEC = np.concatenate([all_DEC, DEC])
    all_Z = np.concatenate([all_Z, z])
    all_lum184 = np.concatenate([all_lum184, lum184])
    all_lum158 = np.concatenate([all_lum158, lum158])

    N = RA.shape[0]
    num = int(sim_file.split("_")[-1].replace(".hdf5",""))
    all_sim = np.concatenate([all_sim, num*np.ones((N,))])
    all_ind = np.concatenate([all_ind, np.arange(N)])

    sim.close()


tbl = Table([all_RA, all_DEC, all_sim, all_ind,all_Z,all_lum184,all_lum158],
            names=('RA', 'DEC', 'SIM', 'IDX','Z','tot_Lum_F184_Av1.6523','tot_Lum_F158_Av1.6523'),
            dtype=(float, float, int, int,float,float,float),)
h=0.6774
Mpc = 3.08568025E24 # cm
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.3089, Tcmb0=2.725)

lum_distance = cosmo.luminosity_distance(tbl['Z']).value
abM = -2.5*np.log10(tbl['tot_Lum_F184_Av1.6523'])
mag = abM+5*np.log10(lum_distance*1e6) - 5 - 2.5*np.log10(1+tbl['Z'])
tbl['mag_F184_Av1.6523'] = mag
abM = -2.5*np.log10(tbl['tot_Lum_F158_Av1.6523'])
mag = abM+5*np.log10(lum_distance*1e6) - 5 - 2.5*np.log10(1+tbl['Z'])
tbl['mag_F158_Av1.6523'] = mag
tbl['unique_ID'] = tbl['IDX']*1000 + tbl['SIM']


print(tbl.info())

print("Writing catalog")
tbl_file = '/global/cfs/cdirs/m4943/grismsim/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits'
tbl.write(tbl_file,overwrite=True)
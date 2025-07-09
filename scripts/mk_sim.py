from grism_sim_psf_dependent import mk_grism
from multiprocessing import Pool
from astropy.table import Table
import numpy as np
import os, sys
import yaml

outdir = sys.argv[1]
conf_file = os.path.join(outdir, "sim_config.yaml")
with open(conf_file) as f:
    sim_config = yaml.safe_load(f)

# assert block
for var in ["stars", "galaxies"]:
    msg = f"{var} not found in sim_config. If not simulating stars or galaxies, set to null."
    assert var in sim_config, msg

for var in ["tel_ra", "tel_dec"]:
    assert var in sim_config, f"{var} not found in sim_config"
    if isinstance(sim_config[var], dict):
        for dvar in ["start", "step", "num"]:
            msg = f"{dvar} not found in {var} definition in sim_config"
            assert dvar in sim_config[var], msg
    else:
        msg = f"{var} must be float, int, or dict containing start, step, and num keys"
        assert isinstance(sim_config[var], (float, int)), msg
    
msg = "tel_pa must be dictionary with start value and rolls list"
assert isinstance(sim_config["tel_pa"], dict), msg
assert "rolls" in sim_config["tel_pa"], msg
assert "start" in sim_config["tel_pa"], msg
assert isinstance(sim_config["tel_pa"]["rolls"], list), msg

msg = "dither must be null, int, float, or dictionary with ra & dec keys"
assert "dither" in sim_config, msg

if len(sim_config["names_of_sims"]) > 1:
    for sim_num in sim_config["names_of_sims"]:
        msg = "extra_ref_name & extra_grism_name must be set to avoid overwriting files whne performing multiple sims"
        assert "extra_ref_name" in sim_config[sim_num], msg
        assert "extra_grism_name" in sim_config[sim_num], msg

if sim_config["stars"] is not None:
    stars = Table.read(sim_config["stars"])
else:
    stars = None
if sim_config["galaxies"] is not None:
    galaxies = Table.read(sim_config["galaxies"])
else:
    galaxies = None

if sim_config["dither"] is None:
    dither = {"ra": [0], "dec": [0]}
elif isinstance(sim_config["dither"], (float, int)):
    dither = {"ra": [0, sim_config["dither"]],
              "dec": [0, sim_config["dither"]]}
else:
    dither = {"ra": [0, sim_config["dither"]["ra"]],
              "dec": [0, sim_config["dither"]["dec"]]}

if isinstance(sim_config["tel_ra"], (float, int)):
    tel_ra = [sim_config["tel_ra"]]
else:
    start = sim_config["tel_ra"]["start"]
    step = sim_config["tel_ra"]["step"]
    num = sim_config["tel_ra"]["num"]
    tel_ra = [start + (step * ii) + dith for ii in range(0, num) for dith in dither["ra"]]

if isinstance(sim_config["tel_dec"], (float, int)):
    tel_dec = [sim_config["tel_dec"]]
else:
    start = sim_config["tel_dec"]["start"]
    step = sim_config["tel_dec"]["step"]
    num = sim_config["tel_dec"]["num"]
    tel_dec = [start + (step * ii) + dith for ii in range(0, num) for dith in dither["dec"]]

if isinstance(sim_config["tel_pa"], (float, int)):
    tel_pa = [sim_config["tel_pa"]]
else:
    start = sim_config["tel_pa"]["start"]
    rolls = sim_config["tel_pa"]["rolls"]
    tel_pa = [start + roll for roll in rolls]

pointings = []
for ra in tel_ra:
    for dec in tel_dec:
        for pa in tel_pa:
            pointings.append({"ra": ra, 
                            "dec": dec, 
                            "pa": pa})

sims = []

for sim_name in sim_config["names_of_sims"]:
    sim = sim_config[sim_name].copy()
    d = {"star_input": None,
         "gal_input": None}

    if stars is not None:
        sel = np.ones(len(stars), dtype=bool)
        if "star_mag_cutoff" in sim and stars is not None:
            cutoffs = sim.pop("star_mag_cutoff")
            if "brighter_than" in cutoffs:
                sel &= stars["magnitude"] <= cutoffs["brighter_than"]
            if "fainter_than" in cutoffs:
                sel &= stars["magnitude"] >= cutoffs["fainter_than"]
        d["star_input"] = stars[sel]

    if galaxies is not None:
        sel = np.ones(len(galaxies), dtype=bool)
        if "galaxy_mag_cutoff" in sim and galaxies is not None:
            cutoffs = sim.pop("galaxy_mag_cutoff")
            if "brighter_than" in cutoffs:
                sel &= galaxies["magnitude"] <= cutoffs["brighter_than"]
            if "fainter_than" in cutoffs:
                sel &= galaxies["magnitude"] >= cutoffs["fainter_than"]
        d["gal_input"] = galaxies[sel]

    scas = sim.pop("SCAs")
    if scas=="all":
        det_nums = list(range(1, 19))
    elif isinstance(scas, int):
        det_nums = [scas]
    else:
        det_nums = [ii for ii in scas]

    for det_num in det_nums:
        sims.append({"det_num": det_num,
                     **sim
                     })

all_sims = []
for pointing in pointings:
    for sim in sims:
        all_sims.append({
            **pointing,
            **sim
        })

def dosim(d, **kwargs):
    mk_grism(tel_ra = d["ra"],
             tel_dec = d["dec"],
             tel_pa = d["pa"],
             det_num = d["det_num"],
             star_input = d["star_input"],
             gal_input = d["gal_input"],
             output_dir = outdir,
             **kwargs)

with Pool(processes=108) as pool:
    res = pool.map(dosim, all_sims)
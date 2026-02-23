from grism_sim_psf_dependent import mk_grism, try_wait_loop
from wrap_with_romanisim import wrap_with_romanisim
from multiprocessing import Pool
import combine_img_utils as ciu
from astropy.table import Table
import numpy as np
import os
import yaml
import argparse
import file_handling_utils as fhu

parser = argparse.ArgumentParser()
parser.add_argument("outdir")
parser.add_argument("--incomplete", action="store_true")
parser.add_argument("--overwrite_sim_args", action="store_true", default=False)
parser.add_argument("--nprocesses", type=int, default=80)
args = parser.parse_args()
outdir = args.outdir

def parse_sim_config(yaml_dir, save_args=True, overwrite=False):
    conf_file = os.path.join(yaml_dir, "sim_config.yaml")
    with open(conf_file) as f:
        sim_config = yaml.safe_load(f)

    # assert block
    for var in ["stars", "galaxies"]:
        msg = f"{var} not found in sim_config. If not simulating stars or galaxies, set to null."
        assert var in sim_config, msg

    for var, backwards_compatible_var in [("wfi_cen_ra", "tel_ra"), ("wfi_cen_dec", "tel_dec")]:
        try:
            assert var in sim_config, f"{var} not found in sim_config"

        # Check old variable names for backwards compatibility
        except AssertionError as e:
            if backwards_compatible_var in sim_config:
                sim_config[var] = sim_config[backwards_compatible_var]
            else:
                raise e

        if isinstance(sim_config[var], dict):
            for dvar in ["start", "step", "num"]:
                msg = f"{dvar} not found in {var} definition in sim_config"
                assert dvar in sim_config[var], msg
        else:
            msg = f"{var} must be float, int, or dict containing start, step, and num keys"
            assert isinstance(sim_config[var], (float, int)), msg



    assert "seed" in sim_config, "Random seed not found in sim_config. Please define rng seed in sim_config.yaml"

    msg = "wfi_cen_pa must be dictionary with start, value, and rolls list"
    assert isinstance(sim_config["wfi_cen_pa"], dict), msg
    assert "rolls" in sim_config["wfi_cen_pa"], msg
    assert "start" in sim_config["wfi_cen_pa"], msg
    assert isinstance(sim_config["wfi_cen_pa"]["rolls"], list), msg

    msg = "dither must be null, int, float, or dictionary with ra & dec keys"
    assert "dither" in sim_config, msg

    if len(sim_config["names_of_sims"]) > 1:
        for sim_num in sim_config["names_of_sims"]:
            msg = "extra_ref_name & extra_grism_name must be set to avoid overwriting files whne performing multiple sims"
            assert "extra_ref_name" in sim_config[sim_num], msg
            assert "extra_grism_name" in sim_config[sim_num], msg

    # Read catalogs
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

    if isinstance(sim_config["wfi_cen_ra"], (float, int)):
        wfi_cen_ra = [sim_config["wfi_cen_ra"]]
    else:
        start = sim_config["wfi_cen_ra"]["start"]
        step = sim_config["wfi_cen_ra"]["step"]
        num = sim_config["wfi_cen_ra"]["num"]
        wfi_cen_ra = [start + (step * ii) for ii in range(0, num)]

    if isinstance(sim_config["wfi_cen_dec"], (float, int)):
        wfi_cen_dec = [sim_config["wfi_cen_dec"]]
    else:
        start = sim_config["wfi_cen_dec"]["start"]
        step = sim_config["wfi_cen_dec"]["step"]
        num = sim_config["wfi_cen_dec"]["num"]
        wfi_cen_dec = [start + (step * ii) for ii in range(0, num)]

    if isinstance(sim_config["wfi_cen_pa"], (float, int)):
        wfi_cen_pa = [sim_config["wfi_cen_pa"]]
    else:
        start = sim_config["wfi_cen_pa"]["start"]
        rolls = sim_config["wfi_cen_pa"]["rolls"]
        wfi_cen_pa = [start + roll for roll in rolls]

    pointings = []
    for ra in wfi_cen_ra:
        for dec in wfi_cen_dec:
            for dither_ra, dither_dec in zip(dither["ra"], dither["dec"]):
                ra += dither_ra
                dec += dither_dec
                for pa in wfi_cen_pa:
                    pointings.append({"wfi_cen_ra": ra,
                                      "wfi_cen_dec": dec,
                                      "wfi_cen_pa": pa})

    sims = []
    seed = sim_config["seed"]
    for sim_name in sim_config["names_of_sims"]:
        sim = sim_config[sim_name].copy()
        catalogs = {"star_input": None,
                    "gal_input": None}

        if stars is not None:
            sel = np.ones(len(stars), dtype=bool)
            if "star_mag_cutoff" in sim and stars is not None:
                cutoffs = sim.pop("star_mag_cutoff")
                if "brighter_than" in cutoffs:
                    sel &= stars["magnitude"] <= cutoffs["brighter_than"]
                if "fainter_than" in cutoffs:
                    sel &= stars["magnitude"] > cutoffs["fainter_than"]
            catalogs["star_input"] = stars[sel]

        if galaxies is not None:
            sel = np.ones(len(galaxies), dtype=bool)
            if "galaxy_mag_cutoff" in sim and galaxies is not None:
                cutoffs = sim.pop("galaxy_mag_cutoff")
                if "brighter_than" in cutoffs:
                    sel &= galaxies["mag_F158_Av1.6523"] <= cutoffs["brighter_than"]
                if "fainter_than" in cutoffs:
                    sel &= galaxies["mag_F158_Av1.6523"] > cutoffs["fainter_than"]
            catalogs["gal_input"] = galaxies[sel]

        scas = sim.pop("SCAs")
        assert isinstance(scas, (int, list, tuple, str))
        if isinstance(scas, int):
            det_nums = [scas]
        elif isinstance(scas, (list, tuple)):
            det_nums = [ii for ii in scas]
        else:
            det_nums = list(range(1, 19))

        for det_num in det_nums:
            sims.append({"seed": seed,
                         "det_num": det_num,
                         **catalogs,
                         **sim
                         })

    all_sim_args = []
    for pointing in pointings:
        for sim in sims:
            all_sim_args.append({
                **pointing,
                **sim
            })

    if "combine_sim" in sim_config:
        combine_sim_args = {"combine": True,
                        "seed": seed}
    else:
        combine_sim_args = {"combine": False}

    if save_args:
        save_sim_args_list = []
        for d in all_sim_args:
            info = d.copy()
            info.pop("star_input", None)
            info.pop("gal_input", None)
            save_sim_args_list.append(info)

        # if file exists, consider appending
        sim_args_path = os.path.join(yaml_dir, "sim_args.yaml")
        if overwrite or not os.path.exists(sim_args_path):
            with open(sim_args_path, "w") as f:
                yaml.dump(save_sim_args_list, f)
        else:
            with open(sim_args_path, "a") as f:
                f.write("\n#" + '-' * 60 + "\n")
                yaml.dump(save_sim_args_list)

        del save_sim_args_list

    return all_sim_args, combine_sim_args

def dosim(dt) -> None:
    mk_grism(output_dir = outdir,
             **dt)

def combine_grisms(dt) -> None:
    if fhu.is_complete(os.path.join(outdir, dt[0] + ".fits")):
        return None
    ciu.combine_sims(outdir, *dt, seed=combine_args["seed"])
    return None

def combine_refs(dt) -> None:
    if fhu.is_complete(os.path.join(outdir, dt[0] + ".fits"), is_ref=True):
        return None
    ciu.combine_refs(outdir, *dt)
    return None

if __name__ == "__main__":
    all_sims, combine_args = parse_sim_config(outdir, overwrite=args.overwrite_sim_args)

    if args.incomplete:
        all_sims = fhu.trim_complete_sims(outdir, all_sims)

    with Pool(processes=args.nprocesses) as pool:
        pool.map(dosim, all_sims)

    if combine_args["combine"]:

        # group files; wrapped in try_wait_loop for NERSC timing/race issue
        grouped_grisms = try_wait_loop(ciu.group_grism_files, outdir, all_sims)
        grouped_refs = try_wait_loop(ciu.group_ref_files, outdir, all_sims)

        # use half of nprocesses for ref_combination; use remainder for grisms
        ref_proc = args.nprocesses // 2
        with Pool(processes=args.processe - ref_proc) as grism_pool, Pool(processes=ref_proc) as ref_pool:
            grism_res = grism_pool.map_async(combine_grisms, grouped_grisms.items())
            ref_res = ref_pool.map_async(combine_refs, grouped_refs.items())
            grism_res.wait()
            ref_res.wait()

    wrap_with_romanisim(outdir)
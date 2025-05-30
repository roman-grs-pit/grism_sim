#!/bin/bash
module load conda
conda activate /global/common/software/m4943/grizli0
PYTHONPATH=$PYTHONPATH:/global/common/software/m4943/grizli0/grism_sim/py
export STPSF_PATH="/global/cfs/cdirs/m4943/grismsim/stpsf-data"
export github_dir=/global/common/software/m4943/grizli0/

for (( ndith=0;ndith<=1;ndith++ ))
do
  for (( ndec=0;ndec<=3;ndec++ ))
  do
    for (( nra=0;nra<=1;nra++ ))
    do
    srun -N 1 -C cpu -t 04:00:00 --qos interactive --account m4943 python scripts/mk_grism_par_fullarea_run1_chunk.py --dith_ind $ndith --dec_ind $ndec --ra_ind $nra
    #echo $ndith $ndec $nra
    done
  done
done  
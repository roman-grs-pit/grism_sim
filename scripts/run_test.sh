#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J "Test stpsf and load_psf"
#SBATCH --mail-user=keith.buckholz@yale.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4943
#SBATCH -t 0:2:0
#SBATCH -L cfs

srun python /global/common/software/m4943/grizli0/grism_sim/scripts/mk_grism_test_case.py

#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 54
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J "Stage 1 Simulation Run"
#SBATCH --mail-user=keith.buckholz@yale.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4943
#SBATCH -t 2:0:0
#SBATCH -L cfs

srun python /global/common/software/m4943/grizli0/grism_sim/scripts/mk_stage_1.py
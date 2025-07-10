#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 108
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J "Stage 2 Simulation Run"
#SBATCH --mail-user=keith.buckholz@yale.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4943
#SBATCH -t 4:0:0
#SBATCH -L cfs

srun python /global/common/software/m4943/grizli0/grism_sim/scripts/mk_sim.py $SCRATCH/stage_2
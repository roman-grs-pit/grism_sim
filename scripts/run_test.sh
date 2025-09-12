#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J "Make Visual Inspection on Current Tag"
#SBATCH --mail-user=keith.buckholz@yale.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4943
#SBATCH -t 0:0:30
#SBATCH -L cfs

srun python /global/common/software/m4943/grizli0/grism_sim/tests/mk_visual_inspection_image.py

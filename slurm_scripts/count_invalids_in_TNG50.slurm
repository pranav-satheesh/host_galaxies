#!/bin/bash

#SBATCH --qos=lblecha-b
#SBATCH -J invalid_TNG
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/tng_invalid_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/tng_invalid_%j.err
ml conda
conda activate hfrl


codePath='/home/pranavsatheesh/host_galaxies/py_files'
python "${codePath}/check_invalid_mergers_And_Controls_in_TNG.py"
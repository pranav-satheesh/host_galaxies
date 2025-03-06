#!/bin/sh

#SBATCH --qos=lblecha
#SBATCH -J control-sample
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/control_sample_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/control_sample_%j.err


ml conda
conda activate hfrl

codePath=/home/pranavsatheesh/host_galaxies/py_files

populationfilePath=/home/pranavsatheesh/host_galaxies/merger_file_runs/

controlfilePath=/home/pranavsatheesh/host_galaxies/data/control_files/


python ${codePath}/create_control_sample.py $populationfilePath $controlfilePath

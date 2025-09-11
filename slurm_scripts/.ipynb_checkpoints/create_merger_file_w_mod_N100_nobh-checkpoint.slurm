#!/bin/sh

#SBATCH --qos=lblecha
#SBATCH -J mrg_file_creation_TNG50-1-N100
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/merger_file_creation_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/errors/merger_file_creation_%j.err

ml conda
conda activate hfrl

basePath=/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output
codePath=/home/pranavsatheesh/host_galaxies/py_files
savePath=/home/pranavsatheesh/host_galaxies/data/merger_files
snapnum=99
minNdm=100
minNgas=100
minNstar=100
minNbh=0

python ${codePath}/merger_file_from_sublink_trees_mod.py $basePath $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
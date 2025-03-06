#!/bin/sh

#SBATCH --qos=lblecha-b
#SBATCH -J mrgt50-1-N100
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200


module load conda
conda activate hfrl 

basePath=/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output
codePath=/home/pranavsatheesh/host_galaxies/merger_file_runs

snapnum=99
minNdm=100
minNgas=100
minNstar=100
minNbh=1

python ${codePath}/merger_file_from_sublink_trees.py $basePath $snapnum $minNdm $minNgas $minNstar $minNbh $codePath

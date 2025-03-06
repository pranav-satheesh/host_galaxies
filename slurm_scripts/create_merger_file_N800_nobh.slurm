#!/bin/sh

#SBATCH --qos=lblecha-b
#SBATCH -J mrgt50-1-nobh-N800
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=lblecha@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200


ml conda
conda activate holo310

basePath=/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output
codePath=/home/lblecha/repos/lb_illustris_python
snapnum=99
minNdm=800
minNgas=800
minNstar=800
minNbh=0

python ${codePath}/merger_file_from_sublink_trees.py $basePath $snapnum $minNdm $minNgas $minNstar $minNbh

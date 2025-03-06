#!/bin/sh

#SBATCH --qos=lblecha-b
#SBATCH -J population_sort
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/population_sort_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/population_sort_%j.err


ml conda
conda activate hfrl

basePath=/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output
codePath=/home/pranavsatheesh/host_galaxies/py_files
merger_file_path=/home/pranavsatheesh/host_galaxies/merger_file_runs
minNdm=100
minNgas=100
minNstar=100
minNbh=1

python ${codePath}/population_sort_by_mergers.py $basePath $merger_file_path $minNdm $minNgas $minNstar $minNbh
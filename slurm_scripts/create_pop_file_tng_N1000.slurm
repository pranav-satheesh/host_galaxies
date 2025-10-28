#!/bin/sh

#SBATCH --qos=lblecha-b
#SBATCH -J ptng_opulation_sort
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 50:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/tng_pop/tng_population_sort_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/tng_pop/tng_population_sort_%j.err

ml conda
conda activate hfrl

basePath=/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output
codePath=/home/pranavsatheesh/host_galaxies/py_files
merger_file_path=/home/pranavsatheesh/host_galaxies/data/merger_files
pop_file_path=/home/pranavsatheesh/host_galaxies/data/population_files

minNdm=0               
minNgas=0
minNstar=1000
minNbh=1

python ${codePath}/population_sort_mergers_and_non_mergers.py TNG $basePath $merger_file_path $pop_file_path $minNgas $minNdm $minNstar $minNbh

 
#!/bin/sh
#SBATCH --qos=lblecha
#SBATCH -J tng_and_brahma_population
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

TNG_basePath=/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output
codePath=/home/pranavsatheesh/host_galaxies/py_files
merger_file_path=/home/pranavsatheesh/host_galaxies/data/merger_files
pop_file_path=/home/pranavsatheesh/host_galaxies/data/population_files

minNdm=0               
minNgas=0
minNstar=1000
minNbh=1

python ${codePath}/population_sort_mergers_and_non_mergers.py TNG $TNG_basePath $merger_file_path $pop_file_path $minNgas $minNdm $minNstar $minNbh

minNdm=0
minNgas=0
minNstar=10
minNbh=1

basePath1=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_TNG/
basePath2=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_TNG/
basePath3=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_LOWSPIN_TNG/
basePath4=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_LOWSPIN_RICH_TNG/
basePath5=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_DFD_3_TNG/
basePath6=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_DFD_3_TNG/

basePaths=($basePath1 $basePath2 $basePath3 $basePath4 $basePath5 $basePath6)
for basePath in "${basePaths[@]}"; do
    python ${codePath}/population_sort_mergers_and_non_mergers.py brahma $basePath $merger_file_path $pop_file_path $minNgas $minNdm $minNstar $minNbh
done
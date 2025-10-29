#!/bin/sh
#SBATCH --qos=lblecha-b
#SBATCH -J all_sim_stat_run
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3500
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/all_subhalos_property_run_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/all_subhalos_property_run_%j.error

ml conda
conda activate hfrl

basePath1=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_TNG/
basePath2=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_TNG/
basePath3=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_LOWSPIN_TNG/
basePath4=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_LOWSPIN_RICH_TNG/

brahma_basePaths=($basePath1 $basePath2 $basePath3 $basePath4)
tng_basepath=/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output

codePath=/home/pranavsatheesh/host_galaxies/py_files
save_loc=/home/pranavsatheesh/host_galaxies/data/

brahma_minNdm=0
brahma_minNgas=0
brahma_minNstar=10
brahma_minNbh=0

tng_minNdm=0
tng_minNgas=0
tng_minNstar=1000
tng_minNbh=0


for brahma_basePath in "${brahma_basePaths[@]}"; do
    python ${codePath}/sim_subhalo_property_statistics.py brahma $brahma_basePath $save_loc $brahma_minNgas $brahma_minNdm $brahma_minNstar $brahma_minNbh
done

python ${codePath}/sim_subhalo_property_statistics.py tng $tng_basepath $save_loc $tng_minNgas $tng_minNdm $tng_minNstar $tng_minNbh
#!/bin/sh

#SBATCH --qos=lblecha
#SBATCH -J brahma_mrg_all_runs
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/brahma_mrg/merger_file_creation_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/brahma_mrg/merger_file_creation_%j.err

ml conda
conda activate hfrl

basePath1=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_TNG/
#basePath2=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_DFD_3_TNG/
basePath3=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_TNG/
basePath4=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_DFD_3_TNG/
basePath5=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_LOWSPIN_TNG/
basePath6=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_LOWSPIN_RICH_TNG/
basePath7=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_LOWSPIN_RICH_DFD_3_TNG-SE_AGN0.01/
basePath8=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LW10_LOWSPIN_RICH_DFD_3_TNG_AGN0.01/

codePath=/home/pranavsatheesh/host_galaxies/py_files
savePath=/home/pranavsatheesh/host_galaxies/data/merger_files
snapnum=32
minNdm=0
minNgas=0
minNstar=10
minNbh=1

# Array of base paths
basePaths=($basePath1 $basePath2 $basePath3 $basePath4 $basePath5 $basePath6 $basePath7 $basePath8)

for basePath in "${basePaths[@]}"; do
    python ${codePath}/merger_file_from_brahma_tree.py $basePath $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
done
# python ${codePath}/merger_file_from_brahma_tree.py $basePath1 $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
# python ${codePath}/merger_file_from_brahma_tree.py $basePath2 $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
# python ${codePath}/merger_file_from_brahma_tree.py $basePath3 $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
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
basePath2=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_DF3_TNG/
basePath3=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LOWSPIN_RICH_TNG/
codePath=/home/pranavsatheesh/host_galaxies/py_files
savePath=/home/pranavsatheesh/host_galaxies/data/merger_files
snapnum=32
minNdm=0
minNgas=0
minNstar=10
minNbh=1

python ${codePath}/merger_file_from_brahma_tree.py $basePath1 $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
python ${codePath}/merger_file_from_brahma_tree.py $basePath2 $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
python ${codePath}/merger_file_from_brahma_tree.py $basePath3 $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
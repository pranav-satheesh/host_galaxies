#!/bin/sh

#SBATCH --qos=lblecha
#SBATCH -J brahma-N100_mrg_file
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/brahma_mrg/merger_file_creation_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/brahma_mrg/merger_file_creation_%j.err

ml conda
conda activate hfrl

basePath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_DFD_3/
codePath=/home/pranavsatheesh/host_galaxies/py_files
savePath=/home/pranavsatheesh/host_galaxies/data/merger_files
snapnum=32
minNdm=0
minNgas=0
minNstar=100
minNbh=0

python ${codePath}/merger_file_from_brahma_tree.py $basePath $snapnum $minNdm $minNgas $minNstar $minNbh $savePath
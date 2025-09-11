#!/bin/sh

#SBATCH --qos=lblecha
#SBATCH -J tree_connect_brahma_run
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/brahma_tree_run_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/brahma_tree_run_%j.err

ml conda
conda activate hfrl

runPath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS//L12p5n512/AREPO/output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_DFD_3/
runName=output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_DFD_3
savePath=/home/pranavsatheesh/host_galaxies/data/
codePath=/home/pranavsatheesh/host_galaxies/py_files

python ${codePath}/brahma_extract_tree_info_for_bh_mergers.py $runPath $runName $savePath



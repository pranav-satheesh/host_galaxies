#!/bin/sh

#SBATCH --qos=lblecha
#SBATCH -J population_sort_brahma
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

basePath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_spin_rich/
codePath=/home/pranavsatheesh/host_galaxies/py_files
merger_file_name=merger_statistics_subhalo_remnants.npy
pop_file_path=/home/pranavsatheesh/host_galaxies/data/population_files

# minNdm=100
# minNgas=100
# minNstar=100
# minNbh=1

python ${codePath}/population_sort_by_mergers.py brahma $basePath $merger_file_name $pop_file_path 
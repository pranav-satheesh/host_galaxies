#!/bin/sh

#SBATCH --qos=lblecha
#SBATCH -J pop_brhm_all_runs
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/brahma_pop/brahma_population_sort_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/brahma_pop/brahma_population_sort_%j.err


ml conda
conda activate hfrl


basePath1=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_DFD_3
basePath2=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_spin
basePath3=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_spin_rich
basePath4=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/output_Msfmp5_Mh1000_double_powerlaw_env_5_0.1_0.3_slope1.6
basePath5=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/output_Msfmp5_Mh3000_double_powerlaw_env_5_0.1_0.3_slope1.6
basePath6=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_TNG/
basePath7=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_DF3_TNG/
basePath8=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_LOWSPIN_RICH_TNG/
codePath=/home/pranavsatheesh/host_galaxies/py_files
merger_file_loc=/home/pranavsatheesh/host_galaxies/data/merger_files
pop_file_path=/home/pranavsatheesh/host_galaxies/data/population_files

minNdm=0
minNgas=0
minNstar=10
minNbh=1

# Array of base paths
basePaths=($basePath1 $basePath2 $basePath3 $basePath4 $basePath5 $basePath6 $basePath7 $basePath8)

# Loop through each base path
for basePath in "${basePaths[@]}"; do
    python ${codePath}/population_sort_by_mergers.py brahma $basePath $merger_file_loc $pop_file_path $minNgas $minNdm $minNstar $minNbh
done
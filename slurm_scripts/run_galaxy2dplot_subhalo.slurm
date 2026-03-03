#!/bin/sh

#SBATCH --qos=lblecha-b
#SBATCH -J galaxy2d_subhalo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/2d_plots/galaxy2d_subhalo_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/2d_plots/galaxy2d_subhalo_%j.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=64G
#SBATCH -t 04:00:00

ml conda
conda activate cosmosims

basePath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_TNG
codePath=/home/pranavsatheesh/host_galaxies/py_files
snap_list=(32 31) # List of snapshot numbers to process. Adjust as needed.
subhaloID_list=(0 1 2) # List of subhalo IDs to process. Adjust as needed.
outDir=/home/pranavsatheesh/host_galaxies/figures/2d_plots
boxLength=50
boxWidth=50
boxHeight=5
nbins=100
vmin=-4
vmax=4
ptype=0
particle_property='Density'

for snap in "${snap_list[@]}"; do
  for subhaloID in "${subhaloID_list[@]}"; do
    python ${codePath}/run_galaxy2dplot_subhalo.py \
      $basePath \
      $snap \
      $subhaloID \
      $outDir \
      $boxLength \
      $boxWidth \
      $boxHeight \
      $nbins \
      $vmin \
      $vmax \
      $particle_property
  done
done

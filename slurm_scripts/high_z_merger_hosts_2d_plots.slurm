#!/bin/bash
#SBATCH --qos=lblecha-b
#SBATCH -J merger_hosts_high_z
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/2d_plots/merger_hosts_high_z_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/2d_plots/merger_hosts_high_z_%j.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=64G
#SBATCH -t 04:00:00

ml conda
conda activate cosmosims

basePath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_DFD_3_TNG
filePath=/home/pranavsatheesh/host_galaxies/data/SM5_DFD_3_TNG_z4-5_positive_sSFR_negative_sBHAR.hdf5
codePath=/home/pranavsatheesh/host_galaxies/py_files
outDir=/home/pranavsatheesh/host_galaxies/figures/2d_plots/z4
boxLength=60
boxWidth=60
boxHeight=60
nbins=500
# vmin=-4
# vmax=4
ptype=0
# particle_property='Density'

python ${codePath}/run_galaxy2dplot_subhalo.py \
  $filePath \
  $basePath \
  $boxLength \
  $boxWidth \
  $boxHeight \
  $nbins \
  $outDir \
  $ptype

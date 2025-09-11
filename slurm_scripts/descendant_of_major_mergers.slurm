#!/bin/sh

#SBATCH --qos=lblecha
#SBATCH -J descendant_of_major_mergers
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/descendants_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/errors/descendants_%j.err

ml conda
conda activate hfrl

codePath=/home/pranavsatheesh/host_galaxies/py_files
merger_file_path=/home/pranavsatheesh/host_galaxies/data/merger_files

python ${codePath}/find_descendants_of_major_mergers.py $merger_file_path

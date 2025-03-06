#!/bin/bash

#SBATCH --job-name=testing_slurm
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --output=../slurm_outputs/test_job_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000mb
#SBATCH --time=10:00

pwd;hostname;date

module load conda
conda activate hfrl 

echo "job running"

python test_script.py
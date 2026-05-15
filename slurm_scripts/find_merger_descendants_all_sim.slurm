#!/bin/bash

#SBATCH --qos=lblecha
#SBATCH -J descednant_all_sims
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/descednant_all_sims_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/descednant_all_sims_%j.err
ml conda
conda activate hfrl

# Configuration
brahma_basepath="/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/"
brahma_simName_array=(
    'SM5_TNG'
    'SM5_LW10_TNG'
    'SM5_DFD_3_TNG'
    'SM5_LW10_DFD_3_TNG'
    'SM5_LW10_LOWSPIN_TNG'
    'SM5_LW10_LOWSPIN_RICH_TNG'
)
TNG_basepath='/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output'
codePath='/home/pranavsatheesh/host_galaxies/py_files'
merger_file_loc='/home/pranavsatheesh/host_galaxies/data/merger_files'
descendant_file_loc='/home/pranavsatheesh/host_galaxies/data/descendant_files/'

# BRAHMA parameters
brahma_minNgas=0
brahma_minNdm=0
brahma_minNstar=10
brahma_minNbh=1

# TNG parameters
TNG_minNgas=0
TNG_minNdm=0
TNG_minNstar=1000
TNG_minNbh=1

echo "======================================="
echo "Processing merger descendants"
echo "Start: $(date)"
echo "======================================="

# Process BRAHMA simulations
echo -e "\n[1/1] Processing BRAHMA simulations..."
for simName in "${brahma_simName_array[@]}"; do
    echo "  - $simName"
    brahma_basepath_sim="${brahma_basepath}${simName}/"
    
    python "${codePath}/merger_descendants.py" "${brahma_basepath_sim}" "${simName}" "${merger_file_loc}" "${brahma_minNgas}" "${brahma_minNdm}" "${brahma_minNstar}" "${brahma_minNbh}" "${descendant_file_loc}"

done

#Process TNG50
echo -e "\n[2/7] Processing TNG50-1..."
python "${codePath}/merger_descendants.py" "${TNG_basepath}" "TNG50-1" "${merger_file_loc}" "${TNG_minNgas}" "${TNG_minNdm}" "${TNG_minNstar}" "${TNG_minNbh}" "${descendant_file_loc}"

echo -e "\n======================================="
echo "Processing complete!"
echo "End: $(date)"
echo "======================================="
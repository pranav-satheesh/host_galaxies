#!/bin/bash
#SBATCH --qos=lblecha-b
#SBATCH -J sSFR_sBHAR_evolution
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 50:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --output=/home/pranavsatheesh/host_galaxies/slurm_outputs/sSFR_sBHAR_evolution_%j.out
#SBATCH --error=/home/pranavsatheesh/host_galaxies/slurm_outputs/sSFR_sBHAR_evolution_%j.err
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
descendant_file_loc='/home/pranavsatheesh/host_galaxies/data/descendant_files/'
pop_file_loc='/home/pranavsatheesh/host_galaxies/data/population_files/'
output_dir='/home/pranavsatheesh/host_galaxies/data/sSFR_sBHAR_evolution/'

# Create output directory if it doesn't exist
mkdir -p "${output_dir}"

# BRAHMA parameters
brahma_minNgas=10
brahma_minNdm=10
brahma_minNstar=10
brahma_minNbh=1

# TNG parameters
TNG_minNgas=0
TNG_minNdm=0
TNG_minNstar=1000
TNG_minNbh=1

echo "======================================="
echo "Processing sSFR and sBHAR evolution"
echo "Start: $(date)"
echo "======================================="

# # Process BRAHMA simulations
# echo -e "\n[BRAHMA] Processing BRAHMA simulations..."
# for simName in "${brahma_simName_array[@]}"; do
#     echo "  - $simName"
#     brahma_basepath_sim="${brahma_basepath}${simName}/"
    
#     python "${codePath}/find_sSFR_sBHAR_evolution_parallel.py" \
#         "${brahma_basepath_sim}" \
#         "${simName}" \
#         "${descendant_file_loc}" \
#         "${pop_file_loc}" \
#         "${brahma_minNgas}" \
#         "${brahma_minNdm}" \
#         "${brahma_minNstar}" \
#         "${brahma_minNbh}" \
#         "${output_dir}"
    
#     if [ $? -eq 0 ]; then
#         echo "    ✓ Completed successfully"
#     else
#         echo "    ✗ Failed with error code $?"
#     fi
# done

# Process TNG50
echo -e "\n[TNG] Processing TNG50-1..."
python "${codePath}/find_sSFR_sBHAR_evolution_parallel.py" \
    "${TNG_basepath}" \
    "TNG50-1" \
    "${descendant_file_loc}" \
    "${pop_file_loc}" \
    "${TNG_minNgas}" \
    "${TNG_minNdm}" \
    "${TNG_minNstar}" \
    "${TNG_minNbh}" \
    "${output_dir}"

if [ $? -eq 0 ]; then
    echo "  ✓ Completed successfully"
else
    echo "  ✗ Failed with error code $?"
fi

echo -e "\n======================================="
echo "Processing complete!"
echo "Results saved to: ${output_dir}"
echo "End: $(date)"
echo "======================================="

# List output files
echo -e "\nGenerated files:"
ls -lh "${output_dir}"sSFR_sBHAR_evolution_*.hdf5

#!/bin/sh
#!/bin/sh
#SBATCH --qos=lblecha-b
#SBATCH -J brahma_subhalo
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

basePath=/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/SM5_DFD_3_TNG
codePath=/home/pranavsatheesh/host_galaxies/py_files
outDir=/home/pranavsatheesh/host_galaxies/figures/2d_plots


boxLength=50
boxWidth=50
boxHeight=10
nbins=500
vmin=-4
vmax=4
ptype=0
particle_property='Density'
hsml_key=False
snap_subhalo_pairs=("32:0" "25:2384" "24:1004" "23:391")
plot_bh_overlay=True
radiative_efficiency=0.1
view='xy'


for pair in "${snap_subhalo_pairs[@]}"; do
  snap=${pair%:*}
  subhaloID=${pair#*:}
  
  python ${codePath}/plot_subhalo_2d_density.py \
    $basePath \
    $snap \
    $subhaloID \
    $ptype \
    $outDir \
    $hsml_key \
    $plot_bh_overlay \
    $radiative_efficiency \
    $view
done

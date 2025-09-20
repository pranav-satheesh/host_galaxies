#!/bin/bash
#SBATCH --export=NONE
##SBATCH --ntasks-per-node=1
#SBATCH --account=lblecha
#SBATCH --qos=lblecha
##SBATCH --reservation=bluetest
#SBATCH --time=72:00:00
#SBATCH --job-name=z_11
#SBATCH --output=cs_tagging.stdout
#SBATCH --error=cs_tagging.stderr
#SBATCH --mem-per-cpu=40000mb
##SBATCH --time=20:00:00
#SBATCH --mail-user=akbhowmi@alumni.cmu.edu
#SBATCH --mail-type=END,FAIL

#python get_subhalo_mass_luminosity_relations.py
#python connect_black_hole_systems_to_AGN_activity_and_massive_BH_new_max_lum_bound_median_all_eddington.py
#python finding_blackhole_systems_MBII.py

#python get_AGN_luminosity_functions.py
#python get_particle_history.py
#python generate_group_ids.py
#python generate_mass_functions_with_bolometric_cuts.py
#python convert_blackhole_data_to_hdf5.py
#python create_central_satellite_tags_max_lum.py
#python generate_maximum_gas_denisty_for_FOFs_with_BHs.py
#python connect_black_hole_systems_to_AGN_activity_and_massive_BH_new_max_lum_bound_median_all_eddington_mass_ratio_cut_random.py
#python get_metallicity_vs_redshift.py
#python get_SFmass_vs_redshift.py
#python get_halomass_SFRZmass_relation.py
#python create_merger_trees.py
#python generate.py
#python generate_halo_gas_spins_updated.py
#python Kinematic_decomposition_groups.py
#python test.py
#python generate_gas_evolution.py
#python generate_halo_LW_properties.py
#python generate_fields.py
#python create_images.py
#python get_cube_overdensities_zoom_region_of_interest.py
#python get_BH_distances_from_halo_centers2_seq.py $1
#python get_progenitor_halos_of_mergers.py $1
#python convert_details_to_hdf5.py
#python get_NFW_fits.py
python get_subhalo_properties_of_mergers.py

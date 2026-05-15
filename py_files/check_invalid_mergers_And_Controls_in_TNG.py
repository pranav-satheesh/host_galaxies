import os
import sys
import numpy as np
import h5py
import illustris_python as il
import control_sample as control
import host_galaxy_enhancement_plots as hostplot
from tqdm import tqdm
h = 0.6774

pop_file_path = '/home/pranavsatheesh/host_galaxies/data/population_files/' 
tng_pop_file_path = '/home/pranavsatheesh/host_galaxies/data/population_files/'

TNG_basepath = '/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output'
TNG_50_pop = control.load_pop_file(TNG_basepath,tng_pop_file_path,[0,0,1000,1])
TNG_50_control = control.control_samples_TNG(TNG_50_pop)


def count_invalid_controls_tng50(sim_obj):
    count = 0
    for i in tqdm(range(len(sim_obj.subhalo_ids_controls))):
        snap_i = sim_obj.pop['non_merging_population']['snap'][:][sim_obj.valid_control_indices][i]
        subhaloid_i = sim_obj.subhalo_ids_controls[i]
        subhalolensinthisnaps = il.groupcat.loadSubhalos(
            TNG_basepath, snap_i, fields=['SubhaloLenType', 'SubhaloBHMass', 'SubhaloBHMdot']
        )
        Ngas = subhalolensinthisnaps['SubhaloLenType'][subhaloid_i, 0]
        BHMdot = subhalolensinthisnaps['SubhaloBHMdot'][subhaloid_i]
        if Ngas == 0 and BHMdot > 0:
            count += 1
    return count

def count_invalid_mergers_tng50(sim_obj):
    count = 0
    for i in tqdm(range(20), desc="Counting invalid TNG50 mergers"):
        snap_i = sim_obj.pop['merging_population']['snap'][:][sim_obj.valid_merger_indices][i]
        subhaloid_i = sim_obj.subhalo_ids_mergers[i]
        subhalolensinthisnaps = il.groupcat.loadSubhalos(
            TNG_basepath, snap_i, fields=['SubhaloLenType', 'SubhaloBHMass', 'SubhaloBHMdot']
        )
        Ngas = subhalolensinthisnaps['SubhaloLenType'][subhaloid_i, 0]
        BHMdot = subhalolensinthisnaps['SubhaloBHMdot'][subhaloid_i]
        print(Ngas, BHMdot)
        if Ngas == 0 and BHMdot > 0:
            count += 1
        print(count)
    print(f"Total invalid TNG50 mergers: {count}")
    return count

if __name__ == "__main__":
    TNG_50_pop = control.load_pop_file(TNG_basepath,tng_pop_file_path,[0,0,1000,1])
    TNG_50_control = control.control_samples_TNG(TNG_50_pop)
    invalid_merger_count = count_invalid_mergers_tng50(TNG_50_control)
    print(f"TNG50: Out of {len(TNG_50_control.subhalo_ids_controls)} valid controls, {invalid_merger_count} have no gas particles but nonzero BH accretion rate")

    invalid_control_count = count_invalid_controls_tng50(TNG_50_control)
    print(f"TNG50: Out of {len(TNG_50_control.subhalo_ids_controls)} valid controls, {invalid_control_count} have no gas particles but nonzero BH accretion rate")

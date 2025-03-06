import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt
import h5py
import sys


MSOL = 1.988409870698051e+33

def generate_population(basePath, snap_population, snaps_galaxy_mergers,subhaloidxs_galaxy_mergers, minN_values):

    #basePath is the path to the simulation directory
    #snap_population is the array of snapshots to consider. The minimum value will be the snapshot where the max redshift for gaalxy mergers is reached
    #snaps_galaxy_mergers is the array of snapshots where galaxy mergers occur
    #subhaloidxs_galaxy_mergers is the array of subhalo indices where galaxy mergers occur
    #minN_values is the array of minimum values for the number of particles in each component (DM, gas, stars, BH)

    redshifts = np.array([il.groupcat.loadHeader(basePath, snap)['Redshift'].item() for snap in snap_population])
    
    merging_population = {
        "snap": np.array([], dtype=int),
        "z": np.array([], dtype=float),
        "subhalo_ids": np.array([], dtype=int),
        "Mstar": np.array([], dtype=float),
        "Mgas": np.array([], dtype=float),
        "MBH": np.array([], dtype=float),
        "Mdot": np.array([], dtype=float),
        "SFR": np.array([], dtype=float)
    }
    
    non_merging_population = {
        "snap": np.array([], dtype=int),
        "z": np.array([], dtype=float),
        "subhalo_ids": np.array([], dtype=int),
        "Mstar": np.array([], dtype=float),
        "Mgas": np.array([], dtype=float),
        "MBH": np.array([], dtype=float),
        "Mdot": np.array([], dtype=float),
        "SFR": np.array([], dtype=float)
    }

    # Iterate over all snapshots
 
    for i,snap in enumerate(snap_population):
        subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=['SubhaloLenType', 'SubhaloMassType', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR'])
        Ngas = subhalos['SubhaloLenType'][:, 0]
        Ndm = subhalos['SubhaloLenType'][:, 1]
        Nstar = subhalos['SubhaloLenType'][:, 4]
        Nbh = subhalos['SubhaloLenType'][:, 5]
        subhalo_ids = np.arange(len(Ngas))
        subhalo_ids = subhalo_ids[(Ngas >= minN_values[1]) & (Ndm >= minN_values[0]) & (Nstar >= minN_values[2]) & (Nbh >= 0)]

        subhalo_ids_merging = subhaloidxs_galaxy_mergers[np.where(snaps_galaxy_mergers == snap)]
        subhalo_ids_non_merging = np.setdiff1d(subhalo_ids, subhalo_ids_merging)

        # if len(subhalo_ids_merging) != 0:
        #     merging_population["snap"] = np.append(merging_population["snap"], snap)
        #     merging_population["z"] = np.append(merging_population["z"], redshifts[i])
       
        merging_population["snap"] = np.append(merging_population["snap"], snap*np.ones(len(subhalo_ids_merging)))
        merging_population["z"] = np.append(merging_population["z"], redshifts[i]*np.ones(len(subhalo_ids_merging)))  
        merging_population["subhalo_ids"] = np.concatenate((merging_population["subhalo_ids"], subhalo_ids_merging))
        merging_population["Mstar"] = np.concatenate((merging_population["Mstar"], subhalos['SubhaloMassType'][subhalo_ids_merging, 4]))
        merging_population["Mgas"] = np.concatenate((merging_population["Mgas"], subhalos['SubhaloMassType'][subhalo_ids_merging, 0]))
        merging_population["MBH"] = np.concatenate((merging_population["MBH"], subhalos['SubhaloBHMass'][subhalo_ids_merging]))
        merging_population["Mdot"] = np.concatenate((merging_population["Mdot"], subhalos['SubhaloBHMdot'][subhalo_ids_merging]))
        merging_population["SFR"] = np.concatenate((merging_population["SFR"], subhalos['SubhaloSFR'][subhalo_ids_merging]))

        non_merging_population["snap"] = np.append(non_merging_population["snap"], snap*np.ones(len(subhalo_ids_non_merging)))
        non_merging_population["z"] = np.append(non_merging_population["z"], redshifts[i]*np.ones(len(subhalo_ids_non_merging)))
        non_merging_population["subhalo_ids"] = np.concatenate((non_merging_population["subhalo_ids"], subhalo_ids_non_merging))
        non_merging_population["Mstar"] = np.concatenate((non_merging_population["Mstar"], subhalos['SubhaloMassType'][subhalo_ids_non_merging, 4]))
        non_merging_population["Mgas"] = np.concatenate((non_merging_population["Mgas"], subhalos['SubhaloMassType'][subhalo_ids_non_merging, 0]))
        non_merging_population["MBH"] = np.concatenate((non_merging_population["MBH"], subhalos['SubhaloBHMass'][subhalo_ids_non_merging]))
        non_merging_population["Mdot"] = np.concatenate((non_merging_population["Mdot"], subhalos['SubhaloBHMdot'][subhalo_ids_non_merging]))
        non_merging_population["SFR"] = np.concatenate((non_merging_population["SFR"], subhalos['SubhaloSFR'][subhalo_ids_non_merging]))

    merging_population['Mstar'] = merging_population['Mstar']*1e10/h
    merging_population['Mgas'] = merging_population['Mgas']*1e10/h
    merging_population['MBH'] = merging_population['MBH']*1e10/h
    merging_population['Mdot'] = merging_population['Mdot']*(1e10/h)/(0.978*1e9/h)

    non_merging_population['Mstar'] = non_merging_population['Mstar']*1e10/h
    non_merging_population['Mgas'] = non_merging_population['Mgas']*1e10/h
    non_merging_population['MBH'] = non_merging_population['MBH']*1e10/h
    non_merging_population['Mdot'] = non_merging_population['Mdot']*(1e10/h)/(0.978*1e9/h)

    return merging_population, non_merging_population

def apply_redshift_cuts(population, snapshot_to_redshift, z_min, z_max):
    # Get the indices of snapshots within the redshift range
    valid_snapshots = [snap for snap, z in snapshot_to_redshift.items() if z_min < z < z_max]
    valid_indices = np.isin(population["snap"], valid_snapshots)
    
    # Filter the population based on the valid indices
    filtered_population = {key: value[valid_indices] for key, value in population.items()}
    return filtered_population

def write_population_to_file(filepath, basePath, snap_population, snaps_galaxy_mergers, subhaloidxs_galaxy_mergers, minN_values):
    merging_population, non_merging_population = generate_population(basePath, snap_population, snaps_galaxy_mergers, subhaloidxs_galaxy_mergers, minN_values)
    outfilename = filepath + f"/population_sort_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5"
    with h5py.File(outfilename, 'w') as f:
            merging_group = f.create_group('merging_population')
            for key, value in merging_population.items():
                merging_group.create_dataset(key, data=value)
            
            non_merging_group = f.create_group('non_merging_population')
            for key, value in non_merging_population.items():
                non_merging_group.create_dataset(key, data=value)

    print(f"Population saved to {outfilename}")

    return None

if __name__ == "__main__":

    basePath = sys.argv[1]
    merger_file_path = sys.argv[2]
    minN_values = np.array([int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])])

    merger_file_1bh = merger_file_path + f'/galaxy-mergers_TNG50-1_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5'
    
    fmergers = h5py.File(merger_file_1bh, 'r')
    subhaloidxs_galaxy_mergers = fmergers["shids_subf"][:,2]
    snaps_galaxy_mergers = fmergers['snaps'][:,2]
    z_galaxy_mergers = 1/(fmergers['time'][:][:,2])-1
    h = fmergers.attrs['HubbleParam']

    snap_list = np.arange(np.min(snaps_galaxy_mergers),100)
    #the minimum value will be the snapshot where the max redshift for gaalxy mergers is reached
    
    population_sort_file_name = merger_file_path
    write_population_to_file(population_sort_file_name, basePath, snap_list, snaps_galaxy_mergers, subhaloidxs_galaxy_mergers, minN_values)

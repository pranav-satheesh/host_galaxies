import numpy as np
import h5py
import sys
from tqdm import tqdm
import os
from joblib import Parallel, delayed
tex_path = '/apps/texlive/2023/bin/x86_64-linux/'
os.environ['PATH'] += os.pathsep + tex_path
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

sys.path.append('../BH_dynamics_analysis')
sys.path.append('/home/pranavsatheesh/arepo_package/')
import arepo_package as arepo
import BRAHMA_python as il_brahma
import illustris_python as il
from astropy.cosmology import Planck15

sys.path.append('../py_files/')

import control_sample as control
import host_galaxy_enhancement_plots as hostplot

sys.path.append('/home/pranavsatheesh/host_galaxies/py_files/')
import merger_descendants as md


def z_to_tlookback(z):
    """Convert redshift to lookback time in Gyr using Planck15 cosmology."""
    return Planck15.lookback_time(z).value  # returns time in Gyr


def find_deltaT(z1, z2):
    """Calculate the difference in lookback time between two redshifts."""
    t1 = z_to_tlookback(z1)
    t2 = z_to_tlookback(z2)
    return t1 - t2  # returns time difference in Gyr


def find_matches_bw_descendants_and_control_samples(descendant_file, pop_control):
    """Find indices where descendant file post-merger hosts."""
    store_matches = []
    merger_indices = []
    pm_snap = []
    pm_subhaloid = []
    for i in range(len(pop_control.valid_merger_indices)):
        post_merger_snap = pop_control.snap_merging_pop[i]
        post_merger_id = pop_control.subhalo_ids_mergers[i]
        
        match_idx = np.argwhere(
            (descendant_file['pm_snap'][:] == post_merger_snap) &
            (descendant_file['pm_subfind'][:] == post_merger_id)
        )
        if match_idx.size > 0:
            store_matches.append(match_idx[0][0])
            merger_indices.append(i)
            pm_snap.append(post_merger_snap)
            pm_subhaloid.append(post_merger_id)

    return np.array(store_matches), np.array(merger_indices), np.array(pm_snap), np.array(pm_subhaloid)


def get_scale_factors(basePath, filename="output_scale_factors.txt"):
    """Load scale factors and compute redshifts."""
    path = basePath.split('/output')[0]
    f = open(path + "/" + filename, 'r')
    snaptimes = np.array([float(line) for line in f.readlines()])
    f.close()
    return snaptimes


def find_most_massive_BHAR(snap, subfind_id, brahma_redshifts, basePath, h, key="BRAHMA"):
    """Find the most massive BH and its Mdot in a subhalo."""
    if key == "BRAHMA":
        redshift = brahma_redshifts[snap]
        MBH_masses_in_subhalo = arepo.get_particle_property_within_postprocessed_groups(
            basePath, particle_property=['BH_Mass'], p_type=5,
            desired_redshift=redshift, subhalo_index=subfind_id, group_type='subhalo'
        )
        Mdot_in_subhalo = arepo.get_particle_property_within_postprocessed_groups(
            basePath, particle_property=['BH_Mdot'], p_type=5,
            desired_redshift=redshift, subhalo_index=subfind_id, group_type='subhalo'
        )
        if len(MBH_masses_in_subhalo[0]) == 0:
            return 0, 0
        MBH_most_massive_BH = np.max(MBH_masses_in_subhalo[0])
        index_of_most_massive_BH = np.argmax(MBH_masses_in_subhalo[0])
        return MBH_most_massive_BH, Mdot_in_subhalo[0][index_of_most_massive_BH]
    else:
        print("TNG doesn't require this function")
        return None, None


def process_single_chain(match_idx, descendant_file, redshifts, basePath, h, sim_type, fields, subhalos_cache=None, brahma_redshifts=None):
    """Process a single descendant chain independently."""
    descendant_chain_length = len(descendant_file['descendants']['snaps'][match_idx])
    snap_0 = int(descendant_file['descendants']['snaps'][match_idx][0])
    redshift_0 = redshifts[snap_0]

    sSFR_descendant_chain = []
    sBHAR_descendant_chain = []
    time_since_merger_chain = []

    for i in range(descendant_chain_length):
        snap_i = int(descendant_file['descendants']['snaps'][match_idx][i])
        subhalo_id_i = int(descendant_file['descendants']['subfind_ids'][match_idx][i])
        redshift_i = redshifts[snap_i]
        time_since_merger = find_deltaT(redshift_0, redshift_i)
        time_since_merger_chain.append(time_since_merger)

        if time_since_merger > 1.5:  # Exit if more than 1.5 Gyr since merger
            break

        if sim_type == 'TNG':
            # Use pre-loaded subhalos from cache
            subhalos = subhalos_cache[snap_i]
            SFR_i = subhalos['SubhaloSFR'][subhalo_id_i]
            Mstar_i = subhalos['SubhaloMassType'][subhalo_id_i, 4] * 1e10 / h
            sSFR_i = SFR_i / Mstar_i if Mstar_i > 0 else 0
            sSFR_descendant_chain.append(sSFR_i)

            Mdot_i = subhalos['SubhaloBHMdot'][subhalo_id_i] * 1e10 / h / (0.978e9 / h)  # convert to Msun/yr
            MBH_i = subhalos['SubhaloBHMass'][subhalo_id_i] * 1e10 / h  # convert to Msun
            sBHAR_i = Mdot_i / MBH_i if MBH_i > 0 else 0
            sBHAR_descendant_chain.append(sBHAR_i)

        elif sim_type == 'BRAHMA':
            # Use pre-loaded subhalos from cache
            subhalos = subhalos_cache[snap_i]
            SFR_i = subhalos['SubhaloSFR'][subhalo_id_i]
            Mstar_i = subhalos['SubhaloMassType'][subhalo_id_i, 4] * 1e10 / h
            sSFR_i = SFR_i / Mstar_i if Mstar_i > 0 else 0
            sSFR_descendant_chain.append(sSFR_i)

            MBH_most_massive_BH, Mdot_most_massive_BH = find_most_massive_BHAR(snap_i, subhalo_id_i, brahma_redshifts, basePath, h)
            MBH_most_massive_BH = MBH_most_massive_BH * 1e10 / h  # convert to Msun
            Mdot_most_massive_BH = Mdot_most_massive_BH * 1e10 / h / (0.978e9 / h)  # convert to Msun/yr
            sBHAR_i = Mdot_most_massive_BH / MBH_most_massive_BH if MBH_most_massive_BH > 0 else 0
            sBHAR_descendant_chain.append(sBHAR_i)

    # Convert lists to numpy arrays
    return (np.array(sSFR_descendant_chain), 
            np.array(sBHAR_descendant_chain), 
            np.array(time_since_merger_chain), 
            len(sSFR_descendant_chain))



if __name__ == "__main__":
    # load the files
    basePath = sys.argv[1]  # e.g. '/path/to/merger/files' or '/path/to/TNG/output'
    simname = sys.argv[2]  # e.g. 'TNG50-1' or 'SM5_DFD_3_TNG'
    descendant_file_loc = sys.argv[3] 
    pop_file_loc = sys.argv[4]           
    minN_gas = int(sys.argv[5])       # e.g. 0
    minN_dm = int(sys.argv[6])        # e.g. 0
    minN_star = int(sys.argv[7])      # e.g. 1000
    minN_bh = int(sys.argv[8])        # e.g. 1
    output_dir = sys.argv[9]         # e.g. '/path/to/output/'

    minNvalues = [minN_gas, minN_dm, minN_star, minN_bh]

    if simname == 'TNG50-1':
        simPath = basePath
    else:
        simPath = os.path.join(basePath, simname)
    
    # Determine simulation type
    sim_type = 'TNG' if 'TNG50-1' in simname else 'BRAHMA'

    # Descendant file name
    descendant_file_name = f'merger_descendants_{simname}.hdf5'
    descendant_file_path = os.path.join(descendant_file_loc, descendant_file_name)
    descendant_file = h5py.File(descendant_file_path, 'r')

    #loading the population file
    pop = control.load_pop_file(simPath, pop_file_loc, minNvalues)

    if sim_type == 'TNG':
        #loading the control sample for TNG
        pop_control = control.control_samples_TNG(pop,max_Mstar_tolerance=0.15,max_z_tolerance=0.1)
        # Load TNG redshifts
        TNG_scalefactors = get_scale_factors(basePath)
        redshifts = 1 / TNG_scalefactors - 1
        fields = ['SubhaloMassType', 'SubhaloMass', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR']
        h = 0.6774
    elif sim_type == 'BRAHMA':
        #loading the control sample for BRAHMA
        if simname == 'SM5_LW10_LOWSPIN_RICH_TNG':
            brahma_control = control.control_sample_brahma(pop,max_Mstar_tolerance=0.6,max_z_tolerance=0.2)
        else:
            brahma_control = control.control_sample_brahma(pop,max_Mstar_tolerance=0.15,max_z_tolerance=0.2)
        pop_control = control.control_sample_brahma(pop)
        # Load BRAHMA redshifts
        brahma_snapshots, redshifts = arepo.get_snapshot_redshift_correspondence(basePath)
        fields = ['SubhaloMassType', 'SubhaloMass', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR']
        h = 0.6774

    match_ids, merger_indices, pm_snap, pm_subhaloid = find_matches_bw_descendants_and_control_samples(descendant_file, pop_control)
    print(f"Found {len(match_ids)} matches between descendant file and control post-merger sample")

    # Extract all unique snapshots needed from descendant file
    print("Extracting unique snapshots from descendant file...")
    unique_snaps = set()
    for match_idx in match_ids:
        snaps = descendant_file['descendants']['snaps'][match_idx]
        for snap in snaps:
            unique_snaps.add(int(snap))
    unique_snaps = sorted(list(unique_snaps))
    print(f"Found {len(unique_snaps)} unique snapshots to pre-load")

    # Pre-load all subhalos for the unique snapshots
    print("Pre-loading subhalos for all snapshots...")
    subhalos_cache = {}
    for snap in unique_snaps:
        if sim_type == 'TNG':
            subhalos_cache[snap] = il.groupcat.loadSubhalos(basePath, snap, fields=fields)
        elif sim_type == 'BRAHMA':
            subhalos_cache[snap] = il_brahma.groupcat.loadSubhalos_postprocessed(basePath, snap, fields)
    print(f"Loaded subhalos for {len(subhalos_cache)} snapshots")

    # Process chains in parallel
    print("Processing descendant chains in parallel...")
    results = Parallel(n_jobs=4, backend='threading', verbose=10)(
        delayed(process_single_chain)(
            match_idx, descendant_file, redshifts, basePath, h, sim_type, fields,
            subhalos_cache=subhalos_cache, brahma_redshifts=redshifts
        ) for match_idx in match_ids
    )
    
    # Unpack results
    all_sSFR_chains = [r[0] for r in results]
    all_sBHAR_chains = [r[1] for r in results]
    all_time_chains = [r[2] for r in results]
    all_chain_lengths = [r[3] for r in results]

    # Save results to HDF5 file
    output_filename = os.path.join(output_dir, f'sSFR_sBHAR_evolution_{simname}.hdf5')
    with h5py.File(output_filename, 'w') as f:
        # Save match indices
        f.create_dataset('match_ids', data=match_ids, dtype=np.int32)
        f.create_dataset('merger_indices', data=merger_indices, dtype=np.int32)
        f.create_dataset('post_merger_snaps', data=pm_snap, dtype=np.int32)
        f.create_dataset('post_merger_subhalo_ids', data=pm_subhaloid, dtype=np.int32)
        #this one correspond to the indices in the control pop file for the merger host
        # Create groups for each match
        grp = f.create_group('chains')
        
        # Store as variable-length datasets
        vlen_float = h5py.vlen_dtype(np.float64)
        ds_sSFR = grp.create_dataset('sSFR', (len(match_ids),), dtype=vlen_float)
        ds_sBHAR = grp.create_dataset('sBHAR', (len(match_ids),), dtype=vlen_float)
        ds_time = grp.create_dataset('time_since_merger', (len(match_ids),), dtype=vlen_float)
        
        for i in range(len(match_ids)):
            ds_sSFR[i] = all_sSFR_chains[i]
            ds_sBHAR[i] = all_sBHAR_chains[i]
            ds_time[i] = all_time_chains[i]
        
        # Save metadata
        grp.attrs['simname'] = simname
        grp.attrs['sim_type'] = sim_type
        grp.attrs['num_chains'] = len(match_ids)
    
    print(f"Results saved to {output_filename}")
    descendant_file.close()

import os
import sys
from typing import Dict, Optional
import numpy as np
import h5py



def load_merger_file(merger_file_folder,simName,minN_values):

    if 'TNG50' in simName:
        merger_file_name = f'galaxy-mergers_{simName}_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5'
    else:
        merger_file_name = f'galaxy-mergers_brahma_{simName}_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5'

    merger_file_path = os.path.join(merger_file_folder, merger_file_name)
    
    return h5py.File(merger_file_path, 'r')


def get_merger_descendants_BRAHMA(
    full_tree: Dict[str, np.ndarray],
    fp_subf: int,
    fp_snap: int,
    verbose: bool = False):
    """
    Trace descendants for a single merger using BRAHMA full tree.
    
    Args:
        full_tree: Dictionary containing BRAHMA tree data
        fp_subf: SubfindID of first progenitor
        fp_snap: Snapshot number of merger
        verbose: Print debug information
        
    Returns:
        Dictionary with keys: 'subfind_ids', 'snaps', 'next_prog_key'
        Returns None if merger not found in tree
    """
    merger_descendants = {
        'subfind_ids': np.array([]),
        'snaps': np.array([]),
        'next_prog_key': np.array([]),
        'next_prog_snap': np.array([]),
        'next_prog_subfind': np.array([])
    }
    
    # Find tree index for the first progenitor
    try:
        tree_index = np.where(
            (full_tree['SnapNum'] == fp_snap) & (full_tree['SubfindID'] == fp_subf)
        )[0][0]
    except IndexError:
        if verbose:
            print(f"Could not find merger at snap={fp_snap}, subfind={fp_subf}")
        return None
    
    rootID = full_tree['SubhaloID'][tree_index]
    rootSubfindID = full_tree['SubfindID'][tree_index]
    rootSnap = full_tree['SnapNum'][tree_index]
    descID = full_tree['DescendantID'][tree_index]
    
    # Add the root (first progenitor)
    merger_descendants['subfind_ids'] = np.append(merger_descendants['subfind_ids'], rootSubfindID)
    merger_descendants['snaps'] = np.append(merger_descendants['snaps'], rootSnap)
    merger_descendants['next_prog_key'] = np.append(merger_descendants['next_prog_key'], 1)
    
    # Walk forward through descendants
    while descID != -1:
        tree_index = tree_index + (descID - rootID)
        
        # Bounds check
        if tree_index < 0 or tree_index >= len(full_tree['SubhaloID']):
            break
        
        descSnap = full_tree['SnapNum'][tree_index]
        descSubfind = full_tree['SubfindID'][tree_index]
        
        merger_descendants['subfind_ids'] = np.append(merger_descendants['subfind_ids'], descSubfind)
        merger_descendants['snaps'] = np.append(merger_descendants['snaps'], descSnap)
        
        rootID = full_tree['SubhaloID'][tree_index]
        descID = full_tree['DescendantID'][tree_index]
        npID = full_tree['NextProgenitorID'][tree_index]
        npSubfind = full_tree['SubfindID'][tree_index + npID - rootID] if npID != -1 else -1
        npSnap = full_tree['SnapNum'][tree_index + npID - rootID] if npID != -1 else -1       
        
        merger_descendants['next_prog_snap'] = np.append(merger_descendants['next_prog_snap'], npSnap)
        merger_descendants['next_prog_subfind'] = np.append(merger_descendants['next_prog_subfind'], npSubfind)

    
    return merger_descendants


def get_merger_descendants_TNG(
    basePath: str,
    fp_subf: int,
    fp_snap: int,
    verbose: bool = False,
    max_chain_length: int = 10):
    """
    Trace descendants for a single merger using TNG SubLink tree.
    
    Args:
        basePath: Path to TNG simulation output directory
        fp_subf: SubfindID of first progenitor
        fp_snap: Snapshot number of merger
        verbose: Print debug information
        max_chain_length: Maximum number of snapshots to include in chain
        
    Returns:
        Dictionary with keys: 'subfind_ids', 'snaps', 'next_prog_key'
        Returns None if tree cannot be loaded
    """
    try:
        import illustris_python as il
    except ImportError:
        raise ImportError("illustris_python package not found. Install with: pip install illustris_python")
    
    merger_descendants = {
        'subfind_ids': np.array([]),
        'snaps': np.array([]),
        'next_prog_snap': np.array([]),
        'next_prog_subfind': np.array([])
    }
    
    try:
        if verbose:
            print(f"  Attempting to load tree for snap={fp_snap}, subfind={fp_subf}...")
        
        # Load the SubLink tree for the first progenitor
        # onlyMDB=True gets the Main Descendant Branch (forward in time)
        tree = il.sublink.loadTree(
            basePath, fp_snap, fp_subf,
            fields=['SubhaloID', 'SubfindID', 'SnapNum', 'DescendantID', 'NextProgenitorID'],
            onlyMDB=True
        )
    except Exception as e:
        if verbose:
            print(f"  Error loading tree for snap={fp_snap}, subfind={fp_subf}: {type(e).__name__}: {e}")
        return None
    
    # With onlyMDB=True, tree[0] is the most recent (snap=fp_snap) 
    # and tree[-1] is the most ancient (snap=0)
    # We iterate forward to trace descendants through time
    
    n_entries = len(tree['SubhaloID'])
    limit = min(n_entries, max_chain_length)
    
    for idx_rev in range(limit):
        subfind = tree['SubfindID'][n_entries - 1 - idx_rev]
        snap = tree['SnapNum'][n_entries - 1 - idx_rev]
        
        merger_descendants['subfind_ids'] = np.append(merger_descendants['subfind_ids'], subfind)
        merger_descendants['snaps'] = np.append(merger_descendants['snaps'], snap)
        
        # Check if there's a next progenitor (indicates multiple progenitors at this snapshot)
        npID = tree['NextProgenitorID'][n_entries - 1 - idx_rev]
        if npID != -1:
            npSnap = tree['SnapNum'][n_entries - 1 - idx_rev + npID] if (n_entries - 1 - idx_rev + npID) < n_entries else -1
            npSubfind = tree['SubfindID'][n_entries - 1 - idx_rev + npID] if (n_entries - 1 - idx_rev + npID) < n_entries else -1
            merger_descendants['next_prog_snap'] = np.append(merger_descendants['next_prog_snap'], npSnap)
            merger_descendants['next_prog_subfind'] = np.append(merger_descendants['next_prog_subfind'], npSubfind)
        else:
            merger_descendants['next_prog_snap'] = np.append(merger_descendants['next_prog_snap'], -1)
            merger_descendants['next_prog_subfind'] = np.append(merger_descendants['next_prog_subfind'], -1)
    
    return merger_descendants


# ============================================================================
# PROCESS ALL MERGERS
# ============================================================================

if __name__ == "__main__":

    basePath = sys.argv[1]  # e.g. '/path/to/merger/files' or '/path/to/TNG/output'
    simname = sys.argv[2]  # e.g. 'TNG50' or 'SM5_DFD_3_TNG'
    merger_file_folder = sys.argv[3]            
    minN_gas = int(sys.argv[4])       # e.g. 0
    minN_dm = int(sys.argv[5])        # e.g. 0
    minN_star = int(sys.argv[6])      # e.g. 1000
    minN_bh = int(sys.argv[7])        # e.g. 1
    output_dir = sys.argv[8]         # e.g. '/path/to/output/'

    minNvalues = [minN_gas, minN_dm, minN_star, minN_bh]
    
    # Determine simulation type
    sim_type = 'TNG' if 'TNG50' in simname else 'BRAHMA'

    # Load merger file
    mrgrfile = load_merger_file(merger_file_folder, simname, minNvalues)
    shids_subf = mrgrfile['shids_subf'][:]
    snaps = mrgrfile['snaps'][:]
    
    # Load full tree for BRAHMA if needed
    full_tree = None
    if sim_type == 'BRAHMA':
        print(f"Loading BRAHMA tree...")
        tree = h5py.File(basePath + 'postprocessing/tree_extended.hdf5', 'r')
        full_tree = {key: tree[key][:] for key in tree.keys()}
        tree.close()
        print(f"Loaded {len(full_tree)} fields from tree")

    nmergers = len(shids_subf)
    print(f"Processing {nmergers} mergers...")
    merger_idx  = np.empty(nmergers, dtype=np.int32)
    fp_subfind  = np.empty(nmergers, dtype=np.int32)
    fp_snap     = np.empty(nmergers, dtype=np.int32)
    pm_subfind  = np.empty(nmergers, dtype=np.int32)
    pm_snap     = np.empty(nmergers, dtype=np.int32)

    desc_list   = []  # still a list since variable-length
    valid_mask = []

    for idx in range(nmergers):
        fp_subf = int(shids_subf[idx][0])
        fp_snap_val = int(snaps[idx][0])
        pm_subf = int(shids_subf[idx][2])
        pm_snap_val = int(snaps[idx][2])

        if sim_type == 'TNG':
            #print("Using TNG SubLink tree loading method...")
            descendants = get_merger_descendants_TNG(basePath, int(fp_subf), int(fp_snap_val), verbose=False)
        else:
            #print("Using BRAHMA tree loading method...")
            descendants = get_merger_descendants_BRAHMA(full_tree, int(fp_subf), int(fp_snap_val), verbose=False)

        if descendants is not None:
            merger_idx[idx] = idx
            fp_subfind[idx] = fp_subf
            fp_snap[idx] = fp_snap_val

            pm_subfind[idx] = pm_subf
            pm_snap[idx] = pm_snap_val

            desc_list.append(descendants)
            valid_mask.append(True)
            
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(shids_subf)} mergers")

    # Trim to valid entries
    valid_mask = np.array(valid_mask)
    merger_idx = merger_idx[valid_mask]
    fp_subfind = fp_subfind[valid_mask]
    fp_snap    = fp_snap[valid_mask]
    pm_subfind = pm_subfind[valid_mask]
    pm_snap    = pm_snap[valid_mask]

# ============================================================================
# SAVE TO HDF5
# ============================================================================


    output_file = os.path.join(output_dir, f'merger_descendants_{simname}.hdf5')
    print(f"\nSaving to: {output_file}")

    vlen_int = h5py.vlen_dtype(np.int32)

    with h5py.File(output_file, 'w') as f:
        #scalar fields 
        f.create_dataset('merger_idx', data=merger_idx, dtype=np.int32)
        f.create_dataset('fp_subfind', data=fp_subfind, dtype=np.int32)
        f.create_dataset('fp_snap', data=fp_snap, dtype=np.int32)
        f.create_dataset('pm_subfind', data=pm_subfind, dtype=np.int32)
        f.create_dataset('pm_snap', data=pm_snap, dtype=np.int32)

        # variable-length descendant chains
        desc_group = f.create_group('descendants')
        n = len(desc_list)

        ds_ids = desc_group.create_dataset('subfind_ids', (n,), dtype=vlen_int)
        ds_snaps = desc_group.create_dataset('snaps', (n,), dtype=vlen_int)
        ds_next_prog_snap = desc_group.create_dataset('next_prog_snap', (n,), dtype=vlen_int)
        ds_next_prog_subfind = desc_group.create_dataset('next_prog_subfind', (n,), dtype=vlen_int)


        for i, desc in enumerate(desc_list):
            ds_ids[i] = desc['subfind_ids']
            ds_snaps[i] = desc['snaps']
            ds_next_prog_snap[i] = desc['next_prog_snap']
            ds_next_prog_subfind[i] = desc['next_prog_subfind']

    # # Filter valid mergers
    # valid_mergers = [entry for entry in all_merger_descendants if entry['descendants'] is not None]
    # print(f"Saving {len(valid_mergers)}/{len(all_merger_descendants)} mergers...")

    # # Write HDF5 with one group per merger to keep individual descendant chains.
    # total_data_points = 0
    # with h5py.File(output_file, 'w') as f:
    #     mergers_group = f.create_group('mergers')

    #     for entry in valid_mergers:
    #         merger_idx = int(entry['merger_idx'])
    #         fp_subfind = int(entry['fp_subfind'])
    #         fp_snap = int(entry['fp_snap'])
    #         descendants = entry['descendants']

    #         subfind_ids = np.asarray(descendants['subfind_ids'], dtype=np.int32)
    #         desc_snaps = np.asarray(descendants['snaps'], dtype=np.int32)
    #         next_prog_key = np.asarray(descendants['next_prog_key'], dtype=np.int32)
    #         chain_length = len(subfind_ids)
    #         total_data_points += chain_length

    #         merger_group = mergers_group.create_group(f'merger_{merger_idx:06d}')
    #         merger_group.create_dataset('subfind_ids', data=subfind_ids, compression='gzip')
    #         merger_group.create_dataset('snaps', data=desc_snaps, compression='gzip')
    #         merger_group.create_dataset('next_prog_key', data=next_prog_key, compression='gzip')

    #         merger_group.attrs['merger_idx'] = merger_idx
    #         merger_group.attrs['fp_subfind'] = fp_subfind
    #         merger_group.attrs['fp_snap'] = fp_snap
    #         merger_group.attrs['chain_length'] = chain_length

    #     f.attrs['num_mergers'] = len(valid_mergers)
    #     f.attrs['total_data_points'] = total_data_points
    #     f.attrs['description'] = f'Descendant chains for all mergers in {simname} simulation'
    #     f.attrs['note'] = "Each merger is stored under /mergers/merger_XXXXXX with its own datasets and attributes"

    # print(f"✓ Successfully saved {len(valid_mergers)} mergers to {output_file}")
    # print(f"  Total data points: {total_data_points}")

    # ============================================================================
    # VERIFY

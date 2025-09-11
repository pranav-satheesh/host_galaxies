import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import illustris_python_mod as il
import sys
import os
sys.path.append('../BH_dynamics_analysis')
sys.path.append('/home/pranavsatheesh/arepo_package/')
import arepo_package as arepo

from tqdm import tqdm

def get_sublink_info_for_merger_event(merger_index):
    o,remnant_snap=arepo.desired_redshift_to_output_redshift(sim_path,remnant_redshift[merger_index])
    mask_snap = SnapNum == remnant_snap
    mask_subfindID = SubfindID == remnant_SubhaloID[merger_index]
    mask = mask_snap & mask_subfindID

    try:
        fpID = FirstProgenitorID[mask][0]
        #The first progenitor
    except:
        # If the mask does not return any results, it means the subhalo is not found in this snapshot
        return -1, -1, -1, -1, remnant_snap

    if fpID == -1:
        #First progenitor ID is -1, which means this remnant subhalo and has no progenitors
        return -1, -1, -1, -1, remnant_snap
    
    else:
        #The other progenitor is found by matching the subhalo ID to the fpID of this subhalo and extracting the nextprogenitor ID
        npID = NextProgenitorID[np.where(SubhaloID==fpID)[0][0]]
        dID = SubhaloID[mask][0]
        subfind_ID = SubfindID[mask][0]

        return subfind_ID, fpID, npID, dID, remnant_snap
    
    


if __name__ == "__main__":
    sim_path = sys.argv[1]
    run_name = sys.argv[2]
    save_path = sys.argv[3]
    
    merger_properties=np.load(sim_path + '/merger_statistics_subhalo_remnants.npy',allow_pickle=True)[0]
    
    merger_type=np.array(merger_properties['merger_type'])
    merger_redshift=np.array(merger_properties['merger_redshift'])
    BH_ID1=np.array(merger_properties['BH_ID1'])
    BH_ID2=np.array(merger_properties['BH_ID2'])
    BH_Mass1=np.array(merger_properties['BH_Mass1'])
    BH_Mass2=np.array(merger_properties['BH_Mass2'])
    remnant_redshift=np.array(merger_properties['remnant_redshift'])
    remnant_SubhaloStellarMass=np.array(merger_properties['remnant_SubhaloStellarMass'])
    remnant_SubhaloDarkMatterMass=np.array(merger_properties['remnant_SubhaloDarkMatterMass'])
    remnant_SubhaloHalfMassRadii=np.array(merger_properties['remnant_SubhaloHalfMassRadii'])
    remnant_central_or_satellite=np.array(merger_properties['remnant_central_or_satellite'])
    remnant_SubhaloID=np.array(merger_properties['remnant_SubhaloID'])
    
    merger_indices = np.arange(0,len(merger_redshift))
    
    #load the tree
    tree=h5py.File(sim_path+'/postprocessing/tree_extended.hdf5')
    SnapNum=tree.get('SnapNum')[:]
    NextProgenitorID=tree.get('NextProgenitorID')[:]
    FirstProgenitorID=tree.get('FirstProgenitorID')[:]
    SubfindID=tree.get('SubfindID')[:]
    SubhaloID = tree.get('SubhaloID')[:]
    
    SubhaloIndex_mergers=[]
    FirstProgenitorID_mergers=[]
    NextProgenitorID_mergers=[]
    
    for merger_index in tqdm(merger_indices):
        subfind_ID, fpID, npID, dID, remnant_snap = get_sublink_info_for_merger_event(merger_index)
        SubhaloIndex_mergers.append(subfind_ID)
        FirstProgenitorID_mergers.append(fpID)
        NextProgenitorID_mergers.append(npID)
        
    SubhaloIndex_mergers = np.array(SubhaloIndex_mergers)
    FirstProgenitorID_mergers = np.array(FirstProgenitorID_mergers)
    NextProgenitorID_mergers = np.array(NextProgenitorID_mergers)
    
    # save_file_name = save_path+"tree_subhalo_bh_connect_data_run_"+run_name+".npz"
    save_file_name = save_path+"tree_subhalo_bh_connect_data_run_%s.npz"%(run_name)
    
    np.savez(save_file_name,
         SubhaloIndex=SubhaloIndex_mergers,
         FirstProgenitorID=FirstProgenitorID_mergers,
         NextProgenitorID=NextProgenitorID_mergers)
    
    extract_proper = (FirstProgenitorID_mergers >= 0) & (NextProgenitorID_mergers >= 0)
    proper_merger_indices = merger_indices[extract_proper]

    extract_bad = (FirstProgenitorID_mergers >= 0) & (NextProgenitorID_mergers == -1 )
    bad_merger_indices = merger_indices[extract_bad]


    extract_worst = (FirstProgenitorID_mergers == -1) & (NextProgenitorID_mergers == -1 )
    worst_merger_indices = merger_indices[extract_worst]


    print("Number of mergers with first progenitor ID =-1 is ",len(FirstProgenitorID_mergers[extract_worst]))
    print("Number of mergers with first progenitor ID != -1 but next progenitor ID = -1 is ",len(FirstProgenitorID_mergers[extract_bad]))
    print("Number of mergers with neither = -1 is ",len(NextProgenitorID_mergers[proper_merger_indices]))





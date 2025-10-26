import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt
import h5py
import sys
from scipy.spatial import cKDTree
sys.path.append('../BH_dynamics_analysis')
sys.path.append('/home/pranavsatheesh/arepo_package/')
import arepo_package as arepo
import BRAHMA_python as il_brahma
from tqdm import tqdm

def find_tng_subhalo_statistics(tng_basepath, save_loc, minN_values):
    sub_hdr = il.groupcat.loadHeader(tng_basepath, snapNum=0)
    h = sub_hdr['HubbleParam']
    minNgas = minN_values[0]
    minNdm = minN_values[1]
    minNstar = minN_values[2]
    minNbh = minN_values[3]
    snap_list = np.arange(1,100,1)
    #should change this away from hard coded values later for an arbitary sim
    z_list = np.array([il.groupcat.loadHeader(tng_basepath, snap)['Redshift'].item() 
                                  for snap in snap_list])
    fields=['SubhaloLenType', 'SubhaloMassType','SubhaloMass', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloHalfmassRadType','SubhaloMassInHalfRadType','SubhaloMassInRadType']

    all_subhalos_data={
    "snap" : [],
    "z" : [],
    "SubhaloLenType": [],
    "Mstar": [],
    "MBH": [],
    "MgasinRad": [],
    "Mgastotal": [],
    "MstarinRad": [],
    "SFR": [],
    "MdotBH": [],
    "SubhaloHalfmassRadType": []
    }

    for snap in tqdm(snap_list,desc="processing each snapshots in TNG"):
        subhalos = il.groupcat.loadSubhalos(tng_basepath, snapNum=snap,fields=fields)
    sub_lentype = subhalos['SubhaloLenType']
    Ngas = sub_lentype[:,0]
    Ndm = sub_lentype[:,1]
    Nstar = sub_lentype[:,4]
    Nbh = sub_lentype[:,5]
    valid_subhalo_idx = np.where((Ngas > minNgas) & (Ndm > minNdm) & (Nstar > minNstar) & (Nbh > 0))
    all_subhalos_data["snap"].append(np.full(len(valid_subhalo_idx[0]), snap))
    all_subhalos_data["z"].append(np.full(len(valid_subhalo_idx[0]), z_list[snap-1]))
    all_subhalos_data["SubhaloLenType"].append(sub_lentype[valid_subhalo_idx])
    all_subhalos_data["Mstar"].append(subhalos['SubhaloMassType'][:,4][valid_subhalo_idx])  
    all_subhalos_data["MBH"].append(subhalos['SubhaloBHMass'][valid_subhalo_idx])
    all_subhalos_data["MgasinRad"].append(subhalos['SubhaloMassInRadType'][:,0][valid_subhalo_idx])
    all_subhalos_data["Mgastotal"].append(subhalos['SubhaloMassType'][:,0][valid_subhalo_idx])
    all_subhalos_data["MstarinRad"].append(subhalos['SubhaloMassInRadType'][:,4][valid_subhalo_idx])
    all_subhalos_data["SFR"].append(subhalos['SubhaloSFR'][valid_subhalo_idx])
    all_subhalos_data["MdotBH"].append(subhalos['SubhaloBHMdot'][valid_subhalo_idx])
    all_subhalos_data["SubhaloHalfmassRadType"].append(subhalos['SubhaloHalfmassRadType'][valid_subhalo_idx])

    update_units(all_subhalos_data,h)
    save_file(all_subhalos_data, save_loc, "TNG-50")

def find_brahma_subhalo_statistics(brahma_basepath, save_loc, minN_values):
    brahma_simName = brahma_basepath.split('/')[-2]
    sub_hdr = il_brahma.groupcat.loadHeader(brahma_basepath, snapNum=0)
    h = sub_hdr['HubbleParam']
    minNgas = minN_values[0]
    minNdm = minN_values[1]
    minNstar = minN_values[2]
    minNbh = minN_values[3]
    snap_list = np.arange(1,33,1)
    z_list = np.array([il_brahma.groupcat.loadHeader(brahma_basepath, snap)['Redshift'].item() 
                                  for snap in snap_list])
    fields=['SubhaloLenType', 'SubhaloMassType','SubhaloMass', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloHalfmassRadType','SubhaloMassInHalfRadType','SubhaloMassInRadType']

    all_subhalos_data={
         "snap" : [],
    "z" : [],
    "SubhaloLenType": [],
    "Mstar": [],
    "MBH": [],
    "MgasinRad": [],
    "Mgastotal": [],
    "MstarinRad": [],
    "SFR": [],
    "MdotBH": [],
    "SubhaloHalfmassRadType": []
    }

    for snap in tqdm(snap_list,desc="processing each snapshots in BRAHMA"):
        subhalos = il_brahma.groupcat.loadSubhalos(brahma_basepath, snapNum=snap,fields=fields)
    sub_lentype = subhalos['SubhaloLenType']
    Ngas = sub_lentype[:,0]
    Ndm = sub_lentype[:,1]
    Nstar = sub_lentype[:,4]
    Nbh = sub_lentype[:,5]
    valid_subhalo_idx = np.where((Ngas > minNgas) & (Ndm > minNdm) & (Nstar > minNstar) & (Nbh > 0))
    all_subhalos_data["snap"].append(np.full(len(valid_subhalo_idx[0]), snap))
    all_subhalos_data["z"].append(np.full(len(valid_subhalo_idx[0]), z_list[snap-1]))
    all_subhalos_data["SubhaloLenType"].append(sub_lentype[valid_subhalo_idx])
    all_subhalos_data["Mstar"].append(subhalos['SubhaloMassType'][:,4][valid_subhalo_idx])  
    all_subhalos_data["MBH"].append(subhalos['SubhaloBHMass'][valid_subhalo_idx])
    all_subhalos_data["MgasinRad"].append(subhalos['SubhaloMassInRadType'][:,0][valid_subhalo_idx])
    all_subhalos_data["Mgastotal"].append(subhalos['SubhaloMassType'][:,0][valid_subhalo_idx])
    all_subhalos_data["MstarinRad"].append(subhalos['SubhaloMassInRadType'][:,4][valid_subhalo_idx])
    all_subhalos_data["SFR"].append(subhalos['SubhaloSFR'][valid_subhalo_idx])
    all_subhalos_data["MdotBH"].append(subhalos['SubhaloBHMdot'][valid_subhalo_idx])
    all_subhalos_data["SubhaloHalfmassRadType"].append(subhalos['SubhaloHalfmassRadType'][valid_subhalo_idx])     

    update_units(all_subhalos_data,h)
    save_file(all_subhalos_data, save_loc, brahma_simName)

def update_units(data_dict,h):
    
    data_dict["Mstar"] = np.concatenate(data_dict["Mstar"]) * 1e10 / h 
    data_dict["MBH"] = np.concatenate(data_dict["MBH"]) * 1e10 / h
    data_dict["MgasinRad"] = np.concatenate(data_dict["MgasinRad"]) * 1e10 / h
    data_dict["Mgastotal"] = np.concatenate(data_dict["Mgastotal"]) * 1e10 / h
    data_dict["MstarinRad"] = np.concatenate(data_dict["MstarinRad"]) * 1e10 / h
    data_dict["SFR"] = np.concatenate(data_dict["SFR"]) * 1e10 / h
    data_dict["MdotBH"] = np.concatenate(data_dict["MdotBH"]) * 1e10 * h / (0.978e9 / h)  # MSOL/yr

    return data_dict
def save_file(data_dict, save_loc, sim_key="TNG-50"):
    
    with h5py.File(save_loc + '/' + sim_key + '_subhalo_statistics.hdf5', 'w') as f:
        for key in data_dict.keys():
            f.create_dataset(key, data=data_dict[key])

    print(sim_key + " subhalo statistics saved at " + save_loc + " successfully.")

if __name__ == "__main__":

    brahma_key = sys.argv[1]
    basepath = sys.argv[2]
    save_loc = sys.argv[3]
    minN_values= np.array([int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])])
    
    if brahma_key == "brahma":
        print(brahma_key,"Processing BRAHMA simulation...")
        find_brahma_subhalo_statistics(basepath, save_loc, minN_values)
    else:
        print(brahma_key)
        find_tng_subhalo_statistics(basepath, save_loc, minN_values)    
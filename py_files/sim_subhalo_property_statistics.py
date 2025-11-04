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
    fields=['SubhaloLenType', 'SubhaloMassType','SubhaloMass', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloHalfmassRadType'
            ,'SubhaloMassInHalfRadType','SubhaloMassInRadType', 'SubhaloStellarPhotometrics']

    #SubhaloSFRinHalfRad
    # gas half mass radius
    
    all_subhalos_data={
    "snap" :  np.array([], dtype=int),
    "z" :  np.array([], dtype=float),
    #"SubhaloLenType": np.empty((0,6), dtype=int),
    "Mstar": np.array([], dtype=float),
    "MBH": np.array([], dtype=float),
    "MgasinRad": np.array([], dtype=float),
    "Mgastotal": np.array([], dtype=float),
    "MstarinRad": np.array([], dtype=float),
    "SFR": np.array([], dtype=float),
    "MdotBH": np.array([], dtype=float),
    "StellarHalfmassRadType": np.array([], dtype=float),
    "SubhaloPhotoMag": np.empty((0, 8), dtype=float)
    }

    for snap in tqdm(snap_list,desc="processing each snapshots in TNG"):
        subhalos = il.groupcat.loadSubhalos(tng_basepath, snapNum=snap,fields=fields)
        sub_lentype = subhalos['SubhaloLenType']
        Ngas = sub_lentype[:,0]
        Ndm = sub_lentype[:,1]
        Nstar = sub_lentype[:,4]
        Nbh = sub_lentype[:,5]
        valid_subhalo_idx = np.where((Ngas > minNgas) & (Ndm > minNdm) & (Nstar > minNstar) & (Nbh > 0))

        all_subhalos_data["snap"] = np.concatenate((all_subhalos_data["snap"],snap*np.ones(len(valid_subhalo_idx[0]))))
        all_subhalos_data["z"] = np.concatenate((all_subhalos_data["z"],z_list[snap-1]*np.ones(len(valid_subhalo_idx[0]))))
        #all_subhalos_data["SubhaloLenType"] = np.concatenate((all_subhalos_data["SubhaloLenType"],sub_lentype[valid_subhalo_idx]))
        all_subhalos_data["Mstar"] = np.concatenate((all_subhalos_data["Mstar"],subhalos['SubhaloMassType'][:,4][valid_subhalo_idx]))
        all_subhalos_data["MBH"] = np.concatenate((all_subhalos_data["MBH"],subhalos['SubhaloBHMass'][valid_subhalo_idx]))
        all_subhalos_data["MgasinRad"] = np.concatenate((all_subhalos_data["MgasinRad"],subhalos['SubhaloMassInRadType'][:,0][valid_subhalo_idx]))
        all_subhalos_data["Mgastotal"] = np.concatenate((all_subhalos_data["Mgastotal"],subhalos['SubhaloMassType'][:,0][valid_subhalo_idx]))
        all_subhalos_data["MstarinRad"] = np.concatenate((all_subhalos_data["MstarinRad"],subhalos['SubhaloMassInRadType'][:,4][valid_subhalo_idx]))
        all_subhalos_data["SFR"] = np.concatenate((all_subhalos_data["SFR"],subhalos['SubhaloSFR'][valid_subhalo_idx]))
        all_subhalos_data["MdotBH"] = np.concatenate((all_subhalos_data["MdotBH"],subhalos['SubhaloBHMdot'][valid_subhalo_idx]))
        all_subhalos_data["StellarHalfmassRadType"] = np.concatenate((all_subhalos_data["StellarHalfmassRadType"],subhalos['SubhaloHalfmassRadType'][:,4][valid_subhalo_idx]))
        all_subhalos_data["SubhaloPhotoMag"] = np.concatenate((all_subhalos_data["SubhaloPhotoMag"], subhalos['SubhaloStellarPhotometrics'][valid_subhalo_idx]))

    update_units(all_subhalos_data,h)
    save_file(all_subhalos_data, save_loc, "TNG50")

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
    fields=['SubhaloLenType', 'SubhaloMassType','SubhaloMass', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloHalfmassRadType','SubhaloMassInHalfRadType','SubhaloMassInRadType','SubhaloStellarPhotometrics']
    #fields - add SubhaloHalfmassRadType
    all_subhalos_data={
    "snap" :  np.array([], dtype=int),
    "z" :  np.array([], dtype=float),
    #"SubhaloLenType": np.empty((0,6), dtype=int),
    "Mstar": np.array([], dtype=float),
    "MBH": np.array([], dtype=float),
    "MgasinRad": np.array([], dtype=float),
    "Mgastotal": np.array([], dtype=float),
    "MstarinRad": np.array([], dtype=float),
    "SFR": np.array([], dtype=float),
    "MdotBH": np.array([], dtype=float),
    "StellarHalfmassRadType": np.array([], dtype=float),
    "SubhaloPhotoMag": np.empty((0, 8), dtype=float)
    }

    for snap in tqdm(snap_list,desc="processing each snapshots in BRAHMA"):
        subhalos = il_brahma.groupcat.loadSubhalos(brahma_basepath, snapNum=snap,fields=fields)
        sub_lentype = subhalos['SubhaloLenType']
        Ngas = sub_lentype[:,0]
        Ndm = sub_lentype[:,1]
        Nstar = sub_lentype[:,4]
        Nbh = sub_lentype[:,5]
        valid_subhalo_idx = np.where((Ngas > minNgas) & (Ndm > minNdm) & (Nstar > minNstar) & (Nbh > 0))
        all_subhalos_data["snap"] = np.concatenate((all_subhalos_data["snap"],snap*np.ones(len(valid_subhalo_idx[0]))))
        all_subhalos_data["z"] = np.concatenate((all_subhalos_data["z"],z_list[snap-1]*np.ones(len(valid_subhalo_idx[0]))))
        #all_subhalos_data["SubhaloLenType"] = np.concatenate((all_subhalos_data["SubhaloLenType"],sub_lentype[valid_subhalo_idx]))
        all_subhalos_data["Mstar"] = np.concatenate((all_subhalos_data["Mstar"],subhalos['SubhaloMassType'][:,4][valid_subhalo_idx]))
        all_subhalos_data["MBH"] = np.concatenate((all_subhalos_data["MBH"],subhalos['SubhaloBHMass'][valid_subhalo_idx]))
        all_subhalos_data["MgasinRad"] = np.concatenate((all_subhalos_data["MgasinRad"],subhalos['SubhaloMassInRadType'][:,0][valid_subhalo_idx]))
        all_subhalos_data["Mgastotal"] = np.concatenate((all_subhalos_data["Mgastotal"],subhalos['SubhaloMassType'][:,0][valid_subhalo_idx]))
        all_subhalos_data["MstarinRad"] = np.concatenate((all_subhalos_data["MstarinRad"],subhalos['SubhaloMassInRadType'][:,4][valid_subhalo_idx]))
        all_subhalos_data["SFR"] = np.concatenate((all_subhalos_data["SFR"],subhalos['SubhaloSFR'][valid_subhalo_idx]))
        all_subhalos_data["MdotBH"] = np.concatenate((all_subhalos_data["MdotBH"],subhalos['SubhaloBHMdot'][valid_subhalo_idx]))
        all_subhalos_data["StellarHalfmassRadType"] = np.concatenate((all_subhalos_data["StellarHalfmassRadType"],subhalos['SubhaloHalfmassRadType'][:,4][valid_subhalo_idx]))
        all_subhalos_data["SubhaloPhotoMag"] = np.concatenate((all_subhalos_data["SubhaloPhotoMag"], subhalos['SubhaloStellarPhotometrics'][valid_subhalo_idx]))


    update_units(all_subhalos_data,h)
    save_file(all_subhalos_data, save_loc, brahma_simName)

def update_units(data_dict,h):

    data_dict["Mstar"] = data_dict["Mstar"] * 1e10 / h
    data_dict["MBH"] = data_dict["MBH"] * 1e10 / h
    data_dict["MgasinRad"] = data_dict["MgasinRad"] * 1e10 / h
    data_dict["Mgastotal"] = data_dict["Mgastotal"] * 1e10 / h
    data_dict["MstarinRad"] = data_dict["MstarinRad"] * 1e10 / h
    data_dict["MdotBH"] = data_dict["MdotBH"] * 1e10 * h / (0.978e9 / h)  # MSOL/yr
    data_dict["StellarHalfmassRadType"] = data_dict["StellarHalfmassRadType"] / h  # in ckpc
    return data_dict


def save_file(data_dict, save_loc, sim_key="TNG50"):
    
    with h5py.File(save_loc + '/' + sim_key + '_subhalo_statistics.hdf5', 'w') as f:
        for key,value in data_dict.items():
            f.create_dataset(key, data=value)

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
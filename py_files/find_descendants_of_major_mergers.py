import sys
import os
sys.path.append('/home/pranavsatheesh/host_galaxies/')
import illustris_python as il
import matplotlib.pyplot as plt
import h5py
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy.cosmology import z_at_value


#merger_file_path = '/home/pranavsatheesh/host_galaxies/data/merger_files'
basePath = '/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output'

snap_list = np.arange(1,100)
redshifts = np.array([il.groupcat.loadHeader(basePath, snap)['Redshift'].item() for snap in snap_list])
scale_factors = np.array([il.groupcat.loadHeader(basePath, snap)['Time'].item() for snap in snap_list])
one_plus_z = 1.0 / scale_factors
z_values = one_plus_z - 1.0

def snap_to_z(snap):
    return one_plus_z[snap-1]-1

def find_descendants_of_mergers(merger_file_name):

    fmergers = h5py.File(merger_file_name, 'r')
    prog_mass_ratio = fmergers['ProgMassRatio_mod'][:]
    prog_mass_ratio[prog_mass_ratio>1] = 1/prog_mass_ratio[prog_mass_ratio>1]
    major_mergers_mask = prog_mass_ratio > 0.1
    total_major_mergers = np.sum(major_mergers_mask)

    merger_descendants = {'shids_subfind': [], 
                               'snap': [],
                              'shids_tree': []
                              }
    
    for i in range(total_major_mergers):
        print("Processing major merger %d of %d" % (i+1, total_major_mergers))

        #descendant of the major merger
        descendant_subfind_id = fmergers['shids_subf'][major_mergers_mask][i][2]
        descendant_snap = fmergers['snaps'][major_mergers_mask][i][2]

        #loading the corresponding tree
        tree = il.sublink.loadTree(basePath,snapNum=descendant_snap,id=descendant_subfind_id,fields=['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 'FirstProgenitorID',
              'LastProgenitorID', 'RootDescendantID', 'SubhaloLenType', 'SubhaloMassType',
              'SnapNum', 'DescendantID', 'SubfindID','FirstSubhaloInFOFGroupID'],onlyMDB=True)

        rootID = tree['SubhaloID'][-1]
        descID = tree['DescendantID'][-1]
        descSnap = tree['SnapNum'][-1]

        desc_time = cosmo.age(snap_to_z(descSnap)).value #in Gyr
        #find the redshift of descendant 2 Gyrs after

        age_of_universe = cosmo.age(0).value #in Gyr
        if desc_time + 2 < age_of_universe:
            target_descendant_z = z_at_value(cosmo.age, (desc_time+2) * u.Gyr).value
            target_descendant_snap = snap_list[np.argmin(np.abs(z_values - target_descendant_z))]
        else:
            target_descendant_snap = 99

        rootID = tree['SubhaloID'][-1]
        descID = tree['DescendantID'][-1]
        descSnap = tree['SnapNum'][-1]
        desc_subfind_id = tree['SubfindID'][-1]

        while descID != -1:
            index = - (1 + rootID-descID)
            subID = tree['SubhaloID'][index]
            descID = tree['DescendantID'][index]
            descSnap = tree['SnapNum'][index]
            desc_subfind_id = tree['SubfindID'][index]
            #find the descendant 
            if(descSnap> target_descendant_snap):
                break
            else:
                # print("Descendant ID: %d, snap: %d, subhaloid: %d"%(descID, descSnap,desc_subfind_id))
                merger_descendants['shids_subfind'].append(desc_subfind_id)
                merger_descendants['snap'].append(descSnap)
                merger_descendants['shids_tree'].append(subID)

    return merger_descendants

def save_merger_descendants(merger_descendants, merger_file_path):

    print("The total number of merger descendants found: %d" % len(merger_descendants['shids_subfind']))

    merger_descendants_file = merger_file_path + '/descendants_after_2Gyr_of_mergers.hdf5'
    with h5py.File(merger_descendants_file, 'w') as f:
        for key, value in merger_descendants.items():
            f.create_dataset(key, data=np.array(value))
    print("Merger descendants saved to %s" % merger_descendants_file)

if __name__ == "__main__":

    
    merger_file_path = sys.argv[1]
    merger_file_name = merger_file_path + '/galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-000.hdf5'
    
    merger_descendant_dict = find_descendants_of_mergers(merger_file_name)
    save_merger_descendants(merger_descendant_dict, merger_file_path)


    
    




  


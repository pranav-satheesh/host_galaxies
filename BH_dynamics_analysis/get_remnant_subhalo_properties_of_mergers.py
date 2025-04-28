import sys
sys.path.append('/home/aklantbhowmick/anaconda3/lib/python3.7/site-packages')
sys.path.append('/home/aklantbhowmick/anaconda3/lib/python3.7/site-packages/scalpy/')
#sys.path.append('/home/aklantbhowmick/anaconda3/envs/nbodykit-env/lib/python3.6/site-packages/')
import mdot_to_Lbol
import arepo_package
import scipy.interpolate
radiative_efficiency=0.2
total_conv=mdot_to_Lbol.get_conversion_factor_arepo(radiative_efficiency)
import h5py
import illustris_python as il
import numpy 
import pandas as pd
import scipy.integrate
from scipy.optimize import curve_fit
import os
from Config import *

hubble = 0.6771
from cosmology import *

DFD = 0
KIN = 0
KIN_DETAILS = 0
MERGE_SNAP = 1

def evaluate(N):
    merger_redshift=redshift_complete_sorted[N]
    pid= primary_id_sorted[N]
    sid= secondary_id_sorted[N]
    p_mass = primary_mass_sorted[N]
    s_mass = secondary_mass_sorted[N]
    temp = z_list-merger_redshift
    print(merger_redshift,temp)
#    if(merger_redshift==0):
#        redshift_snapshots_to_search = snap_list[temp<=0] #list of snapshots after the BH merger
#    else:
    redshift_snapshots_to_search = snap_list[temp<0.0001] #list of snapshots after the BH merger
    found_at_least_one_ID_in_next_snap = 0
    for snap in redshift_snapshots_to_search[0:1]:   # looking for the remnant in the immediate next snapshot
        ParticleIDs = ParticleID_dict[snap]
        BH_Mass = BH_Mass_dict[snap]
        SubhaloID = SubhaloID_dict[snap]
        SubhaloID = SubhaloID.astype(int)
        SubhaloStellarMass = SubhaloStellarMass_dict[snap]
#        print(SubhaloStellarMass)
        SubhaloDarkMatterMass = SubhaloDarkMatterMass_dict[snap]
        central_or_satellite = central_or_satellite_dict[snap]
        SubhaloHalfMassRadii = SubhaloHalfMassRadii_dict[snap]

        if (pid in ParticleIDs) | (sid in ParticleIDs): #if remnant exists in the next snapshot
            found_at_least_one_ID_in_next_snap = 1
            mask = (pid==ParticleIDs) | (sid==ParticleIDs)
            BH_Mass_remnant=(BH_Mass[mask])[0]
            SubhaloID_remnant = (SubhaloID[mask])[0]
            SubhaloStellarMass_remnant = (SubhaloStellarMass[mask])[0]
            SubhaloDarkMatterMass_remnant = (SubhaloDarkMatterMass[mask])[0]
            central_or_satellite_remnant = (central_or_satellite[mask])[0]
            SubhaloHalfMassRadii_remnant = (SubhaloHalfMassRadii[mask])[0]
        else:   # if remnant ID does not exist in the next snapshot, must have undergone another merger before that snapshot
            BH_Mass_remnant = -1
            SubhaloID_remnant = -1
            SubhaloStellarMass_remnant = -1
            SubhaloDarkMatterMass_remnant = -1
            central_or_satellite_remnant = -1
            SubhaloHalfMassRadii_remnant = -1
    
    print("Redshifts check:",merger_redshift,(z_list[temp<0.0001])[0])
    merger_properties['merger_redshift'].append(merger_redshift)
    merger_properties['remnant_redshift'].append((z_list[temp<0.0001])[0])
    merger_properties['BH_ID1'].append(pid)
    merger_properties['BH_ID2'].append(sid)
    merger_properties['BH_Mass1'].append(p_mass)
    merger_properties['BH_Mass2'].append(s_mass)
    merger_properties['remnant_SubhaloID'].append(SubhaloID_remnant)
    merger_properties['remnant_SubhaloStellarMass'].append(SubhaloStellarMass_remnant)
    merger_properties['remnant_SubhaloDarkMatterMass'].append(SubhaloDarkMatterMass_remnant)
    merger_properties['remnant_central_or_satellite'].append(central_or_satellite_remnant)
    merger_properties['remnant_SubhaloHalfMassRadii'].append(SubhaloHalfMassRadii_remnant)
    merger_properties['merger_type'].append(found_at_least_one_ID_in_next_snap)   
    return merger_properties

#path_to_output='/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS//L3p125n512//' # this is the folder containing the simulation run
#run='/AREPO/' # name of the simulation runs
#output='output_ratio1000_SFMFGM5_seed3.19_discreteDF_HDM5.00_strict_velocity/'
basePath=path_to_output+run+output
snap_list, z_list=arepo_package.get_snapshot_redshift_correspondence(basePath)
#z_list=numpy.array([round(zz) for zz in z_list]).astype(int)


upto_redshift=0

if (MERGE_SNAP == 1):
   scale_fac_complete_sorted,primary_mass_sorted,secondary_mass_sorted,primary_id_sorted,secondary_id_sorted,file_id_complete_sorted,N_empty=arepo_package.get_merger_events_from_snapshot(basePath,upto_redshift,HOSTS=0)
else:
   scale_fac_complete_sorted,primary_mass_sorted,secondary_mass_sorted,primary_id_sorted,secondary_id_sorted,file_id_complete_sorted,N_empty= arepo_package.get_merger_events(basePath,get_primary_secondary_indices=0,HDF5=1,SORT_PRIMARY_SECONDARY=0)



redshift_complete_sorted = 1./scale_fac_complete_sorted - 1
print(redshift_complete_sorted[0:2])

ParticleID_dict = dict.fromkeys(snap_list)
BH_Mass_dict = dict.fromkeys(snap_list)
SubhaloID_dict = dict.fromkeys(snap_list)
SubhaloStellarMass_dict = dict.fromkeys(snap_list)
SubhaloDarkMatterMass_dict = dict.fromkeys(snap_list)
central_or_satellite_dict = dict.fromkeys(snap_list)
SubhaloHalfMassRadii_dict = dict.fromkeys(snap_list)

for snap,desired_redshift in list(zip(snap_list,z_list)):
    ParticleIDs,BH_Mass,SubhaloID,SubhaloStellarMass,SubhaloDarkMatterMass,central_or_satellite, SubhaloHalfMassRadii, o_redshift=numpy.load(basePath+'/bhdata_with_subhalos/snapshot_%s.npy'%(snap),allow_pickle=True)   
    ParticleID_dict[snap]=ParticleIDs
    BH_Mass_dict[snap]=BH_Mass
    SubhaloID_dict[snap]=SubhaloID
    SubhaloStellarMass_dict[snap]=SubhaloStellarMass
    SubhaloDarkMatterMass_dict[snap]=SubhaloDarkMatterMass
    central_or_satellite_dict[snap]=central_or_satellite
    SubhaloHalfMassRadii_dict[snap]=SubhaloHalfMassRadii

keys = ['merger_redshift','remnant_redshift','BH_ID1','BH_ID2','BH_Mass1','BH_Mass2','remnant_SubhaloID','remnant_SubhaloDarkMatterMass','remnant_SubhaloStellarMass','remnant_SubhaloHalfMassRadii','remnant_central_or_satellite','merger_type']
merger_properties = {key: [] for key in keys} 
   
for N in range(0,len(scale_fac_complete_sorted)):
    print("Loop ",N)
    evaluate(N)
    
numpy.save(basePath + '/merger_statistics_subhalo_remnants.npy',[merger_properties]) 

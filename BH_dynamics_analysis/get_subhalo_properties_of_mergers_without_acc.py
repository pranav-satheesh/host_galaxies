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

DFD = 1
KIN = 1
def evaluate(N):
    merger_redshift=redshift_complete_sorted[N]
    pid= primary_id_sorted[N]
    sid= secondary_id_sorted[N]
    temp = z_list-merger_redshift
    redshift_snapshots_to_search = numpy.flip(numpy.sort(snap_list[temp>0]))
    #print(redshift_snapshots_to_search)
    seed_redshift_primary=max(Redshift_details[BH_ID_details==pid])
    seed_redshift_secondary=max(Redshift_details[BH_ID_details==sid])
    seed_redshift = min([seed_redshift_primary,seed_redshift_secondary])
        
    found_at_least_one_snap_with_both_ids = 0
    found_at_least_one_snap_with_both_ids_in_one_halo = 0
    for snap in redshift_snapshots_to_search:
        ParticleIDs = ParticleID_dict[snap]
        BH_Mass = BH_Mass_dict[snap]
        SubhaloID = SubhaloID_dict[snap]
        SubhaloStellarMass = SubhaloStellarMass_dict[snap]
        SubhaloDarkMatterMass = SubhaloDarkMatterMass_dict[snap]
        central_or_satellite = central_or_satellite_dict[snap]
        SubhaloHalfMassRadii = SubhaloHalfMassRadii_dict[snap]

        if (pid in ParticleIDs) & (sid in ParticleIDs):
            found_at_least_one_snap_with_both_ids = 1
            BH_Mass_primary=BH_Mass[pid==ParticleIDs]
            BH_Mass_secondary=BH_Mass[sid==ParticleIDs]

            SubhaloID_primary = SubhaloID[pid == ParticleIDs]
            SubhaloID_secondary = SubhaloID[sid == ParticleIDs]

            SubhaloStellarMass_primary = SubhaloStellarMass[pid == ParticleIDs]
            SubhaloStellarMass_secondary = SubhaloStellarMass[sid == ParticleIDs]

            SubhaloDarkMatterMass_primary = SubhaloDarkMatterMass[pid == ParticleIDs]
            SubhaloDarkMatterMass_secondary = SubhaloDarkMatterMass[sid == ParticleIDs]

            central_or_satellite_primary = central_or_satellite[pid == ParticleIDs]
            central_or_satellite_secondary = central_or_satellite[sid == ParticleIDs]

            SubhaloHalfMassRadii_primary = SubhaloHalfMassRadii[pid == ParticleIDs]
            SubhaloHalfMassRadii_secondary = SubhaloHalfMassRadii[sid == ParticleIDs]

            if (SubhaloID_primary!=SubhaloID_secondary):
                merger_type = 2
                subhalo_merger_snap_before = snap
                break
            else:
                found_at_least_one_snap_with_both_ids_in_one_halo = 1
                subhalo_merger_snap_after = snap
                merged_SubhaloID = SubhaloID_primary[0]
                merged_SubhaloStellarMass = SubhaloStellarMass_primary[0]
                merged_SubhaloDarkMatterMass = SubhaloDarkMatterMass_primary[0]
                merged_central_or_satellite = central_or_satellite_primary[0]
                merged_SubhaloHalfMassRadii = SubhaloHalfMassRadii_primary[0]
        else:
            if (found_at_least_one_snap_with_both_ids == 1):                
                merger_type = 1
            else: 
                merger_type = 0
            break
   
    if (merger_type == 2) & (found_at_least_one_snap_with_both_ids_in_one_halo == 0):
        merger_type = 3

    if(merger_type==0):
        subhalo_merger_redshift_before = seed_redshift
        subhalo_merger_redshift_after = merger_redshift   
        merged_SubhaloID = -1
        merged_SubhaloStellarMass = -1
        merged_SubhaloDarkMatterMass = -1
        merged_central_or_satellite = -1
        merged_SubhaloHalfMassRadii = -1
    elif(merger_type==1):
        subhalo_merger_redshift_after=z_list[subhalo_merger_snap_after==snap_list][0]
        subhalo_merger_redshift_before= seed_redshift
    elif(merger_type==3):
        subhalo_merger_redshift_after = merger_redshift
        subhalo_merger_redshift_before= z_list[subhalo_merger_snap_before==snap_list][0]
        merged_SubhaloID = -1
        merged_SubhaloStellarMass = -1
        merged_SubhaloDarkMatterMass = -1
        merged_central_or_satellite = -1
        merged_SubhaloHalfMassRadii = -1
    else:
        subhalo_merger_redshift_after=z_list[subhalo_merger_snap_after==snap_list][0]
        subhalo_merger_redshift_before=z_list[subhalo_merger_snap_before==snap_list][0]

    mask_trajectory = (Redshift_details >= merger_redshift-redshift_tolerance) & (Redshift_details <= subhalo_merger_redshift_before+redshift_tolerance)
    mask_primary = BH_ID_details==pid
    mask_secondary = BH_ID_details==sid
    
    
    xpos_trajectory_primary = xpos_details[mask_trajectory & mask_primary]
    ypos_trajectory_primary = ypos_details[mask_trajectory & mask_primary]
    zpos_trajectory_primary = zpos_details[mask_trajectory & mask_primary]
    
    xvel_trajectory_primary = xvel_details[mask_trajectory & mask_primary]
    yvel_trajectory_primary = yvel_details[mask_trajectory & mask_primary]
    zvel_trajectory_primary = zvel_details[mask_trajectory & mask_primary]

    xacc_trajectory_primary = xacc_details[mask_trajectory & mask_primary]
    yacc_trajectory_primary = yacc_details[mask_trajectory & mask_primary]
    zacc_trajectory_primary = zacc_details[mask_trajectory & mask_primary]

    if(DFD == 1):
       xDFDacc_trajectory_primary = xDFDacc_details[mask_trajectory & mask_primary]
       yDFDacc_trajectory_primary = yDFDacc_details[mask_trajectory & mask_primary]
       zDFDacc_trajectory_primary = zDFDacc_details[mask_trajectory & mask_primary]


    Redshift_trajectory_primary = Redshift_details[mask_trajectory & mask_primary]
    
    
    xpos_trajectory_secondary = xpos_details[mask_trajectory & mask_secondary]
    ypos_trajectory_secondary = ypos_details[mask_trajectory & mask_secondary]
    zpos_trajectory_secondary = zpos_details[mask_trajectory & mask_secondary]
    
    xvel_trajectory_secondary = xvel_details[mask_trajectory & mask_secondary]
    yvel_trajectory_secondary = yvel_details[mask_trajectory & mask_secondary]
    zvel_trajectory_secondary = zvel_details[mask_trajectory & mask_secondary]

    xacc_trajectory_secondary = xacc_details[mask_trajectory & mask_secondary]
    yacc_trajectory_secondary = yacc_details[mask_trajectory & mask_secondary]
    zacc_trajectory_secondary = zacc_details[mask_trajectory & mask_secondary]

    if(DFD == 1):
        xDFDacc_trajectory_secondary = xDFDacc_details[mask_trajectory & mask_secondary]
        yDFDacc_trajectory_secondary = yDFDacc_details[mask_trajectory & mask_secondary]
        zDFDacc_trajectory_secondary = zDFDacc_details[mask_trajectory & mask_secondary]

    Redshift_trajectory_secondary = Redshift_details[mask_trajectory & mask_secondary]
    

    

    numpy.save(path_to_trajectories + '%d_primary.npy'%N,[xpos_trajectory_primary,ypos_trajectory_primary,zpos_trajectory_primary,xvel_trajectory_primary,yvel_trajectory_primary,zvel_trajectory_primary,Redshift_trajectory_primary])
    numpy.save(path_to_trajectories + '%d_secondary.npy'%N,[xpos_trajectory_secondary,ypos_trajectory_secondary,zpos_trajectory_secondary,xvel_trajectory_secondary,yvel_trajectory_secondary,zvel_trajectory_secondary,Redshift_trajectory_secondary])
 
    numpy.save(path_to_trajectories + '%d_acc_primary.npy'%N,[xacc_trajectory_primary,yacc_trajectory_primary,zacc_trajectory_primary])
    numpy.save(path_to_trajectories + '%d_acc_secondary.npy'%N,[xacc_trajectory_secondary,yacc_trajectory_secondary,zacc_trajectory_secondary])

    if(DFD == 1):
        numpy.save(path_to_trajectories + '%d_DFDacc_primary.npy'%N,[xDFDacc_trajectory_primary,yDFDacc_trajectory_primary,zDFDacc_trajectory_primary])
        numpy.save(path_to_trajectories + '%d_DFDacc_secondary.npy'%N,[xDFDacc_trajectory_secondary,yDFDacc_trajectory_secondary,zDFDacc_trajectory_secondary])


    merger_time_scale_upper=T(merger_redshift,subhalo_merger_redshift_before)/hubble/1e6
    merger_time_scale_lower=T(merger_redshift,subhalo_merger_redshift_after)/hubble/1e6
    merger_time_scale_best =(merger_time_scale_upper + merger_time_scale_lower)/2
    merger_time_scale_err = (merger_time_scale_upper - merger_time_scale_lower)/2
        

    
    merger_properties['merger_redshift'].append(merger_redshift)
    merger_properties['seed_redshift'].append(seed_redshift)
    merger_properties['merger_time_scale_best'].append(merger_time_scale_best)
    merger_properties['merger_time_scale_err'].append(merger_time_scale_err)
    merger_properties['subhalo_merger_redshift_after'].append(subhalo_merger_redshift_after)
    merger_properties['subhalo_merger_redshift_before'].append(subhalo_merger_redshift_before)
    merger_properties['merged_SubhaloID'].append(merged_SubhaloID)
    merger_properties['merged_SubhaloStellarMass'].append(merged_SubhaloStellarMass)
    merger_properties['merged_SubhaloDarkMatterMass'].append(merged_SubhaloDarkMatterMass)
    merger_properties['merged_central_or_satellite'].append(merged_central_or_satellite)
    merger_properties['merged_SubhaloHalfMassRadii'].append(merged_SubhaloHalfMassRadii)
    merger_properties['merger_type'].append(merger_type)
    
    
    initial_redshift_trajectory = min([max(Redshift_trajectory_primary),max(Redshift_trajectory_secondary)])
    
    diff = abs(Redshift_trajectory_primary-initial_redshift_trajectory)
    
    
    initial_trajectory_primary['initial_xpos'].append(xpos_trajectory_primary[diff==min(diff)][0])
    initial_trajectory_primary['initial_ypos'].append(ypos_trajectory_primary[diff==min(diff)][0])
    initial_trajectory_primary['initial_zpos'].append(zpos_trajectory_primary[diff==min(diff)][0])
    
    initial_trajectory_primary['initial_xvel'].append(xvel_trajectory_primary[diff==min(diff)][0])
    initial_trajectory_primary['initial_yvel'].append(yvel_trajectory_primary[diff==min(diff)][0])
    initial_trajectory_primary['initial_zvel'].append(zvel_trajectory_primary[diff==min(diff)][0])

#    initial_trajectory_primary['initial_xacc'].append(xacc_trajectory_primary[diff==min(diff)][0])
#    initial_trajectory_primary['initial_yacc'].append(yacc_trajectory_primary[diff==min(diff)][0])
#    initial_trajectory_primary['initial_zacc'].append(zacc_trajectory_primary[diff==min(diff)][0])

    diff = abs(Redshift_trajectory_secondary-initial_redshift_trajectory)
    
    initial_trajectory_secondary['initial_xpos'].append(xpos_trajectory_secondary[diff==min(diff)][0])
    initial_trajectory_secondary['initial_ypos'].append(ypos_trajectory_secondary[diff==min(diff)][0])
    initial_trajectory_secondary['initial_zpos'].append(zpos_trajectory_secondary[diff==min(diff)][0])
    
    initial_trajectory_secondary['initial_xvel'].append(xvel_trajectory_secondary[diff==min(diff)][0])
    initial_trajectory_secondary['initial_yvel'].append(yvel_trajectory_secondary[diff==min(diff)][0])
    initial_trajectory_secondary['initial_zvel'].append(zvel_trajectory_secondary[diff==min(diff)][0])

#    initial_trajectory_secondary['initial_xacc'].append(xacc_trajectory_secondary[diff==min(diff)][0])
#    initial_trajectory_secondary['initial_yacc'].append(yacc_trajectory_secondary[diff==min(diff)][0])
#    initial_trajectory_secondary['initial_zacc'].append(zacc_trajectory_secondary[diff==min(diff)][0])

   

    return merger_properties,initial_trajectory_primary,initial_trajectory_secondary
    


#path_to_output='/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS//L3p125n512//' # this is the folder containing the simulation run
#run='/AREPO/' # name of the simulation runs
#output='output_ratio1000_SFMFGM5_seed3.19_discreteDF_HDM5.00_strict_velocity/'
basePath=path_to_output+run+output
snap_list, z_list=arepo_package.get_snapshot_redshift_correspondence(basePath)
#z_list=numpy.array([round(zz) for zz in z_list]).astype(int)


#print(z_list)
if (os.path.exists(basePath+'blackhole_details.hdf5')==True):
    print("Converting details to hdf5")
    arepo_package.convert_details_to_hdf5(basePath,DFD=DFD,KIN=KIN)
else:
    print('Details already exist')
print("Done converting")
details=h5py.File(basePath+'blackhole_details.hdf5')
BH_ID_details=details.get('BH_ID')[:]
BH_Mass_details=details.get('BH_Mass')[:]
ScaleFactor_details=details.get('ScaleFactor')[:]
Redshift_details = 1./ScaleFactor_details-1.


xpos_details = details.get('xpos')[:]
ypos_details = details.get('ypos')[:]
zpos_details = details.get('zpos')[:]

xvel_details = details.get('xvel')[:]
yvel_details = details.get('yvel')[:]
zvel_details = details.get('zvel')[:]


xacc_details = details.get('xacc')[:]
yacc_details = details.get('yacc')[:]
zacc_details = details.get('zacc')[:]


if (DFD == 1):
   xDFDacc_details = details.get('xDFDacc')[:]
   yDFDacc_details = details.get('yDFDacc')[:]
   zDFDacc_details = details.get('zDFDacc')[:]




Redshift_details=1./ScaleFactor_details-1

redshift_tolerance = 0.1

upto_redshift=0
scale_fac_complete_sorted,primary_mass_sorted,secondary_mass_sorted,primary_id_sorted,secondary_id_sorted,file_id_complete_sorted,N_empty=arepo_package.get_merger_events_from_snapshot(basePath,upto_redshift,HOSTS=0)
redshift_complete_sorted = 1./scale_fac_complete_sorted - 1
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

path_to_trajectories = basePath+'/merger_trajectories2/'
if (os.path.exists(path_to_trajectories)==False):
    os.makedirs(path_to_trajectories)

keys = ['merger_redshift','seed_redshift','merger_time_scale_best','merger_time_scale_err','subhalo_merger_redshift_after','subhalo_merger_redshift_before','merged_SubhaloID','merged_SubhaloDarkMatterMass','merged_SubhaloStellarMass','merged_SubhaloHalfMassRadii','merged_central_or_satellite','merger_type']
merger_properties = {key: [] for key in keys} 
    
keys = ['initial_xpos','initial_ypos','initial_zpos','initial_xvel','initial_yvel','initial_zvel']
initial_trajectory_primary = {key: [] for key in keys} 
    
keys = ['initial_xpos','initial_ypos','initial_zpos','initial_xvel','initial_yvel','initial_zvel']
initial_trajectory_secondary = {key: [] for key in keys} 
for N in range(0,len(scale_fac_complete_sorted)):
    print("Loop ",N)
    evaluate(N)
    
numpy.save(basePath + '/merger_statistics_subhalo.npy',[merger_properties,initial_trajectory_primary,initial_trajectory_secondary])


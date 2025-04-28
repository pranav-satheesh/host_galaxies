import sys
sys.path.append('/home/aklantbhowmick/anaconda3/lib/python3.7/site-packages')
sys.path.append('/home/aklantbhowmick/anaconda3/lib/python3.7/site-packages/scalpy/')
sys.path.append('/home/aklantbhowmick/anaconda3/envs/nbodykit-env/lib/python3.6/site-packages/')
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

title_fontsize=30
import warnings
mass_unit_conversion=1e10


desired_redshift_list = numpy.array([0, 1, 2, 5, 10])
path_to_output='/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L3p125n1024//'
run='/AREPO/' # name of the simulation runs
output='output_ratio1000_SFMFGM5_seed3.19_discreteDF_HDM5.00_strict_velocity_gas_res/'
basePath=path_to_output+run+output


#path_to_output='/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512//'
#run='/AREPO/' # name of the simulation runs
#output='output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_spin/'
#basePath=path_to_output+run+output

snapshot_space,redshift_space = arepo_package.get_snapshot_redshift_correspondence(basePath)
diff_list = numpy.array([abs(redshift_space - redshift) for redshift in desired_redshift_list]) 
actual_redshift_list = numpy.array([redshift_space[diff == min(diff)][0] for diff in diff_list])
actual_snapshot_list = numpy.array([snapshot_space[diff == min(diff)][0] for diff in diff_list])


def get_halo_density_profile(output_path,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,min_edge,max_edge,Nbins,CENTER_AROUND='POTENTIAL_MINIMUM',p_id=0):
    from kdcount import correlate
    def min_dis(median_position, position,box_size):
        pos_1=position-median_position
        pos_2=position-median_position+boxsize
        pos_3=position-median_position-boxsize
        new_position_options=numpy.array([pos_1,pos_2,pos_3])
        get_minimum_distance=numpy.argmin(numpy.abs(new_position_options))
        return new_position_options[get_minimum_distance]
    boxsize=arepo_package.get_box_size(output_path)
    particle_property='Coordinates'
    print("!!!")
    group_positions,output_redshift=arepo_package.get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)
    if (p_type==1):
        MassDM=arepo_package.load_snapshot_header(output_path,desired_redshift_of_selected_halo)['MassTable'][1]
        group_mass=numpy.array([1.]*len(group_positions))*MassDM
    else:
        particle_property='Masses'
        group_mass,output_redshift=arepo_package.get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False) 
    particle_property='Potential'
    group_potential,output_redshift=arepo_package.get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)
    print("!!!2")
    if (p_type==4):
        particle_property='GFM_StellarFormationTime'       
        GFM_StellarFormationTime,output_redshift=arepo_package.get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)
        mask=GFM_StellarFormationTime>0
        group_potential=group_potential[mask]
        group_mass=group_mass[mask]
        group_positions=group_positions[mask]
    
     
    if (CENTER_AROUND=='POTENTIAL_MINIMUM'):
        centers,o = arepo_package.get_group_property(output_path,'GroupPos',desired_redshift_of_selected_halo) 
        center = centers[index_of_selected_halo] 
    print("!!!3")       

    GroupVel,o = arepo_package.get_group_property(output_path,'GroupVel',desired_redshift_of_selected_halo)
    halo_velocity = GroupVel[index_of_selected_halo]

    transposed_group_positions=numpy.transpose(group_positions)
    vectorized_min_dis = numpy.vectorize(min_dis)
    x_dis=vectorized_min_dis(center[0],transposed_group_positions[0],boxsize)
    y_dis=vectorized_min_dis(center[1],transposed_group_positions[1],boxsize)
    z_dis=vectorized_min_dis(center[2],transposed_group_positions[2],boxsize)
    log_distances=numpy.log10(numpy.sqrt(x_dis**2+y_dis**2+z_dis**2))

    log_distance_bins=numpy.linspace(min_edge,max_edge,Nbins)
    binning=correlate.RBinning(log_distance_bins)
    bin_edges=binning.edges
    bin_centers=binning.centers
    mass_distribution=[]
    for i in range(0,len(bin_edges)-1):
        left=bin_edges[i]
        right=bin_edges[i+1]
        mask=(log_distances>left)&(log_distances<right)
        mass_inside_bin=numpy.sum(group_mass[mask])
        mass_distribution.append(mass_inside_bin)

    mass_distribution=numpy.array(mass_distribution)
    mass_volumes=4./3*3.14*numpy.diff((10**bin_edges)**3)
    mass_density=mass_distribution/mass_volumes
    #/4./3.14/(10**bin_centers)**2/((numpy.diff(bin_centers))[0])/numpy.log(10)
    return bin_centers,mass_distribution,mass_density,center,halo_velocity




if(1==1):
    Ntasks = 20
    Nmax = 10000
    size = Ntasks
    min_edge,max_edge,Nbins = -3,3,20
    merger_properties,initial_trajectory_primary,initial_trajectory_secondary=numpy.load(basePath + '/merger_statistics.npy',allow_pickle=True)
    merger_type=numpy.array(merger_properties['merger_type'])
    merger_redshift = numpy.array(merger_properties['merger_redshift'])
    merger_time_scale_best = numpy.array(merger_properties['merger_time_scale_best'])
    merger_time_scale_err = numpy.array(merger_properties['merger_time_scale_err'])
    halo_merger_redshift_before = numpy.array(merger_properties['halo_merger_redshift_before'])
    halo_merger_redshift_after = numpy.array(merger_properties['halo_merger_redshift_after'])
    halo_merger_redshift_mid = (halo_merger_redshift_before+halo_merger_redshift_after)/2
    halo_merger_redshift_err = (halo_merger_redshift_before-halo_merger_redshift_after)/2

    
    upto_redshift=0
    scale_fac_complete_sorted,mass1_sorted,mass2_sorted,id1_sorted,id2_sorted,file_id_complete_sorted,N_empty=arepo_package.get_merger_events_from_snapshot(basePath,upto_redshift,HOSTS=0)
    redshift_complete_sorted = 1./scale_fac_complete_sorted - 1    
    #mask = (merger_type == 2) | (merger_type == 3)
    
    all_indices = numpy.arange(0,len(merger_type))
    eligible_indices = all_indices#[mask]
    
    
    GroupStellarMass_space = []
    GroupGasMass_space = []
    GroupDMMass_space = []
    Group_M_Mean200_space = []
    Group_R_Mean200_space = []
    DM_density_profiles = []
    Gas_density_profiles = []
    Stellar_density_profiles = []
    successful_indices = []
    failed_indices = []
    progenitor_haloid_space = []
    center_space = []
    halo_velocity_space = []
    progenitor_redshift_space = []
    bin_centers_space = []

    Nt=numpy.amin([Nmax,len(merger_type)])

    rank=numpy.int(sys.argv[1])
    chunk_size = Nt // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank < size - 1 else Nt

    
    for index in eligible_indices[start:end]:
        try:
            o,snap=arepo_package.desired_redshift_to_output_redshift(basePath,halo_merger_redshift_before[index],list_all=False)
            haloid_space,halomass_space,haloradii_space,central_satellite_space_halo=numpy.load(basePath+'/bhdata_with_subhalos/snapshot_%s_halo.npy'%(snap),allow_pickle=True)   
            bhid_space,bhmass_space,subhaloid_space,subhalostellarmass_space,subhalodmmass_space,central_satellite_space,halfmassradii_space,redshift=numpy.load(basePath+'/bhdata_with_subhalos/snapshot_%s.npy'%(snap),allow_pickle=True)   
            id_tuple = numpy.array([id1_sorted[index],id2_sorted[index]])
            mass_tuple = numpy.array([mass1_sorted[index],mass2_sorted[index]])
            primary_id = id_tuple[mass_tuple == numpy.amax(mass_tuple)][0]
            haloid_primary = haloid_space[primary_id == bhid_space][0].astype(int)
            print(haloid_primary,o)
            ParticleIDs_group,o = arepo_package.get_particle_property_within_groups(basePath,'ParticleIDs',5,o,haloid_primary,store_all_offsets=1)
            if(primary_id not in ParticleIDs_group):
                print("Warning: Primary id not found in the halo")

            Group_M_Mean200,oo=arepo_package.get_group_property(basePath,'Group_M_Mean200',o)
            Group_M_Mean200_space.append(Group_M_Mean200[haloid_primary])

            Group_R_Mean200,oo=arepo_package.get_group_property(basePath,'Group_R_Mean200',o)
            Group_R_Mean200_space.append(Group_R_Mean200[haloid_primary])


            max_edge = numpy.log10(10. * Group_R_Mean200[haloid_primary])

            GroupMassType,oo=arepo_package.get_group_property(basePath,'GroupMassType',o)
            GroupStellarMass = GroupMassType[:,4]
            GroupGasMass = GroupMassType[:,0]
            GroupDMMass = GroupMassType[:,1]

            if (GroupStellarMass[haloid_primary] > 0):
                bin_centers,mass_distribution,mass_density,center,halo_velocity=get_halo_density_profile(basePath,4,o,haloid_primary,min_edge,max_edge,Nbins,CENTER_AROUND='POTENTIAL_MINIMUM')   
                Stellar_density_profiles.append(mass_density)
            else:
                Stellar_density_profiles.append(numpy.array([0.]*(Nbins-1)))

            bin_centers,mass_distribution,mass_density,center,halo_velocity=get_halo_density_profile(basePath,0,o,haloid_primary,min_edge,max_edge,Nbins,CENTER_AROUND='POTENTIAL_MINIMUM')   
            Gas_density_profiles.append(mass_density)
            bin_centers,mass_distribution,mass_density,center,halo_velocity=get_halo_density_profile(basePath,1,o,haloid_primary,min_edge,max_edge,Nbins,CENTER_AROUND='POTENTIAL_MINIMUM')   
            DM_density_profiles.append(mass_density)
            GroupStellarMass_space.append(GroupStellarMass[haloid_primary])
            GroupGasMass_space.append(GroupGasMass[haloid_primary])
            GroupDMMass_space.append(GroupDMMass[haloid_primary])
            center_space.append(center)
            halo_velocity_space.append(halo_velocity)
            progenitor_redshift_space.append(o)
            successful_indices.append(index)
            progenitor_haloid_space.append(haloid_primary)
            bin_centers_space.append(bin_centers)
        except IndexError:
            failed_indices.append(index)
            aaa = 1

    successful_indices = numpy.array(successful_indices)
    failed_indices = numpy.array(failed_indices)
    Group_M_Mean200_space = numpy.array(Group_M_Mean200_space)
    Group_R_Mean200_space = numpy.array(Group_R_Mean200_space)
    GroupStellarMass_space = numpy.array(GroupStellarMass_space)
    GroupGasMass_space = numpy.array(GroupGasMass_space)
    GroupDMMass_space = numpy.array(GroupDMMass_space)
    DM_density_profiles = numpy.array(DM_density_profiles)
    Stellar_density_profiles = numpy.array(Stellar_density_profiles)
    Gas_density_profiles = numpy.array(Gas_density_profiles)
    center_space = numpy.array(center_space)
    halo_velocity_space = numpy.array(halo_velocity_space)
    progenitor_redshift_space = numpy.array(progenitor_redshift_space)
    progenitor_haloid_space = numpy.array(progenitor_haloid_space)
    bin_centers_space = numpy.array(bin_centers_space)
    output_dir = basePath + '/merger_progenitor_halo_properties/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = 'block_%d.npy'%(rank)
    numpy.save(output_dir+file_name, [successful_indices, failed_indices, Group_M_Mean200_space, Group_R_Mean200_space, GroupDMMass_space, GroupGasMass_space, GroupStellarMass_space, DM_density_profiles, Gas_density_profiles, Stellar_density_profiles,bin_centers_space,center_space,progenitor_redshift_space,progenitor_haloid_space,halo_velocity_space])
    #print(DM_density_profiles)

import sys
import numpy as np
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
#%pylab inline

title_fontsize=30
import warnings


mass_unit_conversion=1e10

Solarmass_to_SI = 1e10*2e30/0.6771
kpc_to_SI = 3.086e19/0.6771
kms_to_SI = 1e3


def integrate_halo_mass(profile, bin_centers):
    """
    Integrates the mass density profile to compute the halo mass.
    
    Parameters:
    profile (array-like): Mass density at different radial bins.
    bin_centers (array-like): Radial bin centers in log10 units.

    Returns:
    float: Total halo mass.
    """
    # Convert log10 radial bin centers to linear scale
    radii = 10**bin_centers
    
    # Calculate the volume of each shell
    shell_volumes = (4/3) * np.pi * (radii[1:]**3 - radii[:-1]**3)
    
    # Compute the mass in each shell
    shell_masses = profile[:-1] * shell_volumes
    
    # Total halo mass is the sum of masses in all shells
    total_mass = np.sum(shell_masses)
    
    return total_mass


def log_pl_profile(r, log_rho_s,gamma):
    """NFW density profile function"""
    return log_rho_s + gamma * numpy.log10(r)

def pl_profile_final(r, rho_s, gamma):
    """NFW density profile function"""
    return rho_s * r**gamma

def fit_pl_profile(r, rho, p0=None):
    """
    Fit an NFW density profile to observed data.
    
    Parameters:
    r (array): Radial distances
    rho (array): Observed density values
    p0 (tuple): Initial guess for (rho_s, r_s), optional
    
    Returns:
    tuple: Best-fit parameters (rho_s, r_s)
    """
    

    
    # Perform the fit
    popt, _ = curve_fit(log_pl_profile, r, rho, p0=p0,method='trf',maxfev=100000)
    
    return popt



if (1==1):
    path_to_output='/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L3p125n1024//'
    run='/AREPO/' # name of the simulation runs
    output='output_ratio1000_SFMFGM5_seed3.19_discreteDF_HDM5.00_strict_velocity_gas_res/'
    basePath=path_to_output+run+output

    output_dir = basePath + '/merger_progenitor_halo_properties/'

    rank=0
    file_name = 'block_%d.npy'%(rank)
    files = os.listdir(output_dir)

    # Iterate over each file and load it using numpy.load
    #successful_indices_space=numpy.array([],dtype=int)
    #failed_indices_space = numpy.array([],dtype=int)
    #Group_M_Mean200_space = numpy.array([])
    #Group_R_Mean200_space = numpy.array([])
    #GroupDMMass_space = numpy.array([])
    #GroupGasMass_space = numpy.array([])
    #GroupStellarMass_space = numpy.array([])
    #DM_density_profiles_space = numpy.array([])
    #Gas_density_profiles_space = numpy.array([])
    #Stellar_density_profiles_space = numpy.array([])
    i = 0
    for file_name in files:
        if (file_name == 'NFW_fit_parameters.npy') | (file_name == 'PL_fit_parameters.npy'):
            continue
        obj=numpy.load(output_dir+file_name,allow_pickle=True)       
        successful_indices, failed_indices, Group_M_Mean200, Group_R_Mean200, GroupDMMass, GroupGasMass, GroupStellarMass, DM_density_profiles, Gas_density_profiles, Stellar_density_profiles,bin_centers,halo_center,progenitor_redshift,progenitor_haloid,halo_velocity = obj
 
        if (i==0):
            successful_indices_space = successful_indices
            failed_indices_space = failed_indices
            Group_M_Mean200_space = Group_M_Mean200
            Group_R_Mean200_space = Group_R_Mean200
            GroupDMMass_space = GroupDMMass
            GroupGasMass_space = GroupGasMass
            GroupStellarMass_space = GroupStellarMass
            DM_density_profiles_space = DM_density_profiles
            Gas_density_profiles_space = Gas_density_profiles
            Stellar_density_profiles_space = Stellar_density_profiles
            progenitor_redshift_space = progenitor_redshift
            progenitor_haloid_space = progenitor_haloid
            halo_center_space = halo_center
            halo_velocity_space = halo_velocity
            bin_centers_space = bin_centers
        #file_name = 'new_BH_distances_from_center_mpi_%d/block_%d.npy'%(desired_redshift,block)
        else:
            successful_indices_space = numpy.append(successful_indices_space,successful_indices)
            failed_indices_space = numpy.append(failed_indices_space, failed_indices)
            Group_M_Mean200_space = numpy.append(Group_M_Mean200_space, Group_M_Mean200)
            Group_R_Mean200_space = numpy.append(Group_R_Mean200_space, Group_R_Mean200)
            GroupDMMass_space = numpy.append(GroupDMMass_space, GroupDMMass)
            GroupGasMass_space = numpy.append(GroupGasMass_space, GroupGasMass)
            GroupStellarMass_space = numpy.append(GroupStellarMass_space, GroupStellarMass)
            DM_density_profiles_space = numpy.append(DM_density_profiles_space, DM_density_profiles,axis=0)
            Gas_density_profiles_space = numpy.append(Gas_density_profiles_space, Gas_density_profiles,axis=0)
            Stellar_density_profiles_space = numpy.append(Stellar_density_profiles_space, Stellar_density_profiles,axis=0)
            progenitor_redshift_space = numpy.append(progenitor_redshift_space,progenitor_redshift)
            progenitor_haloid_space = numpy.append(progenitor_haloid_space,progenitor_haloid)
            halo_center_space = numpy.append(halo_center_space,halo_center,axis=0)
            halo_velocity_space = numpy.append(halo_velocity_space,halo_velocity,axis=0)
            bin_centers_space = numpy.append(bin_centers_space,bin_centers,axis=0)
        i+=1

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
    
    primary_mass_sorted = numpy.amax([mass1_sorted,mass2_sorted],axis = 0)
    
         
        



#integrated_halo_masses=[]
rho_s_final_space,gamma_final_space=[],[]
#f,ax = plt.subplots(1,1,figsize=(10,10))
ii=0
for profile,bin_centers in list(zip(DM_density_profiles_space,bin_centers_space)):
    mask=(profile>0) & (bin_centers < numpy.log10(0.1*Group_R_Mean200_space[ii])) 
    #mask = profile == profile
    r = 10**bin_centers[mask]  # Radial distances
    rho = profile[mask]  # Observed density values

    if (len(profile[mask])==0):
        rho_s_final_space.append(-1)
        gamma_final_space.append(-1)       
        continue

    residuals_pre = 1000
    if(1==1):
        log_rho_s, gamma = fit_pl_profile(r, numpy.log10(rho),p0=[0.000001,-0.1])
        rho_s = 10**log_rho_s
        #print(r_s,r_s_guess)
        #rho_fit = pl_profile(10**(bin_centers[mask]),rho_s,gamma)        
        #residuals = sum((numpy.log10(profile[mask]) - numpy.log10(rho_fit))**2)
        #if (residuals < residuals_pre):
        #    residuals_pre = residuals
        #    rho_s_final, gamma_final = rho_s,r_s
        #print(residuals_pre,rho_s_final,r_s_final)

    print(f"Best-fit parameters: rho_s = {rho_s:.3e}, gamma = {gamma:.3f}")
    
    rho_s_final_space.append(rho_s)
    gamma_final_space.append(gamma)
    
    
#    r_cont = numpy.logspace(-2,numpy.log10(10*Group_R_Mean200_space[ii]),100)
#    rho_cont = nfw_profile_final(r_cont,rho_s_final,r_s_final)
    #print(rho_cont)
#    ax.errorbar(numpy.log10(r_cont),rho_cont)   
    
#    integrated_halo_masses.append(integrate_halo_mass(rho_cont,numpy.log10(r_cont)))
    #ax.errorbar(bin_centers[mask],profile[mask])   
    ii+=1
    
    
    
    
#integrated_halo_masses= numpy.array(integrated_halo_masses)
#ax.set_yscale('log')

rho_s_final_space = numpy.array(rho_s_final_space)
gamma_final_space = numpy.array(gamma_final_space)

numpy.save(output_dir+'/PL_fit_parameters.npy',[rho_s_final_space,gamma_final_space])



#ax.set_xscale('log')


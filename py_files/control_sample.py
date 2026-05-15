import numpy as np
import h5py
import sys
from scipy.spatial import cKDTree
from tqdm import tqdm
import numpy as np
from scipy.stats import ks_2samp
import os
tex_path = '/apps/texlive/2023/bin/x86_64-linux/'
os.environ['PATH'] += os.pathsep + tex_path
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

sys.path.append('../BH_dynamics_analysis')
sys.path.append('/home/pranavsatheesh/arepo_package/')
import arepo_package as arepo

def convert_mdot_to_Lbol(er):
    c=3e8
    mass_unit_conv=1e10/978/1e6
    mass_sun=2e30
    yr_to_sec=3.15e7
    joule_to_ergs=1e7
    lamb=(er/1-er)
    
    total_conv=mass_unit_conv*mass_sun/yr_to_sec*c**2*joule_to_ergs*lamb
    return er*total_conv


class control_samples_TNG:

    def __init__(self,population_file,verbose=False,max_z_tolerance=0.3,max_Mstar_tolerance=0.15):

        self.pop = population_file
        self.N_mergers = len(self.pop['merging_population']['z'])

        self.merger_control_index_pairs =self.find_control_samples(
        max_Mstar_tolerance=max_Mstar_tolerance,
        max_z_tolerance=max_z_tolerance)
        print(f"Number of available mergers in this population is {self.N_mergers:03d}")
        print("Number of cases where a close enough match is not found within the acceptable tolerance:",np.sum(self.merger_control_index_pairs[:,1] == -1))


        self.MBH_not_zero_flag = self.pop['merging_population']['MBH'][:][self.merger_control_index_pairs[:,0]]!=0
        self.control_available_flag = self.merger_control_index_pairs[:,1]!=-1
        self.valid_control_mask  = self.MBH_not_zero_flag&self.control_available_flag

        self.valid_merger_indices = self.merger_control_index_pairs[self.valid_control_mask,0]
        self.valid_control_indices = self.merger_control_index_pairs[self.valid_control_mask,1]

        #filter out only the unique post mergers. Some post merger galaxies are counted multiplle times as they are from progenitors of multiple galaxy mergers.
        Mstar_vals = self.pop['merging_population']['Mstar'][:][self.valid_merger_indices]
        values, counts = np.unique(Mstar_vals, return_counts=True)
        duplicate_values = values[counts > 1]
        #let's also mask the cases where the post merger galaxy is formed as a result of multiple galaxy mergers
        multiple_merger_mask = np.isin(Mstar_vals, duplicate_values)
        self.multi_count_merger_indices = self.valid_merger_indices[multiple_merger_mask]
        self.multi_count_control_indices = self.valid_control_indices[multiple_merger_mask]

        unique_mask = ~np.isin(Mstar_vals, duplicate_values)
        self.valid_merger_indices = self.valid_merger_indices[unique_mask]
        self.valid_control_indices = self.valid_control_indices[unique_mask]

        self.compute_population_properties(verbose)
        self.mini_merger_mask = self.q_merger < 0.1
        self.minor_merger_mask = (self.q_merger >= 0.1) & (self.q_merger < 0.25)
        self.major_merger_mask = self.q_merger >= 0.25

    # def find_control_samples_strict(self):

    #     all_mrgr_z = np.unique(self.pop['merging_population']['z'][:]) #all unique redshifts where BHs/galaxies are merging
    #     merging_pop_Mstar = self.pop['merging_population']['Mstar'][:]
    #     non_merging_pop_Mstar = self.pop['non_merging_population']['Mstar'][:]

    #     merger_control_index_pairs = []
    #     used = np.zeros(len(non_merging_pop_Mstar),dtype=bool)

    #     for z_i in tqdm(all_mrgr_z,"processing each merger redshifts for controls"):
    #         zi_merger_ix = np.where(self.pop['merging_population']['z']==z_i)[0]
    #         zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z']==z_i)[0]
    #         zi_nonmerger_ix = zi_nonmrgr_ix[used[zi_nonmrgr_ix]==False]

    #         merger_Mstars = self.pop['merging_population']['Mstar'][zi_merger_ix]
    #         nonmerger_Mstars = self.pop['non_merging_population']['Mstar'][zi_nonmerger_ix]

    #         for Mstar_merger_i in merger_Mstars:
    #              closest_non_merger_ix = np.argmin(np.abs(nonmerger_Mstars - Mstar_merger_i))
    #              mass_diff = np.abs(np.log(nonmerger_Mstars[closest_non_merger_ix]) - np.log(Mstar_merger_i))
                
    #              if mass_diff <= 0.1:
    #                  merger_index = np.where(merging_pop_Mstar==Mstar_merger_i)[0][0]
    #                  non_merger_index = np.where(non_merging_pop_Mstar==nonmerger_Mstars[closest_non_merger_ix])[0][0]
    #                  merger_control_index_pairs.append([merger_index,non_merger_index])
    #                  used[non_merger_index] = True #mark this non-merging galaxy as used
    #              else:
    #                  merger_control_index_pairs.append([merger_index,-1]) #no suitable control found

    #     return  np.array(merger_control_index_pairs)

    # def find_control_samples_strict_v2(self,max_Mstar_tolerance=0.2,max_z_tolerance=0.1):

    #     all_mrgr_z = np.unique(self.pop['merging_population']['z'][:]) #all unique redshifts where BHs/galaxies are merging
    #     merging_pop_Mstar = self.pop['merging_population']['Mstar'][:]
    #     non_merging_pop_Mstar = self.pop['non_merging_population']['Mstar'][:]

    #     merger_control_index_pairs = []
    #     used = np.zeros(len(non_merging_pop_Mstar),dtype=bool)
    #     starting_Mstar_tol = 0.1

    #     for i, z_i in enumerate(tqdm(all_mrgr_z, "processing each merger redshifts for controls")):
    #         zi_merger_ix = np.where(self.pop['merging_population']['z'] == z_i)[0]          # global indices into merging pop
    #         zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z'] == z_i)[0]    # global indices into non-merging pop
    #         zi_nonmerger_ix = zi_nonmrgr_ix[~used[zi_nonmrgr_ix]]                         # available candidates (global indices)

    #         merger_Mstars = self.pop['merging_population']['Mstar'][zi_merger_ix]
    #         # note: we will read candidate masses from pop inside helper so they are always in sync
    #         starting_Mstar_tol = starting_Mstar_tol  # keep existing name in scope
    #         max_z_tolerance = max_z_tolerance

    #         def _try_match(candidate_global_idx_array, merger_mass):
    #             """Return global index of matched non-merger or -1."""
    #             if candidate_global_idx_array.size == 0:
    #                 return -1
    #             cand_masses = self.pop['non_merging_population']['Mstar'][candidate_global_idx_array]
    #             best_local = np.argmin(np.abs(cand_masses - merger_mass))
    #             best_global = candidate_global_idx_array[best_local]
    #             mass_diff = np.abs(np.log(cand_masses[best_local]) - np.log(merger_mass))
    #             return best_global if mass_diff <= max_Mstar_tolerance else -1

    #         # loop over mergers at this redshift (use global merger index directly)
    #         for merger_global_idx, merger_mass in zip(zi_merger_ix, merger_Mstars):
    #             matched_non_global = _try_match(zi_nonmerger_ix, merger_mass)

    #             # if not found, try next redshift (if within tolerance) then previous
    #             if matched_non_global == -1 and i < len(all_mrgr_z) - 1:
    #                 next_non_idx = np.where(self.pop['non_merging_population']['z'] == all_mrgr_z[i + 1])[0]
    #                 next_candidates = next_non_idx[~used[next_non_idx]]
    #                 if next_candidates.size and (all_mrgr_z[i + 1] - all_mrgr_z[i]) <= max_z_tolerance:
    #                     matched_non_global = _try_match(next_candidates, merger_mass)

    #             if matched_non_global == -1 and i > 0:
    #                 prev_non_idx = np.where(self.pop['non_merging_population']['z'] == all_mrgr_z[i - 1])[0]
    #                 prev_candidates = prev_non_idx[~used[prev_non_idx]]
    #                 if prev_candidates.size and (all_mrgr_z[i] - all_mrgr_z[i - 1]) <= max_z_tolerance:
    #                     matched_non_global = _try_match(prev_candidates, merger_mass)

    #             if matched_non_global != -1:
    #                 # use global indices directly (no expensive np.where on values)
    #                 merger_control_index_pairs.append([merger_global_idx, matched_non_global])
    #                 used[matched_non_global] = True
    #             else:
    #                 merger_control_index_pairs.append([merger_global_idx, -1])

    #     return  np.array(merger_control_index_pairs)
    
    # def find_control_samples_strict_v3(self,max_Mstar_tolerance=0.6,max_z_tolerance=0.3):

    #     all_mrgr_z = np.unique(self.pop['merging_population']['z'][:]) #all unique redshifts where BHs/galaxies are merging
    #     merging_pop_Mstar = self.pop['merging_population']['Mstar'][:]
    #     non_merging_pop_Mstar = self.pop['non_merging_population']['Mstar'][:]

    #     merger_control_index_pairs = []
    #     used = np.zeros(len(non_merging_pop_Mstar),dtype=bool)
    #     starting_Mstar_tol = 0.1

    #     for i,z_i in enumerate(tqdm(all_mrgr_z,"processing each merger redshifts for controls")):
    #         zi_merger_ix = np.where(self.pop['merging_population']['z']==z_i)[0]
    #         zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z']==z_i)[0]
    #         zi_nonmerger_ix = zi_nonmrgr_ix[used[zi_nonmrgr_ix]==False]

    #         merger_Mstars = self.pop['merging_population']['Mstar'][zi_merger_ix]
    #         nonmerger_Mstars = self.pop['non_merging_population']['Mstar'][zi_nonmerger_ix]

    #         for Mstar_merger_i in merger_Mstars:
    #             closest_non_merger_ix = np.argmin(np.abs(nonmerger_Mstars - Mstar_merger_i))
    #             mass_diff = np.abs(np.log(nonmerger_Mstars[closest_non_merger_ix]) - np.log(Mstar_merger_i))

    #             if mass_diff <= starting_Mstar_tol:
    #                 merger_index = np.where(merging_pop_Mstar==Mstar_merger_i)[0][0]
    #                 non_merger_index = np.where(non_merging_pop_Mstar==nonmerger_Mstars[closest_non_merger_ix])[0][0]
    #                 merger_control_index_pairs.append([merger_index,non_merger_index])
    #                 used[non_merger_index] = True #mark this non-merging galaxy as used
    #             else:
                    

    #                 zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z']==all_mrgr_z[i+1])[0]
    #                 print(all_mrgr_z[i],all_mrgr_z[i+1],all_mrgr_z[i+1]-all_mrgr_z[i])
    #                 zi_nonmerger_ix = zi_nonmrgr_ix[used[zi_nonmrgr_ix]==False]
    #                 diff_z = all_mrgr_z[i+1]-all_mrgr_z[i]

    #                 if zi_nonmerger_ix.size != 0 and np.abs(diff_z)<= max_z_tolerance:
    #                     nonmerger_Mstars = self.pop['non_merging_population']['Mstar'][zi_nonmerger_ix]
    #                     closest_non_merger_ix = np.argmin(np.abs(nonmerger_Mstars - Mstar_merger_i))
    #                     mass_diff = np.abs(np.log(nonmerger_Mstars[closest_non_merger_ix]) - np.log(Mstar_merger_i))
    #                     if mass_diff <= starting_Mstar_tol:
    #                         merger_index = np.where(merging_pop_Mstar==Mstar_merger_i)[0][0]
    #                         non_merger_index = np.where(non_merging_pop_Mstar==nonmerger_Mstars[closest_non_merger_ix])[0][0]
    #                         merger_control_index_pairs.append([merger_index,non_merger_index])
    #                         used[non_merger_index] = True #mark this non-merging galaxy as used
    #                 elif i>0:
    #                     zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z']==all_mrgr_z[i-1])[0]
    #                     zi_nonmerger_ix = zi_nonmrgr_ix[used[zi_nonmrgr_ix]==False]
    #                     diff_z = all_mrgr_z[i]-all_mrgr_z[i-1]
    #                     if zi_nonmerger_ix.size != 0 and np.abs(diff_z)<= max_z_tolerance:
    #                         nonmerger_Mstars = self.pop['non_merging_population']['Mstar'][zi_nonmerger_ix]
    #                         closest_non_merger_ix = np.argmin(np.abs(nonmerger_Mstars - Mstar_merger_i))
    #                         mass_diff = np.abs(np.log(nonmerger_Mstars[closest_non_merger_ix]) - np.log(Mstar_merger_i))
    #                         if mass_diff <= starting_Mstar_tol:
    #                             merger_index = np.where(merging_pop_Mstar==Mstar_merger_i)[0][0]
    #                             non_merger_index = np.where(non_merging_pop_Mstar==nonmerger_Mstars[closest_non_merger_ix])[0][0]
    #                             merger_control_index_pairs.append([merger_index,non_merger_index])
    #                             used[non_merger_index] = True #mark this non-merging galaxy as used
    #                     else:
    #                         merger_control_index_pairs.append([merger_index,-1]) #no suitable control found 

    #     return  np.array(merger_control_index_pairs)
    
    def find_control_samples(self, max_Mstar_tolerance=0.15, max_z_tolerance=0.3):
        mrgr_z = self.pop['merging_population']['z'][:]
        mrgr_Mstar = self.pop['merging_population']['Mstar'][:]
        non_mrgr_z = self.pop['non_merging_population']['z'][:]
        non_mrgr_Mstar = self.pop['non_merging_population']['Mstar'][:]
        
        used = np.zeros(len(non_mrgr_Mstar), dtype=bool)
        merger_control_index_pairs = []

        for merger_idx in tqdm(range(len(mrgr_z)), "finding controls"):
            z_i = mrgr_z[merger_idx]
            mass_i = mrgr_Mstar[merger_idx]
            candidates = np.where((np.abs(non_mrgr_z - z_i) <= max_z_tolerance) & ~used)[0]
            
            matched = -1
            if candidates.size > 0 and mass_i > 0:
                cand_masses = non_mrgr_Mstar[candidates]
                valid = cand_masses > 0
                if valid.any():
                    log_diffs = np.abs(np.log(cand_masses[valid]) - np.log(mass_i))
                    if log_diffs.min() <= max_Mstar_tolerance:
                        matched = candidates[valid][np.argmin(log_diffs)]
            
            merger_control_index_pairs.append([merger_idx, matched])
            if matched != -1:
                used[matched] = True

        return np.array(merger_control_index_pairs)

    # def find_control_samples(self,max_z_tolerance,max_Mstar_dex_tolerance):

    #     merging_points = np.column_stack((self.pop['merging_population']['z'], np.log10(self.pop['merging_population']['Mstar'])))
    #     non_merging_points = np.column_stack((self.pop['non_merging_population']['z'], np.log10(self.pop['non_merging_population']['Mstar'])))

    #     tree = cKDTree(non_merging_points)
    #     used = np.zeros(len(non_merging_points), dtype=bool)
    #     control_indices = []
    #     net_tolerances = []

    #     while True:
    #         closest_indices = np.full(len(merging_points), -1)
    #         tolerances = []
    #         for i in tqdm(range(len(merging_points)), desc="Processing merging points", ncols=100):
    #          #find the closest neibhour 
    #             dist, min_idx = tree.query(merging_points[i])
    #             if(used[min_idx]):
    #                 #if the closest neighbouring non merging galaxy is already matched to a merging galaxy, find the next closest one
    #                 dists,idxs = tree.query(merging_points[i],k=len(non_merging_points))
    #                 min_idx = idxs[np.where(~used[idxs])[0][0]] 
    #             #check for tolerance:
    #             del_z = np.abs(merging_points[i][0]-non_merging_points[min_idx][0])
    #             dex_Mstar = np.abs(np.log10(merging_points[i][1])-np.log10(non_merging_points[min_idx][1]))

    #             z_tolerance = 0.05
    #             Mstar_dex_tolerance = 0.1

                
    #             while True:
    #                 if(del_z < z_tolerance and dex_Mstar < Mstar_dex_tolerance):
    #                     used[min_idx] = True
    #                     closest_indices[i] = min_idx
    #                     tolerances.append((del_z, dex_Mstar))
    #                     break
    #                 else:
    #                     z_tolerance = z_tolerance*1.5
    #                     Mstar_dex_tolerance = Mstar_dex_tolerance*1.5

    #                     if(z_tolerance>max_z_tolerance or Mstar_dex_tolerance>max_Mstar_dex_tolerance):
    #                         closest_indices[i] = -1
    #                         break

    #         control_indices.append(closest_indices)
    #         net_tolerances.append(tolerances)

    #         if np.shape(control_indices)[0]>=1:
    #             break

    #     return control_indices, net_tolerances

    # def find_control_sample_indices(self,pop,matching_threshold=0.99):
    #     merging_points = np.column_stack((pop['merging_population']['z'], np.log10(pop['merging_population']['Mstar'])))
    #     non_merging_points = np.column_stack((pop['non_merging_population']['z'], np.log10(pop['non_merging_population']['Mstar'])))

    #     #matching the merging and non merging population in the z-Mstar plane

    #     # Build a KDTree for fast nearest neighbor search
    #     tree = cKDTree(non_merging_points)

    #     # Track used indices
    #     used = np.zeros(len(non_merging_points), dtype=bool)  # False means available

    #     control_indices = []
    #     p_z = 1.0
    #     p_Mstar = 1.0

    #     while True:
    #         closest_indices = np.full(len(merging_points), -1)  # Store assigned indices

    #         for i in tqdm(range(len(merging_points)), desc="Processing merging points", ncols=100):
    #         #for i in range(len(merging_points)):
    #         # Find the closest available neighbor
    #             d, min_index = tree.query(merging_points[i])

    #             if (used[min_index]): # If already taken, find the next closest manually
    #                 dists, idxs = tree.query(merging_points[i], k=len(non_merging_points))  # Get sorted neighbor
    #                 min_index = idxs[np.where(~used[idxs])[0][0]]  # Find first unused index

    #             # Store the match and mark as used
    #             closest_indices[i] = min_index
    #             used[min_index] = True  # Mark as used

    #         control_indices.append(closest_indices)
    #         #calculating the KS statistic
    #         D_mstar, p_Mstar = ks_2samp(pop['non_merging_population']['Mstar'][np.sort(closest_indices)],pop['merging_population']['Mstar'])
    #         D_z, p_z = ks_2samp(pop['non_merging_population']['z'][np.sort(closest_indices)],pop['merging_population']['z'])
    #         print(p_z,p_Mstar,np.shape(control_indices)[0])
    #         if p_z < matching_threshold or p_Mstar < matching_threshold or np.shape(control_indices)[0]>=10:
    #         # if p_z < 0.99 or p_Mstar < 0.99 or np.shape(control_indices)[0]>=10:
    #             break # Exit the loop if the conditions are no longer true

    #     return control_indices

    # def store_control_indices(self):
    #     np.savetxt(self.control_idx_file, np.array(self.control_indices, dtype=int))

    # def match_z_Mstar_plot(self,Mstar_binsize = 0.5,Mstar_min = 7,Mstar_max = 12,z_binsize = 0.6,z_min = 0,z_max = 5):

    #     Nbins_Ms = int((Mstar_max-Mstar_min)/Mstar_binsize)
    #     Mstar_bins = np.linspace(Mstar_min,Mstar_max,Nbins_Ms)

    #     Nbins_z = int((z_max - z_min) / z_binsize)
    #     z_bins = np.linspace(z_min, z_max, Nbins_z)

    #     # control_sample_ids = np.array(self.control_indices).flatten()

    #     fig,ax = plt.subplots(1,2,figsize=(10,4))
    #     ax[0].hist(self.z_control_pop, bins=z_bins, color="black", histtype="step",linewidth=2,density=True)
    #     ax[0].hist(self.z_merging_pop, bins=z_bins, histtype="step",color="Darkorange",linestyle="--",linewidth=2,density=True)
    #     ax[0].set_xlabel("z",fontsize=25)
    #     ax[0].set_ylabel("pdf",fontsize=25)
    #     # ax[0].hist(self.pop['non_merging_population']['z'][:][self.control_indices[self.valid_control_mask]], bins=z_bins, color="black", histtype="step",linewidth=2,density=True)
    #     # ax[0].hist(self.pop['merging_population']['z'][self.valid_control_mask], bins=z_bins, histtype="step",color="Darkorange",linestyle="--",linewidth=2,density=True)

    #     ax[1].hist(np.log10(self.Mstar_control_pop), bins=Mstar_bins,histtype="step",color="black",label="Control",linewidth=2,density=True)
    #     ax[1].hist(np.log10(self.Mstar_merging_pop),bins=Mstar_bins,histtype="step",label="PM",color="Darkorange",linestyle="--",linewidth=2,density=True)
    #     #ax[1].set_xticks([7,8,9,10,11,12])
    #     # ax[1].hist(np.log10(self.pop['non_merging_population']['Mstar'][:][self.control_indices[self.valid_control_mask]]), bins=Mstar_bins,histtype="step",color="black",label="Control",linewidth=2,density=True)
    #     # ax[1].hist(np.log10(self.pop['merging_population']['Mstar'][self.valid_control_mask]),bins=Mstar_bins,histtype="step",label="PM",color="Darkorange",linestyle="--",linewidth=2,density=True)
    #     #ax[1].set_xticks([7,8,9,10,11,12])
    #     ax[1].legend(fontsize=15)
    #     ax[1].set_xlabel("$\log(M_{\star}/M_{\odot})$",fontsize=25)

    #     #fig_name = fig_loc+"control-pm-z-Mstar-match.pdf"
    #     #fig.show()
    #     #fig.savefig(fig_name)
    #     #print("Figure saved in %s"%(fig_name))

    #     return fig,ax

    def compute_population_properties(self,verbose):

        self.subhalo_ids_mergers = self.pop['merging_population']['subhalo_ids'][:][self.valid_merger_indices]
        self.subhalo_ids_controls = self.pop['non_merging_population']['subhalo_ids'][:][self.valid_control_indices]

        self.z_merging_pop = self.pop['merging_population']['z'][:][self.valid_merger_indices]
        self.z_control_pop = self.pop['non_merging_population']['z'][:][self.valid_control_indices]

        self.snap_merging_pop = self.pop['merging_population']['snap'][:][self.valid_merger_indices]
        self.snap_control_pop = self.pop['non_merging_population']['snap'][:][self.valid_control_indices]

        self.Mstar_merging_pop = self.pop['merging_population']['Mstar'][:][self.valid_merger_indices]
        self.Mstar_control_pop = self.pop['non_merging_population']['Mstar'][:][self.valid_control_indices]

        self.Msubhalo_merging_pop = self.pop['merging_population']['Msubhalo'][:][self.valid_merger_indices]
        self.Msubhalo_control_pop = self.pop['non_merging_population']['Msubhalo'][:][self.valid_control_indices]

        self.MBH_merging_pop = self.pop['merging_population']['MBH'][:][self.valid_merger_indices]
        self.MBH_control_pop = self.pop['non_merging_population']['MBH'][:][self.valid_control_indices]

        self.SFR_merging_pop = self.pop['merging_population']['SFR'][:][self.valid_merger_indices]
        self.SFR_control_pop = self.pop['non_merging_population']['SFR'][:][self.valid_control_indices]

        self.Mdot_merging_pop = self.pop['merging_population']['Mdot'][:][self.valid_merger_indices]
        self.Mdot_control_pop = self.pop['non_merging_population']['Mdot'][:][self.valid_control_indices]

        self.sSFR_merging_pop = self.SFR_merging_pop/self.Mstar_merging_pop
        self.sSFR_control_pop = self.SFR_control_pop/self.Mstar_control_pop

        self.Mgas_merging_pop = self.pop['merging_population']['Mgas'][:][self.valid_merger_indices]
        self.Mgas_control_pop = self.pop['non_merging_population']['Mgas'][:][self.valid_control_indices]

        self.sBHAR_merging_pop = self.Mdot_merging_pop/self.MBH_merging_pop
        self.sBHAR_control_pop = self.Mdot_control_pop/self.MBH_control_pop

        self.StellarHalfmassRad_merging_pop = self.pop['merging_population']['StellarHalfmassRad'][:][self.valid_merger_indices]
        self.StellarHalfmassRad_control_pop = self.pop['non_merging_population']['StellarHalfmassRad'][:][self.valid_control_indices]

        self.q_merger = self.pop['merging_population']['q_merger'][:][self.valid_merger_indices]

        self.merger_progenitor_properties()

        self.MgasInRad = self.pop['merging_population']['MgasInRad'][:][self.valid_merger_indices]
        self.MstarInRad = self.pop['merging_population']['MstarInRad'][:][self.valid_merger_indices]
        self.fgas_post_merger = self.MgasInRad/(self.MgasInRad + self.MstarInRad)

        self.fgas_control = self.pop['non_merging_population']['MgasInRad'][:][self.valid_control_indices]/(self.pop['non_merging_population']['MgasInRad'][:][self.valid_control_indices] + self.pop['non_merging_population']['MstarInRad'][:][self.valid_control_indices])
        
        self.SubhaloPhotoMag_merging_pop = self.pop['merging_population']['SubhaloPhotoMag'][:][self.valid_merger_indices]
        self.SubhaloPhotoMag_control_pop = self.pop['non_merging_population']['SubhaloPhotoMag'][:][self.valid_control_indices]

        if verbose:
        #sSFR averages
            print("The average sSFR for merging galaxies is %1.3e"%(np.mean(self.sSFR_merging_pop)))
            print("The average sSFR for non-merging galaxies is %1.3e"%(np.mean(self.sSFR_control_pop)))
            print("The sSFR enhancement in post mergers is %1.3f"%(np.mean(self.sSFR_merging_pop)/np.mean(self.sSFR_control_pop)))

            #Mgas averages
            print("The average Mgas for merging galaxies is %1.3e" % (np.mean(self.Mgas_merging_pop)))
            print("The average Mgas for non-merging galaxies is %1.3e" % (np.mean(self.Mgas_control_pop)))
            print("The Mgas enhancement in post mergers is %1.3f" % (np.mean(self.Mgas_merging_pop) / np.mean(self.Mgas_control_pop)))
            
            # fgas averages
            print("The average fgas for merging galaxies is %1.3e" % (np.mean(self.fgas_merging_pop)))
            print("The average fgas for non-merging galaxies is %1.3e" % (np.mean(self.fgas_control_pop)))
            print("The fgas enhancement in post mergers is %1.3f" % (np.mean(self.fgas_merging_pop) / np.mean(self.fgas_control_pop)))
            
            # Mdot averages
            print("The average Mdot for merging galaxies is %1.3e" % (np.mean(self.Mdot_merging_pop)))
            print("The average Mdot for non-merging galaxies is %1.3e" % (np.mean(self.Mdot_control_pop)))
            print("The Mdot enhancement in post mergers is %1.3f" % (np.mean(self.Mdot_merging_pop) / np.mean(self.Mdot_control_pop)))

        return None

    def merger_progenitor_properties(self):
        MgasInRad_progs = self.pop['merging_population']['prog_MgasInRad'][:].reshape(self.N_mergers,2)
        self.MgasInRad_progs = MgasInRad_progs[self.valid_merger_indices]
        MstarInRad_progs = self.pop['merging_population']['prog_MstarInRad'][:].reshape(self.N_mergers,2)
        self.MstarInRad_progs = MstarInRad_progs[self.valid_merger_indices]
        self.fgas_progs = np.sum(self.MgasInRad_progs,axis=1)/(np.sum(self.MgasInRad_progs,axis=1)+np.sum(self.MstarInRad_progs,axis=1))

        StellarHalfmassRad_progs = self.pop['merging_population']['prog_StellarHalfmassRad'][:].reshape(self.N_mergers,2)
        self.StellarHalfmassRad_progs = StellarHalfmassRad_progs[self.valid_merger_indices]

        z_progs = self.pop['merging_population']['prog_redshift'][:].reshape(self.N_mergers,2)
        self.z_progs = z_progs[self.valid_merger_indices]
        return None
        
class control_sample_brahma:

    def __init__(self,pop_file,verbose=False,max_Mstar_tolerance=0.15, max_z_tolerance=0.3):

        #note that for brahma, the population file contains the merging and control population unlike the TNG files
        
        self.pop = pop_file
        self.N_mergers = len(self.pop['merging_population']['z'])

        #self.control_indices = self.find_control_samples()
        merger_control_index_pairs = self.find_control_samples(
        max_Mstar_tolerance=max_Mstar_tolerance,
        max_z_tolerance=max_z_tolerance)

        self.merger_control_index_pairs = np.array(merger_control_index_pairs)

        print("Number of cases where a close enough match is not found within the acceptable tolerance:",np.sum(np.array(self.merger_control_index_pairs)[:,1]==-1))

        self.MBH_not_zero_flag = self.pop['merging_population']['MBH'][:][self.merger_control_index_pairs[:,0]]!=0
        self.Ngas_min_flag = self.pop['merging_population']['SubhaloLenType'][:,0][:][self.merger_control_index_pairs[:,0]]>10
        self.control_available_flag = self.merger_control_index_pairs[:,1]!=-1
        self.valid_control_mask  = self.MBH_not_zero_flag&self.Ngas_min_flag&self.control_available_flag

        self.N_mergers_w_controls = np.sum(self.valid_control_mask)
        print("number of processable mergers with valid controls:")
        print(self.N_mergers_w_controls)

        self.valid_merger_indices = self.merger_control_index_pairs[self.valid_control_mask,0]
        self.valid_control_indices = self.merger_control_index_pairs[self.valid_control_mask,1]

        #filter out only the unique post mergers. Some post merger galaxies are counted multiplle times as they are from progenitors of multiple galaxy mergers.
        Mstar_vals = self.pop['merging_population']['Mstar'][:][self.valid_merger_indices]
        values, counts = np.unique(Mstar_vals, return_counts=True)
        duplicate_values = values[counts > 1]
        

        #let's also mask the cases where the post merger galaxy is formed as a result of multiple galaxy mergers
        multiple_merger_mask = np.isin(Mstar_vals, duplicate_values)
        self.multi_count_merger_indices = self.valid_merger_indices[multiple_merger_mask]
        self.multi_count_control_indices = self.valid_control_indices[multiple_merger_mask]


        unique_mask = ~np.isin(Mstar_vals, duplicate_values)
        self.valid_merger_indices = self.valid_merger_indices[unique_mask]
        self.valid_control_indices = self.valid_control_indices[unique_mask]


        self.compute_population_properties(verbose)
        self.mini_merger_mask = self.q_merger < 0.1
        self.minor_merger_mask = (self.q_merger >= 0.1) & (self.q_merger < 0.25)
        self.major_merger_mask = self.q_merger >= 0.25

    # def find_control_samples(self):
    #     brahma_mergers_Mstar = self.pop['merging_population']['Mstar']
    #     brahma_mergers_z = self.pop['merging_population']['z']

    #     brahma_nonmergers_Mstar = self.pop['non_merging_population']['Mstar']
    #     brahma_nonmergers_z = self.pop['non_merging_population']['z']

    #     merging_points = np.column_stack((brahma_mergers_z,np.log10(brahma_mergers_Mstar)))
    #     non_merging_points = np.column_stack((brahma_nonmergers_z,np.log10(brahma_nonmergers_Mstar)))
    #     tree = cKDTree(non_merging_points)
    #     used = np.zeros(len(non_merging_points), dtype=bool)

    #     control_indices = []

    #     while True:
    #         closest_indices = np.full(len(merging_points), -1)
    #         tolerances = []
    #         for i in tqdm(range(len(merging_points)), desc="Processing merging points", ncols=100):
    #             #find the closest neibhour 
    #             dist, min_idx = tree.query(merging_points[i])
    #             if(used[min_idx]):
    #                 dists,idxs = tree.query(merging_points[i],k=len(non_merging_points))
    #                 min_idx = idxs[np.where(~used[idxs])[0][0]] 

    #             #check for tolerance:
    #             del_z = np.abs(merging_points[i][0]-non_merging_points[min_idx][0])
    #             dex_Mstar = np.abs(np.log10(merging_points[i][1]/non_merging_points[min_idx][1]))

    #             z_tolerance = 0.1
    #             Mstar_dex_tolerance = 0.01

    #             while True:
    #                 if(del_z<z_tolerance and dex_Mstar<Mstar_dex_tolerance):
    #                     used[min_idx] = True
    #                     closest_indices[i] = min_idx
    #                     tolerances.append((z_tolerance, Mstar_dex_tolerance))
    #                     break
    #                 else:
    #                     closest_indices[i] = -1
    #                     break

    #         control_indices.append(closest_indices)
    #             #D_mstar, p_Mstar = ks_2samp(pop['non_merging_population']['Mstar'][np.sort(closest_indices)],pop['merging_population']['Mstar'])
    #             #D_z, p_z = ks_2samp(pop['non_merging_population']['z'][np.sort(closest_indices)],pop['merging_population']['z'])
                
    #             #print(p_z,p_Mstar,np.shape(control_indices)[0])
    #             #print(z_tolerance,Mstar_dex_tolerance)

    #         if np.shape(control_indices)[0]>=1:
    #             break
    #             #only selecting one control sample for each merging galaxy for now

    #     return control_indices

    # def find_control_samples_strict_v2(self,max_Mstar_tolerance=0.6,max_z_tolerance=0.3):

    #     all_mrgr_z = np.unique(self.pop['merging_population']['z'][:]) #all unique redshifts where BHs/galaxies are merging
    #     merging_pop_Mstar = self.pop['merging_population']['Mstar'][:]
    #     non_merging_pop_Mstar = self.pop['non_merging_population']['Mstar'][:]

    #     merger_control_index_pairs = []
    #     used = np.zeros(len(non_merging_pop_Mstar),dtype=bool)
    #     starting_Mstar_tol = 0.1

    #     for i, z_i in enumerate(tqdm(all_mrgr_z, "processing each merger redshifts for controls")):
    #         zi_merger_ix = np.where(self.pop['merging_population']['z'] == z_i)[0]          # global indices into merging pop
    #         zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z'] == z_i)[0]    # global indices into non-merging pop
    #         zi_nonmerger_ix = zi_nonmrgr_ix[~used[zi_nonmrgr_ix]]                         # available candidates (global indices)

    #         merger_Mstars = self.pop['merging_population']['Mstar'][zi_merger_ix]
    #         # note: we will read candidate masses from pop inside helper so they are always in sync
    #         starting_Mstar_tol = starting_Mstar_tol  # keep existing name in scope
    #         max_z_tolerance = max_z_tolerance

    #         def _try_match(candidate_global_idx_array, merger_mass):
    #             """Return global index of matched non-merger or -1."""
    #             if candidate_global_idx_array.size == 0:
    #                 return -1
    #             cand_masses = self.pop['non_merging_population']['Mstar'][candidate_global_idx_array]
    #             best_local = np.argmin(np.abs(cand_masses - merger_mass))
    #             best_global = candidate_global_idx_array[best_local]
    #             mass_diff = np.abs(np.log(cand_masses[best_local]) - np.log(merger_mass))
    #             return best_global if mass_diff <= starting_Mstar_tol else -1

    #         # loop over mergers at this redshift (use global merger index directly)
    #         for merger_global_idx, merger_mass in zip(zi_merger_ix, merger_Mstars):
    #             matched_non_global = _try_match(zi_nonmerger_ix, merger_mass)

    #             # if not found, try next redshift (if within tolerance) then previous
    #             if matched_non_global == -1 and i < len(all_mrgr_z) - 1:
    #                 next_non_idx = np.where(self.pop['non_merging_population']['z'] == all_mrgr_z[i + 1])[0]
    #                 next_candidates = next_non_idx[~used[next_non_idx]]
    #                 if next_candidates.size and (all_mrgr_z[i + 1] - all_mrgr_z[i]) <= max_z_tolerance:
    #                     matched_non_global = _try_match(next_candidates, merger_mass)

    #             if matched_non_global == -1 and i > 0:
    #                 prev_non_idx = np.where(self.pop['non_merging_population']['z'] == all_mrgr_z[i - 1])[0]
    #                 prev_candidates = prev_non_idx[~used[prev_non_idx]]
    #                 if prev_candidates.size and (all_mrgr_z[i] - all_mrgr_z[i - 1]) <= max_z_tolerance:
    #                     matched_non_global = _try_match(prev_candidates, merger_mass)

    #             if matched_non_global != -1:
    #                 # use global indices directly (no expensive np.where on values)
    #                 merger_control_index_pairs.append([merger_global_idx, matched_non_global])
    #                 used[matched_non_global] = True
    #             else:
    #                 merger_control_index_pairs.append([merger_global_idx, -1])

    #     return  np.array(merger_control_index_pairs)
    
    def find_control_samples(self, max_Mstar_tolerance, max_z_tolerance):
        mrgr_z = self.pop['merging_population']['z'][:]
        mrgr_Mstar = self.pop['merging_population']['Mstar'][:]
        non_mrgr_z = self.pop['non_merging_population']['z'][:]
        non_mrgr_Mstar = self.pop['non_merging_population']['Mstar'][:]
        
        used = np.zeros(len(non_mrgr_Mstar), dtype=bool)
        merger_control_index_pairs = []

        for merger_idx in tqdm(range(len(mrgr_z)), "finding controls"):
            z_i = mrgr_z[merger_idx]
            mass_i = mrgr_Mstar[merger_idx]
            candidates = np.where((np.abs(non_mrgr_z - z_i) <= max_z_tolerance) & ~used)[0]
            
            matched = -1
            if candidates.size > 0 and mass_i > 0:
                cand_masses = non_mrgr_Mstar[candidates]
                valid = cand_masses > 0
                if valid.any():
                    log_diffs = np.abs(np.log(cand_masses[valid]) - np.log(mass_i))
                    if log_diffs.min() <= max_Mstar_tolerance:
                        matched = candidates[valid][np.argmin(log_diffs)]
            
            merger_control_index_pairs.append([merger_idx, matched])
            if matched != -1:
                used[matched] = True

        return np.array(merger_control_index_pairs)

    def compute_population_properties(self,verbose=False):

        self.subhalo_ids_mergers = self.pop['merging_population']['subhalo_ids'][:][self.valid_merger_indices]
        self.subhalo_ids_controls = self.pop['non_merging_population']['subhalo_ids'][:][self.valid_control_indices]

        self.z_merging_pop = self.pop['merging_population']['z'][:][self.valid_merger_indices]
        self.z_control_pop = self.pop['non_merging_population']['z'][:][self.valid_control_indices]

        self.snap_merging_pop = self.pop['merging_population']['snap'][:][self.valid_merger_indices]
        self.snap_control_pop = self.pop['non_merging_population']['snap'][:][self.valid_control_indices]

        self.Mstar_merging_pop = self.pop['merging_population']['Mstar'][:][self.valid_merger_indices]
        self.Mstar_control_pop = self.pop['non_merging_population']['Mstar'][:][self.valid_control_indices]

        self.Msubhalo_merging_pop = self.pop['merging_population']['Msubhalo'][:][self.valid_merger_indices]
        self.Msubhalo_control_pop = self.pop['non_merging_population']['Msubhalo'][:][self.valid_control_indices]
        
        self.MBH_merging_pop = self.pop['merging_population']['MBH'][:][self.valid_merger_indices]
        self.MBH_control_pop = self.pop['non_merging_population']['MBH'][:][self.valid_control_indices]

        self.MBH_massive_merging_pop = self.pop['merging_population']['MBH_massive'][:][self.valid_merger_indices]
        self.MBH_massive_control_pop = self.pop['non_merging_population']['MBH_massive'][:][self.valid_control_indices]

        self.MBH_luminous_merging_pop = self.pop['merging_population']['MBH_luminous'][:][self.valid_merger_indices]
        self.MBH_luminous_control_pop = self.pop['non_merging_population']['MBH_luminous'][:][self.valid_control_indices]

        self.SFR_merging_pop = self.pop['merging_population']['SFR'][:][self.valid_merger_indices]
        self.SFR_control_pop = self.pop['non_merging_population']['SFR'][:][self.valid_control_indices]

        self.Mdot_merging_pop = self.pop['merging_population']['Mdot'][:][self.valid_merger_indices]
        self.Mdot_control_pop = self.pop['non_merging_population']['Mdot'][:][self.valid_control_indices]

        self.Mdot_massive_merging_pop = self.pop['merging_population']['Mdot_massive'][:][self.valid_merger_indices]
        self.Mdot_massive_control_pop = self.pop['non_merging_population']['Mdot_massive'][:][self.valid_control_indices]

        self.Mdot_luminous_merging_pop = self.pop['merging_population']['Mdot_luminous'][:][self.valid_merger_indices]
        self.Mdot_luminous_control_pop = self.pop['non_merging_population']['Mdot_luminous'][:][self.valid_control_indices]

        self.sSFR_merging_pop = self.SFR_merging_pop/self.Mstar_merging_pop
        self.sSFR_control_pop = self.SFR_control_pop/self.Mstar_control_pop

        self.Mgas_merging_pop = self.pop['merging_population']['Mgas'][:][self.valid_merger_indices]
        self.Mgas_control_pop = self.pop['non_merging_population']['Mgas'][:][self.valid_control_indices]

        self.fgas_merging_pop = self.Mgas_merging_pop/(self.Mgas_merging_pop+self.Mstar_merging_pop)
        self.fgas_control_pop = self.Mgas_control_pop/(self.Mgas_control_pop+self.Mstar_control_pop)

        self.sBHAR_merging_pop = self.Mdot_merging_pop/self.MBH_merging_pop
        self.sBHAR_control_pop = self.Mdot_control_pop/self.MBH_control_pop


        self.sBHAR_merging_massive_pop = self.Mdot_massive_merging_pop/self.MBH_massive_merging_pop
        self.sBHAR_control_massive_pop = self.Mdot_massive_control_pop/self.MBH_massive_control_pop

        self.sBHAR_merging_luminous_pop = self.Mdot_luminous_merging_pop/self.MBH_luminous_merging_pop
        self.sBHAR_control_luminous_pop = self.Mdot_luminous_control_pop/self.MBH_luminous_control_pop

        self.StellarHalfmassRad_merging_pop = self.pop['merging_population']['StellarHalfmassRad'][:][self.valid_merger_indices]
        self.StellarHalfmassRad_control_pop = self.pop['non_merging_population']['StellarHalfmassRad'][:][self.valid_control_indices]

        self.q_merger = self.pop['merging_population']['q_merger'][:][self.valid_merger_indices]

        self.merger_progenitor_properties()

        self.MgasInRad = self.pop['merging_population']['MgasInRad'][:][self.valid_merger_indices]
        self.MstarInRad = self.pop['merging_population']['MstarInRad'][:][self.valid_merger_indices]
        self.fgas_post_merger = self.MgasInRad/(self.MgasInRad + self.MstarInRad)

        self.fgas_control = self.pop['non_merging_population']['MgasInRad'][:][self.valid_control_indices]/(self.pop['non_merging_population']['MgasInRad'][:][self.valid_control_indices] + self.pop['non_merging_population']['MstarInRad'][:][self.valid_control_indices])
        
        self.SubhaloPhotoMag_merging_pop = self.pop['merging_population']['SubhaloPhotoMag'][:][self.valid_merger_indices]
        self.SubhaloPhotoMag_control_pop = self.pop['non_merging_population']['SubhaloPhotoMag'][:][self.valid_control_indices]

        if verbose:
        #sSFR averages
            print("The average sSFR for merging galaxies is %1.3e"%(np.mean(self.sSFR_merging_pop)))
            print("The average sSFR for non-merging galaxies is %1.3e"%(np.mean(self.sSFR_control_pop)))
            print("The sSFR enhancement in post mergers is %1.3f"%(np.mean(self.sSFR_merging_pop)/np.mean(self.sSFR_control_pop)))

            #Mgas averages
            print("The average Mgas for merging galaxies is %1.3e" % (np.mean(self.Mgas_merging_pop)))
            print("The average Mgas for non-merging galaxies is %1.3e" % (np.mean(self.Mgas_control_pop)))
            print("The Mgas enhancement in post mergers is %1.3f" % (np.mean(self.Mgas_merging_pop) / np.mean(self.Mgas_control_pop)))
            
            # fgas averages
            print("The average fgas for merging galaxies is %1.3e" % (np.mean(self.fgas_merging_pop)))
            print("The average fgas for non-merging galaxies is %1.3e" % (np.mean(self.fgas_control_pop)))
            print("The fgas enhancement in post mergers is %1.3f" % (np.mean(self.fgas_merging_pop) / np.mean(self.fgas_control_pop)))
            
            # Mdot averages
            print("The average Mdot for merging galaxies is %1.3e" % (np.mean(self.Mdot_merging_pop)))
            print("The average Mdot for non-merging galaxies is %1.3e" % (np.mean(self.Mdot_control_pop)))
            print("The Mdot enhancement in post mergers is %1.3f" % (np.mean(self.Mdot_merging_pop) / np.mean(self.Mdot_control_pop)))

        return None

    def merger_progenitor_properties(self):
        MgasInRad_progs = self.pop['merging_population']['prog_MgasInRad'][:].reshape(self.N_mergers,2)
        self.MgasInRad_progs = MgasInRad_progs[self.valid_merger_indices]
        MstarInRad_progs = self.pop['merging_population']['prog_MstarInRad'][:].reshape(self.N_mergers,2)
        self.MstarInRad_progs = MstarInRad_progs[self.valid_merger_indices]
        self.fgas_progs = np.sum(self.MgasInRad_progs,axis=1)/(np.sum(self.MgasInRad_progs,axis=1)+np.sum(self.MstarInRad_progs,axis=1))

        StellarHalfmassRad_progs = self.pop['merging_population']['prog_StellarHalfmassRad'][:].reshape(self.N_mergers,2)
        self.StellarHalfmassRad_progs = StellarHalfmassRad_progs[self.valid_merger_indices]
        z_progs = self.pop['merging_population']['prog_redshift'][:].reshape(self.N_mergers,2)
        self.z_progs = z_progs[self.valid_merger_indices]
        return None
    
    # def plot_PM_and_control_histograms(self, bin_settings=None):
            
    #     # Default bin settings if none are provided
    #         if bin_settings is None:
    #             bin_settings = {
    #             'sSFR': {'binsize': 0.2, 'bin_min': -14, 'bin_max': -7},
    #             'Mdot': {'binsize': 0.2, 'bin_min': -8, 'bin_max': 1},
    #             'Mgas_half': {'binsize': 0.2, 'bin_min': 5, 'bin_max': 14},
    #             'fgas': {'binsize': 0.05, 'bin_min': 0, 'bin_max': 1}
    #             }

    #         properties = {
    #             'sSFR': (np.log10(self.sSFR_merging_pop[self.sSFR_merging_pop>0]), np.log10(self.sSFR_control_pop[self.sSFR_control_pop>0])),
    #             'Mdot': (np.log10(self.Mdot_merging_pop[self.Mdot_merging_pop>0]), np.log10(self.Mdot_control_pop[self.Mdot_control_pop>0])),
    #             'Mgas_half': (np.log10(self.Mgas_half_merging_pop), np.log10(self.Mgas_half_control_pop)),
    #             'fgas': (self.fgas_merging_pop, self.fgas_control_pop)
    #         }

    #         properties_xlabel = {
    #             'sSFR': r"$\log_{10}(\mathrm{sSFR}[\mathrm{yr}^{-1}])$",
    #             'Mdot': r"$\log_{10}(\dot{M}_{\mathrm{BH}}[M_{\odot}\, \mathrm{yr}^{-1}])$",
    #             'Mgas_half': r"$\log_{10}(M_{\mathrm{gas}}[M_{\odot}])$",
    #             'fgas': r"$f_{\mathrm{gas}}$",
    #             }
            
    #         self.set_plot_style()
    #         fig, axes = plt.subplots(2, 2, figsize=(9,4))
    #         axes = axes.flatten()

    #         for i, (prop_name, (prop_merging, prop_control)) in enumerate(properties.items()):
        
    #             binsize = bin_settings[prop_name].get('binsize')
    #             bin_min = bin_settings[prop_name].get('bin_min')
    #             bin_max = bin_settings[prop_name].get('bin_max')
    #             bins = np.arange(bin_min, bin_max + binsize, binsize)
            
    #             # Plot histograms
    #             axes[i].hist(prop_merging, bins=bins,color='dodgerblue', label='PM', density=True,histtype="step", linewidth=2)
    #             axes[i].hist(prop_control, bins=bins,color='orange', label='Control', density=True,histtype="step", linewidth=2)
    #             axes[i].set_xlabel(properties_xlabel[prop_name])
    #             axes[i].set_ylabel('pdf')
    #             axes[i].legend()
            
    #         fig.tight_layout()
    #         #fig.show()
    #         return axes,fig
    

def load_pop_file(basePath,pop_file_path,minN_values):

    simName = basePath.split('/')[-2]
    pop_file_name = pop_file_path+ f"{simName}_population_sort_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5"
    #pop_file_name = pop_file_path+'population_sort'+'_gas-'+f'{minN_values[0]:03d}'+'_dm-'+f'{minN_values[1]:03d}'+'_star-'+f'{minN_values[2]:03d}'+'_bh-'+f'{minN_values[3]:03d}'+'_brahma.hdf5'
    return h5py.File(pop_file_name,'r')


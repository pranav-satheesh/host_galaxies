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



#default values
# plt.rcParams.update({'font.size': 25})
# plt.rcParams.update({'xtick.labelsize': 15, 'ytick.labelsize': 15})


# plt.rcParams.update({
#             'lines.linewidth': 3,
#             'axes.labelsize': 25,
#             'axes.titlesize': 25,
#             'xtick.labelsize': 25,
#             'ytick.labelsize': 25,
#             'legend.fontsize': 20
#         })


#TO-DO
# - Return the relative seperations for merging galaxies and control
# Add a user input to the maximum tolerances allowed for z and Mstar
# Function to compute KS statistic or Hellinger distances

class control_samples_TNG:

    def __init__(self,population_file,verbose=False):

        self.pop = population_file
        self.N_mergers = len(self.pop['merging_population']['z'])

        self.merger_control_index_pairs = self.find_control_samples_strict_v2()
        print(f"Number of available mergers in this population is {self.N_mergers:03d}")
        print("Number of cases where a close enough match is not found within the acceptable tolerance:",np.sum(self.merger_control_index_pairs[:,1] == -1))


        self.MBH_not_zero_flag = self.pop['merging_population']['MBH'][:][self.merger_control_index_pairs[:,0]]!=0
        self.control_available_flag = self.merger_control_index_pairs[:,1]!=-1
        self.valid_control_mask  = self.MBH_not_zero_flag&self.control_available_flag

        self.compute_population_properties(verbose)
        self.major_merger_mask = self.q_merger >= 0.1
        self.major_major_merger_mask = self.q_merger >= 0.25

    def find_control_samples_strict(self):

        all_mrgr_z = np.unique(self.pop['merging_population']['z'][:]) #all unique redshifts where BHs/galaxies are merging
        merging_pop_Mstar = self.pop['merging_population']['Mstar'][:]
        non_merging_pop_Mstar = self.pop['non_merging_population']['Mstar'][:]

        merger_control_index_pairs = []
        used = np.zeros(len(non_merging_pop_Mstar),dtype=bool)

        for z_i in tqdm(all_mrgr_z,"processing each merger redshifts for controls"):
            zi_merger_ix = np.where(self.pop['merging_population']['z']==z_i)[0]
            zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z']==z_i)[0]
            zi_nonmerger_ix = zi_nonmrgr_ix[used[zi_nonmrgr_ix]==False]

            merger_Mstars = self.pop['merging_population']['Mstar'][zi_merger_ix]
            nonmerger_Mstars = self.pop['non_merging_population']['Mstar'][zi_nonmerger_ix]

            for Mstar_merger_i in merger_Mstars:
                 closest_non_merger_ix = np.argmin(np.abs(nonmerger_Mstars - Mstar_merger_i))
                 mass_diff = np.abs(np.log(nonmerger_Mstars[closest_non_merger_ix]) - np.log(Mstar_merger_i))
                
                 if mass_diff <= 0.1:
                     merger_index = np.where(merging_pop_Mstar==Mstar_merger_i)[0][0]
                     non_merger_index = np.where(non_merging_pop_Mstar==nonmerger_Mstars[closest_non_merger_ix])[0][0]
                     merger_control_index_pairs.append([merger_index,non_merger_index])
                     used[non_merger_index] = True #mark this non-merging galaxy as used
                 else:
                     merger_control_index_pairs.append([merger_index,-1]) #no suitable control found

        return  np.array(merger_control_index_pairs)

    def find_control_samples_strict_v2(self,max_Mstar_tolerance=0.2,max_z_tolerance=0.1):

        all_mrgr_z = np.unique(self.pop['merging_population']['z'][:]) #all unique redshifts where BHs/galaxies are merging
        merging_pop_Mstar = self.pop['merging_population']['Mstar'][:]
        non_merging_pop_Mstar = self.pop['non_merging_population']['Mstar'][:]

        merger_control_index_pairs = []
        used = np.zeros(len(non_merging_pop_Mstar),dtype=bool)
        starting_Mstar_tol = 0.1

        for i, z_i in enumerate(tqdm(all_mrgr_z, "processing each merger redshifts for controls")):
            zi_merger_ix = np.where(self.pop['merging_population']['z'] == z_i)[0]          # global indices into merging pop
            zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z'] == z_i)[0]    # global indices into non-merging pop
            zi_nonmerger_ix = zi_nonmrgr_ix[~used[zi_nonmrgr_ix]]                         # available candidates (global indices)

            merger_Mstars = self.pop['merging_population']['Mstar'][zi_merger_ix]
            # note: we will read candidate masses from pop inside helper so they are always in sync
            starting_Mstar_tol = starting_Mstar_tol  # keep existing name in scope
            max_z_tolerance = max_z_tolerance

            def _try_match(candidate_global_idx_array, merger_mass):
                """Return global index of matched non-merger or -1."""
                if candidate_global_idx_array.size == 0:
                    return -1
                cand_masses = self.pop['non_merging_population']['Mstar'][candidate_global_idx_array]
                best_local = np.argmin(np.abs(cand_masses - merger_mass))
                best_global = candidate_global_idx_array[best_local]
                mass_diff = np.abs(np.log(cand_masses[best_local]) - np.log(merger_mass))
                return best_global if mass_diff <= max_Mstar_tolerance else -1

            # loop over mergers at this redshift (use global merger index directly)
            for merger_global_idx, merger_mass in zip(zi_merger_ix, merger_Mstars):
                matched_non_global = _try_match(zi_nonmerger_ix, merger_mass)

                # if not found, try next redshift (if within tolerance) then previous
                if matched_non_global == -1 and i < len(all_mrgr_z) - 1:
                    next_non_idx = np.where(self.pop['non_merging_population']['z'] == all_mrgr_z[i + 1])[0]
                    next_candidates = next_non_idx[~used[next_non_idx]]
                    if next_candidates.size and (all_mrgr_z[i + 1] - all_mrgr_z[i]) <= max_z_tolerance:
                        matched_non_global = _try_match(next_candidates, merger_mass)

                if matched_non_global == -1 and i > 0:
                    prev_non_idx = np.where(self.pop['non_merging_population']['z'] == all_mrgr_z[i - 1])[0]
                    prev_candidates = prev_non_idx[~used[prev_non_idx]]
                    if prev_candidates.size and (all_mrgr_z[i] - all_mrgr_z[i - 1]) <= max_z_tolerance:
                        matched_non_global = _try_match(prev_candidates, merger_mass)

                if matched_non_global != -1:
                    # use global indices directly (no expensive np.where on values)
                    merger_control_index_pairs.append([merger_global_idx, matched_non_global])
                    used[matched_non_global] = True
                else:
                    merger_control_index_pairs.append([merger_global_idx, -1])

        return  np.array(merger_control_index_pairs)
    
    def find_control_samples_strict_v3(self,max_Mstar_tolerance=0.6,max_z_tolerance=0.3):

        all_mrgr_z = np.unique(self.pop['merging_population']['z'][:]) #all unique redshifts where BHs/galaxies are merging
        merging_pop_Mstar = self.pop['merging_population']['Mstar'][:]
        non_merging_pop_Mstar = self.pop['non_merging_population']['Mstar'][:]

        merger_control_index_pairs = []
        used = np.zeros(len(non_merging_pop_Mstar),dtype=bool)
        starting_Mstar_tol = 0.1

        for i,z_i in enumerate(tqdm(all_mrgr_z,"processing each merger redshifts for controls")):
            zi_merger_ix = np.where(self.pop['merging_population']['z']==z_i)[0]
            zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z']==z_i)[0]
            zi_nonmerger_ix = zi_nonmrgr_ix[used[zi_nonmrgr_ix]==False]

            merger_Mstars = self.pop['merging_population']['Mstar'][zi_merger_ix]
            nonmerger_Mstars = self.pop['non_merging_population']['Mstar'][zi_nonmerger_ix]

            for Mstar_merger_i in merger_Mstars:
                closest_non_merger_ix = np.argmin(np.abs(nonmerger_Mstars - Mstar_merger_i))
                mass_diff = np.abs(np.log(nonmerger_Mstars[closest_non_merger_ix]) - np.log(Mstar_merger_i))

                if mass_diff <= starting_Mstar_tol:
                    merger_index = np.where(merging_pop_Mstar==Mstar_merger_i)[0][0]
                    non_merger_index = np.where(non_merging_pop_Mstar==nonmerger_Mstars[closest_non_merger_ix])[0][0]
                    merger_control_index_pairs.append([merger_index,non_merger_index])
                    used[non_merger_index] = True #mark this non-merging galaxy as used
                else:
                    

                    zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z']==all_mrgr_z[i+1])[0]
                    print(all_mrgr_z[i],all_mrgr_z[i+1],all_mrgr_z[i+1]-all_mrgr_z[i])
                    zi_nonmerger_ix = zi_nonmrgr_ix[used[zi_nonmrgr_ix]==False]
                    diff_z = all_mrgr_z[i+1]-all_mrgr_z[i]

                    if zi_nonmerger_ix.size != 0 and np.abs(diff_z)<= max_z_tolerance:
                        nonmerger_Mstars = self.pop['non_merging_population']['Mstar'][zi_nonmerger_ix]
                        closest_non_merger_ix = np.argmin(np.abs(nonmerger_Mstars - Mstar_merger_i))
                        mass_diff = np.abs(np.log(nonmerger_Mstars[closest_non_merger_ix]) - np.log(Mstar_merger_i))
                        if mass_diff <= starting_Mstar_tol:
                            merger_index = np.where(merging_pop_Mstar==Mstar_merger_i)[0][0]
                            non_merger_index = np.where(non_merging_pop_Mstar==nonmerger_Mstars[closest_non_merger_ix])[0][0]
                            merger_control_index_pairs.append([merger_index,non_merger_index])
                            used[non_merger_index] = True #mark this non-merging galaxy as used
                    elif i>0:
                        zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z']==all_mrgr_z[i-1])[0]
                        zi_nonmerger_ix = zi_nonmrgr_ix[used[zi_nonmrgr_ix]==False]
                        diff_z = all_mrgr_z[i]-all_mrgr_z[i-1]
                        if zi_nonmerger_ix.size != 0 and np.abs(diff_z)<= max_z_tolerance:
                            nonmerger_Mstars = self.pop['non_merging_population']['Mstar'][zi_nonmerger_ix]
                            closest_non_merger_ix = np.argmin(np.abs(nonmerger_Mstars - Mstar_merger_i))
                            mass_diff = np.abs(np.log(nonmerger_Mstars[closest_non_merger_ix]) - np.log(Mstar_merger_i))
                            if mass_diff <= starting_Mstar_tol:
                                merger_index = np.where(merging_pop_Mstar==Mstar_merger_i)[0][0]
                                non_merger_index = np.where(non_merging_pop_Mstar==nonmerger_Mstars[closest_non_merger_ix])[0][0]
                                merger_control_index_pairs.append([merger_index,non_merger_index])
                                used[non_merger_index] = True #mark this non-merging galaxy as used
                        else:
                            merger_control_index_pairs.append([merger_index,-1]) #no suitable control found 

        return  np.array(merger_control_index_pairs)
    
    def find_control_samples(self,pop,max_z_tolerance=0.6,max_Mstar_dex_tolerance=0.6):

        merging_points = np.column_stack((pop['merging_population']['z'], np.log10(pop['merging_population']['Mstar'])))
        non_merging_points = np.column_stack((pop['non_merging_population']['z'], np.log10(pop['non_merging_population']['Mstar'])))

        tree = cKDTree(non_merging_points)
        used = np.zeros(len(non_merging_points), dtype=bool)
        control_indices = []
        net_tolerances = []

        while True:
            closest_indices = np.full(len(merging_points), -1)
            tolerances = []
            for i in tqdm(range(len(merging_points)), desc="Processing merging points", ncols=100):
             #find the closest neibhour 
                dist, min_idx = tree.query(merging_points[i])
                if(used[min_idx]):
                    #if the closest neighbouring non merging galaxy is already matched to a merging galaxy, find the next closest one
                    dists,idxs = tree.query(merging_points[i],k=len(non_merging_points))
                    min_idx = idxs[np.where(~used[idxs])[0][0]] 
                #check for tolerance:
                del_z = np.abs(merging_points[i][0]-non_merging_points[min_idx][0])
                dex_Mstar = np.abs(np.log10(merging_points[i][1])-np.log10(non_merging_points[min_idx][1]))

                z_tolerance = 0.05
                Mstar_dex_tolerance = 0.1

                
                while True:
                    if(del_z < z_tolerance and dex_Mstar < Mstar_dex_tolerance):
                        used[min_idx] = True
                        closest_indices[i] = min_idx
                        tolerances.append((del_z, dex_Mstar))
                        break
                    else:
                        z_tolerance = z_tolerance*1.5
                        Mstar_dex_tolerance = Mstar_dex_tolerance*1.5

                        if(z_tolerance>max_z_tolerance or Mstar_dex_tolerance>max_Mstar_dex_tolerance):
                            closest_indices[i] = -1
                            break

            control_indices.append(closest_indices)
            net_tolerances.append(tolerances)

            if np.shape(control_indices)[0]>=1:
                break

        return control_indices, net_tolerances

    def find_control_sample_indices(self,pop,matching_threshold=0.99):
        merging_points = np.column_stack((pop['merging_population']['z'], np.log10(pop['merging_population']['Mstar'])))
        non_merging_points = np.column_stack((pop['non_merging_population']['z'], np.log10(pop['non_merging_population']['Mstar'])))

        #matching the merging and non merging population in the z-Mstar plane

        # Build a KDTree for fast nearest neighbor search
        tree = cKDTree(non_merging_points)

        # Track used indices
        used = np.zeros(len(non_merging_points), dtype=bool)  # False means available

        control_indices = []
        p_z = 1.0
        p_Mstar = 1.0

        while True:
            closest_indices = np.full(len(merging_points), -1)  # Store assigned indices

            for i in tqdm(range(len(merging_points)), desc="Processing merging points", ncols=100):
            #for i in range(len(merging_points)):
            # Find the closest available neighbor
                d, min_index = tree.query(merging_points[i])

                if (used[min_index]): # If already taken, find the next closest manually
                    dists, idxs = tree.query(merging_points[i], k=len(non_merging_points))  # Get sorted neighbor
                    min_index = idxs[np.where(~used[idxs])[0][0]]  # Find first unused index

                # Store the match and mark as used
                closest_indices[i] = min_index
                used[min_index] = True  # Mark as used

            control_indices.append(closest_indices)
            #calculating the KS statistic
            D_mstar, p_Mstar = ks_2samp(pop['non_merging_population']['Mstar'][np.sort(closest_indices)],pop['merging_population']['Mstar'])
            D_z, p_z = ks_2samp(pop['non_merging_population']['z'][np.sort(closest_indices)],pop['merging_population']['z'])
            print(p_z,p_Mstar,np.shape(control_indices)[0])
            if p_z < matching_threshold or p_Mstar < matching_threshold or np.shape(control_indices)[0]>=10:
            # if p_z < 0.99 or p_Mstar < 0.99 or np.shape(control_indices)[0]>=10:
                break # Exit the loop if the conditions are no longer true

        return control_indices

    def store_control_indices(self):
        np.savetxt(self.control_idx_file, np.array(self.control_indices, dtype=int))

    def set_plot_style(self, linewidth=3, titlesize=20,labelsize=25,ticksize=20,legendsize=20):
        """Set matplotlib rcParams for consistent plot style."""
        plt.rcParams.update({
            'lines.linewidth': linewidth,
            'axes.labelsize': labelsize,
            'axes.titlesize': titlesize,
            'xtick.labelsize': ticksize,
            'ytick.labelsize': ticksize,
            'legend.fontsize': legendsize
        })

    def match_z_Mstar_plot(self,Mstar_binsize = 0.5,Mstar_min = 7,Mstar_max = 12,z_binsize = 0.6,z_min = 0,z_max = 5):

        Nbins_Ms = int((Mstar_max-Mstar_min)/Mstar_binsize)
        Mstar_bins = np.linspace(Mstar_min,Mstar_max,Nbins_Ms)

        Nbins_z = int((z_max - z_min) / z_binsize)
        z_bins = np.linspace(z_min, z_max, Nbins_z)

        # control_sample_ids = np.array(self.control_indices).flatten()

        fig,ax = plt.subplots(1,2,figsize=(10,4))
        ax[0].hist(self.z_control_pop, bins=z_bins, color="black", histtype="step",linewidth=2,density=True)
        ax[0].hist(self.z_merging_pop, bins=z_bins, histtype="step",color="Darkorange",linestyle="--",linewidth=2,density=True)
        ax[0].set_xlabel("z",fontsize=25)
        ax[0].set_ylabel("pdf",fontsize=25)
        # ax[0].hist(self.pop['non_merging_population']['z'][:][self.control_indices[self.valid_control_mask]], bins=z_bins, color="black", histtype="step",linewidth=2,density=True)
        # ax[0].hist(self.pop['merging_population']['z'][self.valid_control_mask], bins=z_bins, histtype="step",color="Darkorange",linestyle="--",linewidth=2,density=True)

        ax[1].hist(np.log10(self.Mstar_control_pop), bins=Mstar_bins,histtype="step",color="black",label="Control",linewidth=2,density=True)
        ax[1].hist(np.log10(self.Mstar_merging_pop),bins=Mstar_bins,histtype="step",label="PM",color="Darkorange",linestyle="--",linewidth=2,density=True)
        #ax[1].set_xticks([7,8,9,10,11,12])
        # ax[1].hist(np.log10(self.pop['non_merging_population']['Mstar'][:][self.control_indices[self.valid_control_mask]]), bins=Mstar_bins,histtype="step",color="black",label="Control",linewidth=2,density=True)
        # ax[1].hist(np.log10(self.pop['merging_population']['Mstar'][self.valid_control_mask]),bins=Mstar_bins,histtype="step",label="PM",color="Darkorange",linestyle="--",linewidth=2,density=True)
        #ax[1].set_xticks([7,8,9,10,11,12])
        ax[1].legend(fontsize=15)
        ax[1].set_xlabel("$\log(M_{\star}/M_{\odot})$",fontsize=25)

        #fig_name = fig_loc+"control-pm-z-Mstar-match.pdf"
        #fig.show()
        #fig.savefig(fig_name)
        #print("Figure saved in %s"%(fig_name))

        return fig,ax

    def compute_population_properties(self,verbose):

        self.z_merging_pop = self.pop['merging_population']['z'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.z_control_pop = self.pop['non_merging_population']['z'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.Mstar_merging_pop = self.pop['merging_population']['Mstar'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.Mstar_control_pop = self.pop['non_merging_population']['Mstar'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.Msubhalo_merging_pop = self.pop['merging_population']['Msubhalo'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.Msubhalo_control_pop = self.pop['non_merging_population']['Msubhalo'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.MBH_merging_pop = self.pop['merging_population']['MBH'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.MBH_control_pop = self.pop['non_merging_population']['MBH'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.SFR_merging_pop = self.pop['merging_population']['SFR'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.SFR_control_pop = self.pop['non_merging_population']['SFR'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.Mdot_merging_pop = self.pop['merging_population']['Mdot'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.Mdot_control_pop = self.pop['non_merging_population']['Mdot'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]   

        self.sSFR_merging_pop = self.SFR_merging_pop/self.Mstar_merging_pop
        self.sSFR_control_pop = self.SFR_control_pop/self.Mstar_control_pop

        self.Mgas_merging_pop = self.pop['merging_population']['Mgas'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.Mgas_control_pop = self.pop['non_merging_population']['Mgas'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]
 
        self.sBHAR_merging_pop = self.Mdot_merging_pop/self.MBH_merging_pop
        self.sBHAR_control_pop = self.Mdot_control_pop/self.MBH_control_pop

        self.StellarHalfmassRad_merging_pop = self.pop['merging_population']['StellarHalfmassRad'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.StellarHalfmassRad_control_pop = self.pop['non_merging_population']['StellarHalfmassRad'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.q_merger = self.pop['merging_population']['q_merger'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]

        self.merger_progenitor_properties()

        self.MgasInRad = self.pop['merging_population']['MgasInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.MstarInRad = self.pop['merging_population']['MstarInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.fgas_post_merger = self.MgasInRad/(self.MgasInRad + self.MstarInRad)

        self.fgas_control = self.pop['non_merging_population']['MgasInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]/(self.pop['non_merging_population']['MgasInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,1]] + self.pop['non_merging_population']['MstarInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,1]])
        
        self.SubhaloPhotoMag_merging_pop = self.pop['merging_population']['SubhaloPhotoMag'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.SubhaloPhotoMag_control_pop = self.pop['non_merging_population']['SubhaloPhotoMag'][:][self.merger_control_index_pairs[self.valid_control_mask,1]] 

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
        self.MgasInRad_progs = MgasInRad_progs[self.merger_control_index_pairs[self.valid_control_mask,0]]
        MstarInRad_progs = self.pop['merging_population']['prog_MstarInRad'][:].reshape(self.N_mergers,2)
        self.MstarInRad_progs = MstarInRad_progs[self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.fgas_progs = np.sum(self.MgasInRad_progs,axis=1)/(np.sum(self.MgasInRad_progs,axis=1)+np.sum(self.MstarInRad_progs,axis=1))

        StellarHalfmassRad_progs = self.pop['merging_population']['prog_StellarHalfmassRad'][:].reshape(self.N_mergers,2)
        self.StellarHalfmassRad_progs = StellarHalfmassRad_progs[self.merger_control_index_pairs[self.valid_control_mask,0]]

        return None
        

class control_sample_brahma:

    def __init__(self,pop_file,verbose=False):
        #note that for brahma, the population file contains the merging and control population unlike the TNG files
        
        self.pop = pop_file
        self.N_mergers = len(self.pop['merging_population']['z'])

        #self.control_indices = self.find_control_samples()
        merger_control_index_pairs = self.find_control_samples_strict_v2()  
        self.merger_control_index_pairs = np.array(merger_control_index_pairs)

        print("Number of cases where a close enough match is not found within the acceptable tolerance:",np.sum(np.array(self.merger_control_index_pairs)[:,1]==-1))

        self.MBH_not_zero_flag = self.pop['merging_population']['MBH'][:][self.merger_control_index_pairs[:,0]]!=0
        self.control_available_flag = self.merger_control_index_pairs[:,1]!=-1
        self.valid_control_mask  = self.MBH_not_zero_flag&self.control_available_flag

        self.N_mergers_w_controls = np.sum(self.valid_control_mask)
        print("number of processable mergers with valid controls:")
        print(self.N_mergers_w_controls)

        self.compute_population_properties(verbose)
        self.major_merger_mask = self.q_merger >= 0.1
        self.major_major_merger_mask = self.q_merger >= 0.25

    def find_control_samples(self):
        brahma_mergers_Mstar = self.pop['merging_population']['Mstar']
        brahma_mergers_z = self.pop['merging_population']['z']

        brahma_nonmergers_Mstar = self.pop['non_merging_population']['Mstar']
        brahma_nonmergers_z = self.pop['non_merging_population']['z']

        merging_points = np.column_stack((brahma_mergers_z,np.log10(brahma_mergers_Mstar)))
        non_merging_points = np.column_stack((brahma_nonmergers_z,np.log10(brahma_nonmergers_Mstar)))
        tree = cKDTree(non_merging_points)
        used = np.zeros(len(non_merging_points), dtype=bool)

        control_indices = []

        while True:
            closest_indices = np.full(len(merging_points), -1)
            tolerances = []
            for i in tqdm(range(len(merging_points)), desc="Processing merging points", ncols=100):
                #find the closest neibhour 
                dist, min_idx = tree.query(merging_points[i])
                if(used[min_idx]):
                    dists,idxs = tree.query(merging_points[i],k=len(non_merging_points))
                    min_idx = idxs[np.where(~used[idxs])[0][0]] 

                #check for tolerance:
                del_z = np.abs(merging_points[i][0]-non_merging_points[min_idx][0])
                dex_Mstar = np.abs(np.log10(merging_points[i][1]/non_merging_points[min_idx][1]))

                z_tolerance = 0.1
                Mstar_dex_tolerance = 0.01

                while True:
                    if(del_z<z_tolerance and dex_Mstar<Mstar_dex_tolerance):
                        used[min_idx] = True
                        closest_indices[i] = min_idx
                        tolerances.append((z_tolerance, Mstar_dex_tolerance))
                        break
                    else:
                        closest_indices[i] = -1
                        break

            control_indices.append(closest_indices)
                #D_mstar, p_Mstar = ks_2samp(pop['non_merging_population']['Mstar'][np.sort(closest_indices)],pop['merging_population']['Mstar'])
                #D_z, p_z = ks_2samp(pop['non_merging_population']['z'][np.sort(closest_indices)],pop['merging_population']['z'])
                
                #print(p_z,p_Mstar,np.shape(control_indices)[0])
                #print(z_tolerance,Mstar_dex_tolerance)

            if np.shape(control_indices)[0]>=1:
                break
                #only selecting one control sample for each merging galaxy for now

        return control_indices

    def find_control_samples_strict_v2(self,max_Mstar_tolerance=0.6,max_z_tolerance=0.3):

        all_mrgr_z = np.unique(self.pop['merging_population']['z'][:]) #all unique redshifts where BHs/galaxies are merging
        merging_pop_Mstar = self.pop['merging_population']['Mstar'][:]
        non_merging_pop_Mstar = self.pop['non_merging_population']['Mstar'][:]

        merger_control_index_pairs = []
        used = np.zeros(len(non_merging_pop_Mstar),dtype=bool)
        starting_Mstar_tol = 0.1

        for i, z_i in enumerate(tqdm(all_mrgr_z, "processing each merger redshifts for controls")):
            zi_merger_ix = np.where(self.pop['merging_population']['z'] == z_i)[0]          # global indices into merging pop
            zi_nonmrgr_ix = np.where(self.pop['non_merging_population']['z'] == z_i)[0]    # global indices into non-merging pop
            zi_nonmerger_ix = zi_nonmrgr_ix[~used[zi_nonmrgr_ix]]                         # available candidates (global indices)

            merger_Mstars = self.pop['merging_population']['Mstar'][zi_merger_ix]
            # note: we will read candidate masses from pop inside helper so they are always in sync
            starting_Mstar_tol = starting_Mstar_tol  # keep existing name in scope
            max_z_tolerance = max_z_tolerance

            def _try_match(candidate_global_idx_array, merger_mass):
                """Return global index of matched non-merger or -1."""
                if candidate_global_idx_array.size == 0:
                    return -1
                cand_masses = self.pop['non_merging_population']['Mstar'][candidate_global_idx_array]
                best_local = np.argmin(np.abs(cand_masses - merger_mass))
                best_global = candidate_global_idx_array[best_local]
                mass_diff = np.abs(np.log(cand_masses[best_local]) - np.log(merger_mass))
                return best_global if mass_diff <= starting_Mstar_tol else -1

            # loop over mergers at this redshift (use global merger index directly)
            for merger_global_idx, merger_mass in zip(zi_merger_ix, merger_Mstars):
                matched_non_global = _try_match(zi_nonmerger_ix, merger_mass)

                # if not found, try next redshift (if within tolerance) then previous
                if matched_non_global == -1 and i < len(all_mrgr_z) - 1:
                    next_non_idx = np.where(self.pop['non_merging_population']['z'] == all_mrgr_z[i + 1])[0]
                    next_candidates = next_non_idx[~used[next_non_idx]]
                    if next_candidates.size and (all_mrgr_z[i + 1] - all_mrgr_z[i]) <= max_z_tolerance:
                        matched_non_global = _try_match(next_candidates, merger_mass)

                if matched_non_global == -1 and i > 0:
                    prev_non_idx = np.where(self.pop['non_merging_population']['z'] == all_mrgr_z[i - 1])[0]
                    prev_candidates = prev_non_idx[~used[prev_non_idx]]
                    if prev_candidates.size and (all_mrgr_z[i] - all_mrgr_z[i - 1]) <= max_z_tolerance:
                        matched_non_global = _try_match(prev_candidates, merger_mass)

                if matched_non_global != -1:
                    # use global indices directly (no expensive np.where on values)
                    merger_control_index_pairs.append([merger_global_idx, matched_non_global])
                    used[matched_non_global] = True
                else:
                    merger_control_index_pairs.append([merger_global_idx, -1])

        return  np.array(merger_control_index_pairs)
    
    def compute_population_properties(self,verbose=False):

        self.z_merging_pop = self.pop['merging_population']['z'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.z_control_pop = self.pop['non_merging_population']['z'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.Mstar_merging_pop = self.pop['merging_population']['Mstar'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.Mstar_control_pop = self.pop['non_merging_population']['Mstar'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]
        
        self.Msubhalo_merging_pop = self.pop['merging_population']['Msubhalo'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.Msubhalo_control_pop = self.pop['non_merging_population']['Msubhalo'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]
        
        self.MBH_merging_pop = self.pop['merging_population']['MBH'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.MBH_control_pop = self.pop['non_merging_population']['MBH'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.SFR_merging_pop = self.pop['merging_population']['SFR'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.SFR_control_pop = self.pop['non_merging_population']['SFR'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.Mdot_merging_pop = self.pop['merging_population']['Mdot'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.Mdot_control_pop = self.pop['non_merging_population']['Mdot'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]   

        self.sSFR_merging_pop = self.SFR_merging_pop/self.Mstar_merging_pop
        self.sSFR_control_pop = self.SFR_control_pop/self.Mstar_control_pop

        self.Mgas_merging_pop = self.pop['merging_population']['Mgas'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.Mgas_control_pop = self.pop['non_merging_population']['Mgas'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.fgas_merging_pop = self.Mgas_merging_pop/(self.Mgas_merging_pop+self.Mstar_merging_pop)
        self.fgas_control_pop = self.Mgas_control_pop/(self.Mgas_control_pop+self.Mstar_control_pop)

        self.sBHAR_merging_pop = self.Mdot_merging_pop/self.MBH_merging_pop
        self.sBHAR_control_pop = self.Mdot_control_pop/self.MBH_control_pop

        self.StellarHalfmassRad_merging_pop = self.pop['merging_population']['StellarHalfmassRad'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.StellarHalfmassRad_control_pop = self.pop['non_merging_population']['StellarHalfmassRad'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]

        self.q_merger = self.pop['merging_population']['q_merger'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]

        self.merger_progenitor_properties()

        self.MgasInRad = self.pop['merging_population']['MgasInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.MstarInRad = self.pop['merging_population']['MstarInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.fgas_post_merger = self.MgasInRad/(self.MgasInRad + self.MstarInRad)

        self.fgas_control = self.pop['non_merging_population']['MgasInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,1]]/(self.pop['non_merging_population']['MgasInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,1]] + self.pop['non_merging_population']['MstarInRad'][:][self.merger_control_index_pairs[self.valid_control_mask,1]])
        
        self.SubhaloPhotoMag_merging_pop = self.pop['merging_population']['SubhaloPhotoMag'][:][self.merger_control_index_pairs[self.valid_control_mask,0]]
        self.SubhaloPhotoMag_control_pop = self.pop['non_merging_population']['SubhaloPhotoMag'][:][self.merger_control_index_pairs[self.valid_control_mask,1]] 

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


def load_pop_file(basePath,pop_file_path,minN_values):

    simName = basePath.split('/')[-2]
    pop_file_name = pop_file_path+ f"{simName}_population_sort_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5"
    #pop_file_name = pop_file_path+'population_sort'+'_gas-'+f'{minN_values[0]:03d}'+'_dm-'+f'{minN_values[1]:03d}'+'_star-'+f'{minN_values[2]:03d}'+'_bh-'+f'{minN_values[3]:03d}'+'_brahma.hdf5'
    return h5py.File(pop_file_name,'r')


# def load_pop_file(basePath,pop_file_path,minN_values):

#     simName = basePath.split('/')[-2]
#     if(simName=='TNG50-1'):
#         pop_file_name = pop_file_path+ f"population_sort_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}_w_rsep_cut_1bh.hdf5"
#     else:    
#         pop_file_name = pop_file_path+ f"{simName}_population_sort_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}_brahma.hdf5"
#     #pop_file_name = pop_file_path+'population_sort'+'_gas-'+f'{minN_values[0]:03d}'+'_dm-'+f'{minN_values[1]:03d}'+'_star-'+f'{minN_values[2]:03d}'+'_bh-'+f'{minN_values[3]:03d}'+'_brahma.hdf5'
#     return h5py.File(pop_file_name,'r')


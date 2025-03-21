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


#default values
plt.rcParams.update({'font.size': 25})
plt.rcParams.update({'xtick.labelsize': 15, 'ytick.labelsize': 15})

class control_samples:

    def __init__(self,population_file,control_file_loc,control_idx_file_name):
        self.pop = population_file
        self.control_idx_file = control_file_loc+control_idx_file_name

        if os.path.exists(self.control_idx_file):
            self.control_indices = np.loadtxt(self.control_idx_file).astype(int)
        else:
            self.control_indices = self.find_control_sample_indices(self.pop)
            self.strore_control_indices()

    def find_control_sample_indices(self,pop):
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
            if p_z < 0.99 or p_Mstar < 0.99 or np.shape(control_indices)[0]>=10:
                break # Exit the loop if the conditions are no longer true

        return control_indices

    def strore_control_indices(self):
        np.savetxt(self.control_idx_file,self.control_indices,dtype=int)

    def match_z_Mstar_plot(self,fig_loc,Mstar_binsize = 0.5,Mstar_min = 7,Mstar_max = 12,z_binsize = 0.6,z_min = 0,z_max = 5):

        Nbins_Ms = int((Mstar_max-Mstar_min)/Mstar_binsize)
        Mstar_bins = np.linspace(Mstar_min,Mstar_max,Nbins_Ms)

        Nbins_z = int((z_max - z_min) / z_binsize)
        z_bins = np.linspace(z_min, z_max, Nbins_z)

        control_sample_ids = np.array(self.control_indices).flatten()

        fig,ax = plt.subplots(1,2,figsize=(10,4))
        ax[0].hist(self.pop['non_merging_population']['z'][np.sort(control_sample_ids)], bins=z_bins, color="black", histtype="step",linewidth=2,density=True)
        ax[0].hist(self.pop['merging_population']['z'], bins=z_bins, histtype="step",color="Darkorange",linestyle="--",linewidth=2,density=True)
        ax[0].set_xlabel("z",fontsize=25)
        ax[0].set_ylabel("Density",fontsize=25)
        ax[0].set_xticks([0,1,2,3,4,5])

        ax[1].hist(np.log10(self.pop['non_merging_population']['Mstar'][np.sort(control_sample_ids)]), bins=Mstar_bins,histtype="step",color="black",label="Control",linewidth=2,density=True)
        ax[1].hist(np.log10(self.pop['merging_population']['Mstar']),bins=Mstar_bins,histtype="step",label="PM",color="Darkorange",linestyle="--",linewidth=2,density=True)
        ax[1].set_xticks([7,8,9,10,11,12])
        ax[1].legend(fontsize=15)
        ax[1].set_xlabel("$\log(M_{\star}/M_{\odot})$",fontsize=25)

        fig_name = fig_loc+"control-pm-z-Mstar-match.pdf"

        fig.savefig(fig_name)
        print("Figure saved in %s"%(fig_name))

        return fig,ax

    


    
def control_idxs(control_file_idx_name):
    control_file_loc = "/home/pranavsatheesh/host_galaxies/data/control_files/"
    control_idx_file = control_file_loc + control_file_idx_name

    return np.loadtxt(control_idx_file)


def check_control_z_Mstar_match(control_idx_file,Mstar_binsize=0.5,Mstar_min=7,Mstar_max=12,z_binsize=0.6,z_min=0,z_max=5):

    Nbins_Ms = int((Mstar_max-Mstar_min)/Mstar_binsize)
    Mstar_bins = np.linspace(Mstar_min,Mstar_max,Nbins_Ms)

    Nbins_z = int((z_max - z_min) / z_binsize)
    z_bins = np.linspace(z_min, z_max, Nbins_z)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(control_z_avg, bins=z_bins, histtype="step", color="black", label="control")
    ax[0].hist(pop['merging_population']['z'], bins=z_bins, histtype="step", label="mergers", color="orange", linestyle="--")
    ax[0].set_xlabel("z", fontsize=25)
    ax[0].set_ylabel("N", fontsize=25)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))


# class ControlSampleGenerator:
#     def __init__(self, population, z_tol_default=0.01, Mstar_dex_tol_default=0.1):
#         self.pop = population
#         self.z_tol_default = z_tol_default
#         self.Mstar_dex_tol_default = Mstar_dex_tol_default

#         self.control_sample = {
#             "idx": np.array([], dtype=int),
#             "subhalo_ids": [],
#             "snap": [],
#             "z": [],
#             "Mstar": [],
#             "Mgas": [],
#             "MBH": [],
#             "Mdot": [],
#             "SFR": [],
#             "z_tol": np.array([], dtype=float),
#             "Mstar_dex_tol": np.array([], dtype=float)
#         }

#     def generate_control_sample(self):
#         for i in range(len(self.pop['merging_population']["z"])):
#             z_mrg = self.pop['merging_population']["z"][i]
#             Mstar_mrg = self.pop['merging_population']["Mstar"][i]

#             z_tol = self.z_tol_default
#             Mstar_dex_tol = self.Mstar_dex_tol_default

#             idxs = np.where((np.abs(self.pop['non_merging_population']["z"] - z_mrg) <= z_tol) &
#                             (np.abs(np.log10(self.pop['non_merging_population']["Mstar"]) - np.log10(Mstar_mrg)) <= Mstar_dex_tol))

#             while (np.size(idxs) < 10):
#                 z_tol *= 1.5  # increase the tolerances by 50 percent
#                 Mstar_dex_tol *= 1.5

#                 idxs = np.where((np.abs(self.pop['non_merging_population']["z"] - z_mrg) <= z_tol) & 
#                                 (np.abs(np.log10(self.pop['non_merging_population']["Mstar"]) - np.log10(Mstar_mrg)) <= Mstar_dex_tol))

#             self.control_sample["idx"] = np.append(self.control_sample["idx"], i)
#             self.control_sample["subhalo_ids"].append(self.pop['non_merging_population']["subhalo_ids"][idxs])
#             self.control_sample["snap"].append(self.pop['non_merging_population']["snap"][idxs])
#             self.control_sample["z"].append(self.pop['non_merging_population']["z"][idxs])
#             self.control_sample["Mstar"].append(self.pop['non_merging_population']["Mstar"][idxs])
#             self.control_sample["Mgas"].append(self.pop['non_merging_population']["Mgas"][idxs])
#             self.control_sample["MBH"].append(self.pop['non_merging_population']["MBH"][idxs])
#             self.control_sample["Mdot"].append(self.pop['non_merging_population']["Mdot"][idxs])
#             self.control_sample["SFR"].append(self.pop['non_merging_population']["SFR"][idxs])
#             self.control_sample["z_tol"]=np.append(self.control_sample["z_tol"],z_tol)
#             self.control_sample["Mstar_dex_tol"]=np.append(self.control_sample["Mstar_dex_tol"],Mstar_dex_tol)
#             #self.control_sample["z_tol"].append(z_tol)
#             #self.control_sample["Mstar_dex_tol"].append(Mstar_dex_tol)

#         return self.control_sample

#     def write_control_sample(self, pop_file_loc):
#         control_sample_file = pop_file_loc + "control_sample.hdf5"
#         with h5py.File(control_sample_file, 'w') as f:
#             for key, value in self.control_sample.items():
#                 # Define a variable-length data type
#                 vlen_dtype = h5py.special_dtype(vlen=np.dtype('float64'))
                
#                 # Create a dataset with the variable-length data type
#                 if isinstance(value, list):
#                     dset = f.create_dataset(key, (len(value),), dtype=vlen_dtype)
#                     dset[:] = value
#                 else:
#                     f.create_dataset(key, data=value)
#         print(f"Control sample saved to {control_sample_file}")

# if __name__ == "__main__":
    
#     # Get the population file location from the command-line arguments
#     pop_file_loc = sys.argv[1]
#     pop_file = pop_file_loc + "population_sort_gas-100_dm-100_star-100_bh-001.hdf5"

#     # Open the population file
#     pop = h5py.File(pop_file, 'r')

#     control_generator = ControlSampleGenerator(pop)
#     control_sample = control_generator.generate_control_sample()

#     print(f"The maximum tolerance in z is {np.max(control_sample['z_tol']):.3f}")
#     print(f"The maximum tolerance in $M_{{\star}}$ is {np.max(control_sample['Mstar_dex_tol']):.3f}")
#     lengths_subhalo_ids = [len(sublist) for sublist in control_sample['subhalo_ids']]
#     print(f"The average number of control samples is {np.mean(lengths_subhalo_ids):.0f}")
#     print(f"The minimum number of control samples is {np.min(lengths_subhalo_ids):.0f}")

#     control_sample_file_loc = sys.argv[2]
#     # Write the control sample to a file
#     control_generator.write_control_sample(control_sample_file_loc)


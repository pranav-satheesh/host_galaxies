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


##TO-DO
# - Return the relative seperations for merging galaxies and control
# Add a user input to the maximum tolerances allowed for z and Mstar
# Function to compute KS statistic or Hellinger distances

class control_samples:

    def __init__(self,population_file,control_file_loc,control_name,brahma_key=False):

        
        self.pop = population_file
        self.control_file_loc = control_file_loc
        self.control_name = control_name
        self.control_idx_file = self.control_file_loc + self.control_name+".txt"
        # self.control_idx_file = self.control_file_loc + "control_indices_for_TNG_%1.2f.txt"%(matching_threshold)

        if os.path.exists(self.control_idx_file):
            self.control_indices = np.loadtxt(self.control_idx_file).astype(int)
        else:
            self.control_indices,self.tolerances = self.find_control_samples(self.pop)
            # self.control_indices,self.tolerances = self.find_control_sample_indices(self.pop,matching_threshold)
            self.store_control_indices()

        self.control_sample_ids = np.array(self.control_indices).flatten()
        #self.compute_population_properties()

        mergers_with_controls_found = self.control_indices[0]!=-1 #these mergers have a valid matching control non merging galaxy
        mergers_with_MBH_not_zero = self.pop['merging_population']['MBH'][:]!=0
        mergers_with_Mstar_not_zero = self.pop['merging_population']['Mstar'][:]!=0

        self.valid_control_mask =  mergers_with_controls_found & mergers_with_MBH_not_zero & mergers_with_Mstar_not_zero

        self.compute_population_properties()

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
        ax[0].hist(self.pop['non_merging_population']['z'][:][self.control_indices[self.valid_control_mask]], bins=z_bins, color="black", histtype="step",linewidth=2,density=True)
        ax[0].hist(self.pop['merging_population']['z'][self.valid_control_mask], bins=z_bins, histtype="step",color="Darkorange",linestyle="--",linewidth=2,density=True)
        ax[0].set_xlabel("z",fontsize=25)
        ax[0].set_ylabel("pdf",fontsize=25)
        ax[0].set_xticks([0,1,2,3,4,5])

        ax[1].hist(np.log10(self.pop['non_merging_population']['Mstar'][:][self.control_indices[self.valid_control_mask]]), bins=Mstar_bins,histtype="step",color="black",label="Control",linewidth=2,density=True)
        ax[1].hist(np.log10(self.pop['merging_population']['Mstar'][self.valid_control_mask]),bins=Mstar_bins,histtype="step",label="PM",color="Darkorange",linestyle="--",linewidth=2,density=True)
        ax[1].set_xticks([7,8,9,10,11,12])
        ax[1].legend(fontsize=15)
        ax[1].set_xlabel("$\log(M_{\star}/M_{\odot})$",fontsize=25)

        #fig_name = fig_loc+"control-pm-z-Mstar-match.pdf"
        #fig.show()
        #fig.savefig(fig_name)
        #print("Figure saved in %s"%(fig_name))

        return fig,ax

    def compute_population_properties(self):

        self.Mstar_merging_pop = self.pop['merging_population']['Mstar'][:][self.valid_control_mask]
        self.Mstar_control_pop = self.pop['non_merging_population']['Mstar'][:][self.control_sample_ids[self.valid_control_mask]]

        self.MBH_merging_pop = self.pop['merging_population']['MBH'][:][self.valid_control_mask]
        self.MBH_control_pop = self.pop['non_merging_population']['MBH'][:][self.control_sample_ids[self.valid_control_mask]]

        self.SFR_merging_pop = self.pop['merging_population']['SFR'][:][self.valid_control_mask]
        self.SFR_control_pop = self.pop['non_merging_population']['SFR'][:][self.control_sample_ids[self.valid_control_mask]]

        self.z_merging_pop = self.pop['merging_population']['z'][:][self.valid_control_mask]
        self.z_control_pop = self.pop['non_merging_population']['z'][:][self.control_sample_ids[self.valid_control_mask]]

        self.Mgas_merging_pop = self.pop['merging_population']['Mgas'][:][self.valid_control_mask]
        self.Mgas_control_pop = self.pop['non_merging_population']['Mgas'][:][self.control_sample_ids[self.valid_control_mask]]

        self.Mdot_merging_pop = self.pop['merging_population']['Mdot'][:][self.valid_control_mask]
        self.Mdot_control_pop = self.pop['non_merging_population']['Mdot'][:][self.control_sample_ids[self.valid_control_mask]]

        self.sSFR_merging_pop = self.SFR_merging_pop/self.Mstar_merging_pop
        self.sSFR_control_pop = self.SFR_control_pop/self.Mstar_control_pop

        self.fgas_merging_pop = self.Mgas_merging_pop/(self.Mgas_merging_pop+self.Mstar_merging_pop)
        self.fgas_control_pop = self.Mgas_control_pop/(self.Mgas_control_pop+self.Mstar_control_pop)

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

    def plot_PM_and_control_histograms(self, bin_settings=None):
    # Default bin settings if none are provided
        if bin_settings is None:
            bin_settings = {
            'sSFR': {'binsize': 0.2, 'bin_min': -14, 'bin_max': -7},
            'Mdot': {'binsize': 0.2, 'bin_min': -8, 'bin_max': 1},
            'Mgas': {'binsize': 0.2, 'bin_min': 5, 'bin_max': 14},
            'fgas': {'binsize': 0.05, 'bin_min': 0, 'bin_max': 1}
            }

        properties = {
            'sSFR': (np.log10(self.sSFR_merging_pop[self.sSFR_merging_pop>0]), np.log10(self.sSFR_control_pop[self.sSFR_control_pop>0])),
            'Mdot': (np.log10(self.Mdot_merging_pop[self.Mdot_merging_pop>0]), np.log10(self.Mdot_control_pop[self.Mdot_control_pop>0])),
            'Mgas': (np.log10(self.Mgas_merging_pop), np.log10(self.Mgas_control_pop)),
            'fgas': (self.fgas_merging_pop, self.fgas_control_pop)
        }

        properties_xlabel = {
            'sSFR': r"$\log_{10}(\mathrm{sSFR}[\mathrm{yr}^{-1}])$",
            'Mdot': r"$\log_{10}(\dot{M}_{\mathrm{BH}}[M_{\odot}\, \mathrm{yr}^{-1}])$",
            'Mgas': r"$\log_{10}(M_{\mathrm{gas}}[M_{\odot}])$",
            'fgas': r"$f_{\mathrm{gas}}$",
            }
        
        self.set_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(9,4))
        axes = axes.flatten()

        for i, (prop_name, (prop_merging, prop_control)) in enumerate(properties.items()):
    
            binsize = bin_settings[prop_name].get('binsize')
            bin_min = bin_settings[prop_name].get('bin_min')
            bin_max = bin_settings[prop_name].get('bin_max')
            bins = np.arange(bin_min, bin_max + binsize, binsize)
        
            # Plot histograms
            axes[i].hist(prop_merging, bins=bins,color='dodgerblue', label='PM', density=True,histtype="step", linewidth=2)
            axes[i].hist(prop_control, bins=bins,color='orange', label='Control', density=True,histtype="step", linewidth=2)
            axes[i].set_xlabel(properties_xlabel[prop_name])
            axes[i].set_ylabel('pdf')
            axes[i].legend()
        
        fig.tight_layout()
        #fig.show()
        return axes,fig

    def plot_sSFR_evolution(self,z_min=0,z_max=5,z_binsize=0.3):
        # Initialize lists to store the results

        Nbins_z = int((z_max - z_min) / z_binsize)
        z_bins = np.linspace(z_min, z_max, Nbins_z)

        #avg_logSFR_control = []
        avg_sSFR_control = []
        std_sSFR_control = []

        #avg_logSFR_merger = []
        avg_sSFR_merger = []
        std_sSFR_merger = []

        # Loop through redshift bins
        for i in range(len(z_bins) - 1):
            # Create masks for merging and control populations within each redshift bin
            merger_z_mask = (self.z_merging_pop > z_bins[i]) & (self.z_merging_pop < z_bins[i+1])
            control_z_mask = (self.z_control_pop > z_bins[i]) & (self.z_control_pop < z_bins[i+1])

            sSFR_merging_pop_filtered = self.sSFR_merging_pop[merger_z_mask]
            sSFR_control_pop_filtered = self.sSFR_control_pop[control_z_mask]

            avg_sSFR_merger.append(np.mean(sSFR_merging_pop_filtered))
            std_sSFR_merger.append(np.std(sSFR_merging_pop_filtered)/ np.sqrt(len(sSFR_merging_pop_filtered)))

            avg_sSFR_control.append(np.mean(sSFR_control_pop_filtered))
            std_sSFR_control.append(np.std(sSFR_control_pop_filtered)/ np.sqrt(len(sSFR_control_pop_filtered)))
        
        self.avg_sSFR_merger = np.array(avg_sSFR_merger)
        self.std_sSFR_merger = np.array(std_sSFR_merger)

        self.avg_sSFR_control = np.array(avg_sSFR_control)
        self.std_sSFR_control = np.array(std_sSFR_control)

        self.Q_sSFR = self.avg_sSFR_merger / self.avg_sSFR_control
        self.Q_sSFR_SE = self.Q_sSFR * np.sqrt((self.std_sSFR_merger / self.avg_sSFR_merger) ** 2 +(self.std_sSFR_control / self.avg_sSFR_control) ** 2)
        # Plot the results
        fig, ax = plt.subplots(2, 1, figsize=(6, 5))
        ax[0].plot(z_bins[:-1] + z_binsize / 2, np.log10(self.avg_sSFR_merger[self.avg_sSFR_merger>0]), label='PM', color="dodgerblue")
        ax[0].fill_between(z_bins[:-1] + z_binsize / 2, np.log10(self.avg_sSFR_merger-self.std_sSFR_merger), np.log10(self.avg_sSFR_merger+self.std_sSFR_merger), alpha=0.3,color='dodgerblue')
        ax[0].plot(z_bins[:-1] + z_binsize / 2, np.log10(self.avg_sSFR_control[self.avg_sSFR_control>0]), label='control', color='orange')
        ax[0].fill_between(z_bins[:-1] + z_binsize / 2, np.log10(self.avg_sSFR_control-self.std_sSFR_control), np.log10(self.avg_sSFR_control+self.std_sSFR_control), alpha=0.3,color='orange')
        ax[0].legend()
        ax[0].set_xlabel('z')
        ax[0].set_ylabel(r'$\log_{10}\langle sSFR \; [\mathrm{yr}^{-1}]\rangle$')

        ax[1].plot(z_bins[:-1] + z_binsize / 2, self.Q_sSFR,color='purple')
        ax[1].fill_between(z_bins[:-1] + z_binsize / 2, self.Q_sSFR - self.Q_sSFR_SE, self.Q_sSFR + self.Q_sSFR_SE, alpha=0.3,color='purple')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('Q(sSFR)')
        ax[1].set_ylim(0, 4)

        # Final layout adjustments
        fig.tight_layout()

        return fig,ax
    
    def plot_mdot_evolution(self, z_min=0, z_max=5, z_binsize=0.3):
        # Initialize lists to store the results

        Nbins_z = int((z_max - z_min) / z_binsize)
        z_bins = np.linspace(z_min, z_max, Nbins_z)

        #avg_logMdot_control = []
        avg_Mdot_control = []
        std_Mdot_control = []

        #avg_logMdot_merger = []
        avg_Mdot_merger = []
        std_Mdot_merger = []

        # Loop through redshift bins
        for i in range(len(z_bins) - 1):
            # Create masks for merging and control populations within each redshift bin
            merger_z_mask = (self.z_merging_pop > z_bins[i]) & (self.z_merging_pop < z_bins[i+1])
            control_z_mask = (self.z_control_pop > z_bins[i]) & (self.z_control_pop < z_bins[i+1])

            # Get the Mdot for each population
            Mdot_merging_pop_filtered = self.Mdot_merging_pop[merger_z_mask]
            Mdot_control_pop_filtered = self.Mdot_control_pop[control_z_mask]

            avg_Mdot_merger.append(np.mean(Mdot_merging_pop_filtered))
            std_Mdot_merger.append(np.std(Mdot_merging_pop_filtered)/ np.sqrt(len(Mdot_merging_pop_filtered)))

            #avg_logMdot_control.append(np.mean(log_Mdot_control_filtered))
            avg_Mdot_control.append(np.mean(Mdot_control_pop_filtered))
            std_Mdot_control.append(np.std(Mdot_control_pop_filtered)/ np.sqrt(len(Mdot_control_pop_filtered)))


        #avg_logMdot_merger = np.array(avg_logMdot_merger)
        self.avg_Mdot_merger = np.array(avg_Mdot_merger)
        self.std_Mdot_merger = np.array(std_Mdot_merger)

        #avg_logMdot_control = np.array(avg_logMdot_control)
        self.avg_Mdot_control = np.array(avg_Mdot_control)
        self.std_Mdot_control = np.array(std_Mdot_control)

        self.Q_Mdot = self.avg_Mdot_merger / self.avg_Mdot_control
        self.Q_Mdot_SE = self.Q_Mdot * np.sqrt((self.std_Mdot_merger / self.avg_Mdot_merger) ** 2 +(self.std_Mdot_control / self.avg_Mdot_control) ** 2)
        
        # Plot the results
        fig, ax = plt.subplots(2, 1, figsize=(6, 5))
        ax[0].plot(z_bins[:-1] + z_binsize / 2, np.log10(self.avg_Mdot_merger[self.avg_Mdot_merger>0]), label='PM', color='dodgerblue')
        ax[0].fill_between(z_bins[:-1] + z_binsize / 2, np.log10(self.avg_Mdot_merger-self.std_Mdot_merger), np.log10(self.avg_Mdot_merger+self.std_Mdot_merger), alpha=0.3,color='dodgerblue')
        
        ax[0].plot(z_bins[:-1] + z_binsize / 2, np.log10(self.avg_Mdot_control[self.avg_Mdot_control>0]), label='Control', color="orange")
        ax[0].fill_between(z_bins[:-1] + z_binsize / 2, np.log10(self.avg_Mdot_control-self.std_Mdot_control), np.log10(self.avg_Mdot_control+self.std_Mdot_control), alpha=0.3,color='orange')
        
        ax[0].legend()
        ax[0].set_xlabel('z')
        ax[0].set_ylabel(r'$\log_{10}\langle \dot{M}_{\mathrm{BH}} \; [M_{\odot} \, \mathrm{yr}^{-1}]\rangle$')

        ax[1].plot(z_bins[:-1] + z_binsize / 2, self.Q_Mdot,color='purple')
        ax[1].fill_between(z_bins[:-1] + z_binsize / 2, self.Q_Mdot - self.Q_Mdot_SE, self.Q_Mdot + self.Q_Mdot_SE, alpha=0.3,color='purple')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('Q($\dot{M}_{\mathrm{BH}}$)')
        ax[1].set_ylim(0, 5)

        # Final layout adjustments
        fig.tight_layout()

        return fig, ax

    def fgas_evolution(self, z_min=0, z_max=5, z_binsize=0.3):

        Nbins_z = int((z_max - z_min) / z_binsize)
        z_bins = np.linspace(z_min, z_max, Nbins_z)

        avg_fgas_control = []
        avg_fgas_merger = []

        std_fgas_control = []
        std_fgas_merger = []

        for i in range(len(z_bins) - 1):
            # Create masks for merging and control populations within each redshift bin
            merger_z_mask = (self.z_merging_pop > z_bins[i]) & (self.z_merging_pop < z_bins[i+1])
            control_z_mask = (self.z_control_pop > z_bins[i]) & (self.z_control_pop < z_bins[i+1])

            fgas_merging_pop_filtered = self.fgas_merging_pop[merger_z_mask]
            fgas_control_pop_filtered = self.fgas_control_pop[control_z_mask]

            avg_fgas_merger.append(np.mean(fgas_merging_pop_filtered))
            std_fgas_merger.append(np.std(fgas_merging_pop_filtered)/ np.sqrt(len(fgas_merging_pop_filtered)))

            avg_fgas_control.append(np.mean(fgas_control_pop_filtered))
            std_fgas_control.append(np.std(fgas_control_pop_filtered)/ np.sqrt(len(fgas_control_pop_filtered)))
        self.avg_fgas_merger = np.array(avg_fgas_merger)
        self.std_fgas_merger = np.array(std_fgas_merger)
        self.avg_fgas_control = np.array(avg_fgas_control)
        self.std_fgas_control = np.array(std_fgas_control)

        self.Q_fgas = self.avg_fgas_merger / self.avg_fgas_control
        self.Q_fgas_SE = self.Q_fgas * np.sqrt((self.std_fgas_merger / self.avg_fgas_merger) ** 2 +(self.std_fgas_control / self.avg_fgas_control) ** 2)

        # Plot the results
        fig, ax = plt.subplots(2, 1, figsize=(6, 5))
        ax[0].plot(z_bins[:-1] + z_binsize / 2, self.avg_fgas_merger, label='PM', color='dodgerblue')
        ax[0].fill_between(z_bins[:-1] + z_binsize / 2, self.avg_fgas_merger-self.std_fgas_merger, self.avg_fgas_merger+self.std_fgas_merger, alpha=0.3,color='dodgerblue')
        ax[0].plot(z_bins[:-1] + z_binsize / 2, self.avg_fgas_control, label='Control', color="orange")
        ax[0].fill_between(z_bins[:-1] + z_binsize / 2, self.avg_fgas_control-self.std_fgas_control, self.avg_fgas_control+self.std_fgas_control, alpha=0.3,color='orange')
        ax[0].legend()
        ax[0].set_xlabel('z')
        ax[0].set_ylabel(r'$\langle f_{\mathrm{gas}} \rangle$')
        ax[1].plot(z_bins[:-1] + z_binsize / 2, self.Q_fgas,color='purple')
        ax[1].fill_between(z_bins[:-1] + z_binsize / 2, self.Q_fgas - self.Q_fgas_SE, self.Q_fgas + self.Q_fgas_SE, alpha=0.3,color='purple')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('Q($f_{\mathrm{gas}}$)')
        ax[1].set_ylim(0, 5)
        # Final layout adjustments
        fig.tight_layout()
        return fig, ax

class control_sample_brahma:

    def __init__(self,pop_file_loc,pop_file_name):
        #note that for brahma, the population file contains the merging and control population unlike the TNG files
        self.pop = h5py.File(pop_file_loc+pop_file_name, 'r')

        self.N_mergers = len(self.pop['merging_population']['z'])

        valid_cases_count = np.sum(self.pop['control_population']['subhalo_ids'][:]!=-1,axis=1)
        valid_merger_elements = np.sum(valid_cases_count > 0)

        self.valid_merger_mask = valid_cases_count > 0
        self.N_filtered_mergers = valid_merger_elements

        print("The number of mergers in the population file is %d"%(self.N_mergers))
        print("%d mergers are invalid"%(self.N_mergers-valid_merger_elements))

    def match_z_Mstar_plot(self,Mstar_binsize = 0.5,Mstar_min = 7,Mstar_max = 12,z_binsize = 0.6,z_min = 0,z_max = 5):

        Nbins_Ms = int((Mstar_max-Mstar_min)/Mstar_binsize)
        Mstar_bins = np.linspace(Mstar_min,Mstar_max,Nbins_Ms)

        Nbins_z = int((z_max - z_min) / z_binsize)
        z_bins = np.linspace(z_min, z_max, Nbins_z)

        #control_sample_ids = np.array(self.control_indices).flatten()

        fig,ax = plt.subplots(1,2,figsize=(10,4))

        merging_z = self.pop['merging_population']['z'][self.valid_merger_mask]
        control_z = self.pop['control_population']['z'][self.valid_merger_mask,0]


        ax[0].hist(control_z, bins=z_bins, color="black", histtype="step",linewidth=2,density=True)
        ax[0].hist(merging_z, bins=z_bins, histtype="step",color="Darkorange",linestyle="--",linewidth=2,density=True)
        ax[0].set_xlabel("z",fontsize=25)
        ax[0].set_ylabel("pdf",fontsize=25)
        #ax[0].set_xticks([0,1,2,3,4,5])


        merging_Mstar = self.pop['merging_population']['Mstar'][self.valid_merger_mask]
        control_Mstar = self.pop['control_population']['Mstar'][self.valid_merger_mask,0]

        ax[1].hist(np.log10(control_Mstar), bins=Mstar_bins,histtype="step",color="black",label="Control",linewidth=2,density=True)
        ax[1].hist(np.log10(merging_Mstar),bins=Mstar_bins,histtype="step",label="PM",color="Darkorange",linestyle="--",linewidth=2,density=True)
        #ax[1].set_xticks([7,8,9,10,11,12])
        ax[1].legend(fontsize=15)
        ax[1].set_xlabel("$\log(M_{\star}/M_{\odot})$",fontsize=25)

        #fig_name = fig_loc+"control-pm-z-Mstar-match.pdf"
        #fig.show()
        #fig.savefig(fig_name)
        #print("Figure saved in %s"%(fig_name))

        return fig,ax


# def control_idxs(control_file_idx_name):
#     control_file_loc = "/home/pranavsatheesh/host_galaxies/data/control_files/"
#     control_idx_file = control_file_loc + control_file_idx_name

#     return np.loadtxt(control_idx_file)


# def check_control_z_Mstar_match(control_idx_file,Mstar_binsize=0.5,Mstar_min=7,Mstar_max=12,z_binsize=0.6,z_min=0,z_max=5):

#     Nbins_Ms = int((Mstar_max-Mstar_min)/Mstar_binsize)
#     Mstar_bins = np.linspace(Mstar_min,Mstar_max,Nbins_Ms)

#     Nbins_z = int((z_max - z_min) / z_binsize)
#     z_bins = np.linspace(z_min, z_max, Nbins_z)

#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#     ax[0].hist(control_z_avg, bins=z_bins, histtype="step", color="black", label="control")
#     ax[0].hist(pop['merging_population']['z'], bins=z_bins, histtype="step", label="mergers", color="orange", linestyle="--")
#     ax[0].set_xlabel("z", fontsize=25)
#     ax[0].set_ylabel("N", fontsize=25)

#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))


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


import numpy as np
import h5py
import sys
from tqdm import tqdm
import numpy as np
import os
tex_path = '/apps/texlive/2023/bin/x86_64-linux/'
os.environ['PATH'] += os.pathsep + tex_path
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')


def find_best_z_width(z_dist,z_min,z_max,z_width_initial=0.1,min_N_values=2):

    zbin_width = z_width_initial
    while True:
        z_bins = np.arange(z_min,z_max,zbin_width)
        N_values,z_bin_edges = np.histogram(z_dist,bins=z_bins)
        if np.min(N_values>=min_N_values):
            break
        else:
            zbin_width+=0.1
            continue
    
    print(zbin_width)
    return zbin_width,np.arange(z_min,z_max,zbin_width)


def find_adaptive_z_bins(z_dist,z_min,z_max,zbin_width=0.1,min_N_values=5):
    
    z_bins=[]
    N_vals=[]
    z_lower = z_min
    while True:
        z_upper = z_lower + zbin_width
        if z_upper > z_max:
            break
        else:
            N_values = np.where((z_dist>z_lower)&(z_dist<z_upper))[0].shape[0]
            if N_values>=min_N_values:
                N_vals.append(N_values)
                z_bins.append([z_lower,z_upper])
                z_lower = z_upper
            else:
            #print(z_lower,z_upper,N_values)
                zbin_width += 0.1
    z_bins = np.array(z_bins)
    z_bins = np.unique(z_bins.flatten())
    print(N_vals)
    return z_bins

def find_brahma_adaptive_z_bins(brahma_sim_obj,brahma_simName_array,z_lower=0,z_max=12,zbin_width=0.3,min_N_values=5):
    
    z_bins=[]

    while True:
        z_upper = z_lower + zbin_width
        if z_upper > z_max:
            break
        else:
        #print(z_lower,z_upper)
            Nval_sims = []
            for i,sim in enumerate(brahma_simName_array):
                N_values = np.where((brahma_sim_obj[sim].z_merging_pop>z_lower)&(brahma_sim_obj[sim].z_merging_pop<z_upper))[0].shape[0]
                Nval_sims.append(N_values)
            
            if np.min(np.array(Nval_sims))>=min_N_values:
                z_bins.append([z_lower,z_upper])
                z_lower = z_upper
            else:
            #print(z_lower,z_upper,N_values)
                zbin_width += 0.1
    z_bins = np.array(z_bins)
    z_bins = np.unique(z_bins.flatten())
    return z_bins


def sSFR_evolution_comparison_plot(ax,control_obj,z_bins):

    # Nbins_z = int((z_max - z_min) / z_binsize)
    # z_bins = np.linspace(z_min, z_max, Nbins_z)

    #avg_logSFR_control = []
    avg_sSFR_control = []
    std_sSFR_control = []

    #avg_logSFR_merger = []
    avg_sSFR_merger = []
    std_sSFR_merger = []

    # Loop through redshift bins
    for i in range(len(z_bins) - 1):
        # Create masks for merging and control populations within each redshift bin
        merger_z_mask = (control_obj.z_merging_pop >= z_bins[i]) & (control_obj.z_merging_pop < z_bins[i+1])
        control_z_mask = (control_obj.z_control_pop >= z_bins[i]) & (control_obj.z_control_pop < z_bins[i+1])

        sSFR_merging_pop_filtered = control_obj.sSFR_merging_pop[merger_z_mask]
        sSFR_control_pop_filtered = control_obj.sSFR_control_pop[control_z_mask]

        avg_sSFR_merger.append(np.mean(sSFR_merging_pop_filtered))
        std_sSFR_merger.append(np.std(sSFR_merging_pop_filtered)/ np.sqrt(len(sSFR_merging_pop_filtered)))

        avg_sSFR_control.append(np.mean(sSFR_control_pop_filtered))
        std_sSFR_control.append(np.std(sSFR_control_pop_filtered)/ np.sqrt(len(sSFR_control_pop_filtered)))
            
    avg_sSFR_merger = np.array(avg_sSFR_merger)
    std_sSFR_merger = np.array(std_sSFR_merger)
    avg_sSFR_control = np.array(avg_sSFR_control)
    std_sSFR_control = np.array(std_sSFR_control)

    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    else:
        ax.plot(z_bins[:-1], np.log10(avg_sSFR_merger[avg_sSFR_merger>0]), label='Merger host', color="dodgerblue")
        ax.fill_between(z_bins[:-1], np.log10(avg_sSFR_merger-std_sSFR_merger), np.log10(avg_sSFR_merger+std_sSFR_merger), alpha=0.3,color='dodgerblue')
        ax.plot(z_bins[:-1], np.log10(avg_sSFR_control[avg_sSFR_control>0]), label='Control', color='orange')
        ax.fill_between(z_bins[:-1], np.log10(avg_sSFR_control-std_sSFR_control), np.log10(avg_sSFR_control+std_sSFR_control), alpha=0.3,color='orange')
        #ax[0].legend()
        
    return ax

def Mdot_evolution_comparison_plot(ax,control_obj,z_min=0, z_max=5,z_binsize=0.3):
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
            merger_z_mask = (control_obj.z_merging_pop > z_bins[i]) & (control_obj.z_merging_pop < z_bins[i+1])
            control_z_mask = (control_obj.z_control_pop > z_bins[i]) & (control_obj.z_control_pop < z_bins[i+1])

            # Get the Mdot for each population
            Mdot_merging_pop_filtered = control_obj.Mdot_merging_pop[merger_z_mask]
            Mdot_control_pop_filtered = control_obj.Mdot_control_pop[control_z_mask]

            avg_Mdot_merger.append(np.mean(Mdot_merging_pop_filtered))
            std_Mdot_merger.append(np.std(Mdot_merging_pop_filtered)/ np.sqrt(len(Mdot_merging_pop_filtered)))

            #avg_logMdot_control.append(np.mean(log_Mdot_control_filtered))
            avg_Mdot_control.append(np.mean(Mdot_control_pop_filtered))
            std_Mdot_control.append(np.std(Mdot_control_pop_filtered)/ np.sqrt(len(Mdot_control_pop_filtered)))


        avg_Mdot_merger = np.array(avg_Mdot_merger)
        std_Mdot_merger = np.array(std_Mdot_merger)
        avg_Mdot_control = np.array(avg_Mdot_control)
        std_Mdot_control = np.array(std_Mdot_control)


        ax.plot(z_bins[:-1] + z_binsize / 2, np.log10(avg_Mdot_merger[avg_Mdot_merger>0]), label='Merger host', color='dodgerblue')
        ax.fill_between(z_bins[:-1] + z_binsize / 2, np.log10(avg_Mdot_merger-std_Mdot_merger), np.log10(avg_Mdot_merger+std_Mdot_merger), alpha=0.3,color='dodgerblue')
        ax.plot(z_bins[:-1] + z_binsize / 2, np.log10(avg_Mdot_control[avg_Mdot_control>0]), label='Control', color="orange")
        ax.fill_between(z_bins[:-1] + z_binsize / 2, np.log10(avg_Mdot_control-std_Mdot_control), np.log10(avg_Mdot_control+std_Mdot_control), alpha=0.3,color='orange')

        return ax

def sBHAR_evolution_comparison_plot(ax,control_obj,z_min=0, z_max=5,z_binsize=0.3):
        # Initialize lists to store the results

        Nbins_z = int((z_max - z_min) / z_binsize)
        z_bins = np.linspace(z_min, z_max, Nbins_z)

        #avg_logsBHAR_control = []
        avg_sBHAR_control = []
        std_sBHAR_control = []

        #avg_logsBHAR_merger = []
        avg_sBHAR_merger = []
        std_sBHAR_merger = []

        # Loop through redshift bins
        for i in range(len(z_bins) - 1):
            # Create masks for merging and control populations within each redshift bin
            merger_z_mask = (control_obj.z_merging_pop > z_bins[i]) & (control_obj.z_merging_pop < z_bins[i+1])
            control_z_mask = (control_obj.z_control_pop > z_bins[i]) & (control_obj.z_control_pop < z_bins[i+1])

            # Get the sBHAR for each population
            sBHAR_merging_pop_filtered = control_obj.sBHAR_merging_pop[merger_z_mask]
            sBHAR_control_pop_filtered = control_obj.sBHAR_control_pop[control_z_mask]

            avg_sBHAR_merger.append(np.mean(sBHAR_merging_pop_filtered))
            std_sBHAR_merger.append(np.std(sBHAR_merging_pop_filtered)/ np.sqrt(len(sBHAR_merging_pop_filtered)))

            #avg_logsBHAR_control.append(np.mean(log_sBHAR_control_filtered))
            avg_sBHAR_control.append(np.mean(sBHAR_control_pop_filtered))
            std_sBHAR_control.append(np.std(sBHAR_control_pop_filtered)/ np.sqrt(len(sBHAR_control_pop_filtered)))


        avg_sBHAR_merger = np.array(avg_sBHAR_merger)
        std_sBHAR_merger = np.array(std_sBHAR_merger)
        avg_sBHAR_control = np.array(avg_sBHAR_control)
        std_sBHAR_control = np.array(std_sBHAR_control)


        ax.plot(z_bins[:-1] + z_binsize / 2, np.log10(avg_sBHAR_merger[avg_sBHAR_merger>0]), label='Merger host', color='dodgerblue')
        ax.fill_between(z_bins[:-1] + z_binsize / 2, np.log10(avg_sBHAR_merger-std_sBHAR_merger), np.log10(avg_sBHAR_merger+std_sBHAR_merger), alpha=0.3,color='dodgerblue')
        ax.plot(z_bins[:-1] + z_binsize / 2, np.log10(avg_sBHAR_control[avg_sBHAR_control>0]), label='Control', color="orange")
        ax.fill_between(z_bins[:-1] + z_binsize / 2, np.log10(avg_sBHAR_control-std_sBHAR_control), np.log10(avg_sBHAR_control+std_sBHAR_control), alpha=0.3,color='orange')

        return ax

def sSFR_enhancement_calculate(control_obj,z_bins):

    # Nbins_z = int((z_max - z_min) / z_binsize)
    # z_bins = np.linspace(z_min, z_max, Nbins_z)

    avg_sSFR_log_enhancement = []
    std_sSFR_log_enhancement = []

    # Loop through redshift bins
    for i in range(len(z_bins) - 1):
        # Create masks for merging and control populations within each redshift bin
        merger_z_mask = (control_obj.z_merging_pop >= z_bins[i]) & (control_obj.z_merging_pop < z_bins[i+1])
        control_z_mask = (control_obj.z_control_pop >= z_bins[i]) & (control_obj.z_control_pop < z_bins[i+1])

        sSFR_merging_pop_filtered = control_obj.sSFR_merging_pop[merger_z_mask]
        sSFR_control_pop_filtered = control_obj.sSFR_control_pop[control_z_mask]

        sSFR_log_enhancement = []
        for i in range(len(sSFR_control_pop_filtered)):
            if sSFR_merging_pop_filtered[i]>0 and sSFR_control_pop_filtered[i]>0:
                sSFR_log_enhancement.append(np.log10(sSFR_merging_pop_filtered[i]) - np.log10(sSFR_control_pop_filtered[i]))
        
        avg_sSFR_log_enhancement.append(np.mean(sSFR_log_enhancement))
        std_sSFR_log_enhancement.append(np.std(sSFR_log_enhancement)/ np.sqrt(len(sSFR_log_enhancement)))

    avg_sSFR_log_enhancement = np.array(avg_sSFR_log_enhancement)
    std_sSFR_log_enhancement = np.array(std_sSFR_log_enhancement)

    return avg_sSFR_log_enhancement,std_sSFR_log_enhancement

def Mdot_enhancement_calculate(control_obj,z_bins):

    # Nbins_z = int((z_max - z_min) / z_binsize)
    # z_bins = np.linspace(z_min, z_max, Nbins_z)

    avg_Mdot_log_enhancement = []
    std_Mdot_log_enhancement = []

    # Loop through redshift bins
    for i in range(len(z_bins) - 1):
        # Create masks for merging and control populations within each redshift bin
        merger_z_mask = (control_obj.z_merging_pop >= z_bins[i]) & (control_obj.z_merging_pop < z_bins[i+1])
        control_z_mask = (control_obj.z_control_pop >= z_bins[i]) & (control_obj.z_control_pop < z_bins[i+1])

        Mdot_merging_pop_filtered = control_obj.Mdot_merging_pop[merger_z_mask]
        Mdot_control_pop_filtered = control_obj.Mdot_control_pop[control_z_mask]

        Mdot_log_enhancement = []
        for i in range(len(Mdot_control_pop_filtered)):
            if Mdot_merging_pop_filtered[i]>0 and Mdot_control_pop_filtered[i]>0:
                Mdot_log_enhancement.append(np.log10(Mdot_merging_pop_filtered[i]) - np.log10(Mdot_control_pop_filtered[i]))
        
        avg_Mdot_log_enhancement.append(np.mean(Mdot_log_enhancement))
        std_Mdot_log_enhancement.append(np.std(Mdot_log_enhancement)/ np.sqrt(len(Mdot_log_enhancement)))

    return np.array(avg_Mdot_log_enhancement),np.array(std_Mdot_log_enhancement)

def sBHAR_enhancement_calculate(control_obj,z_bins):

    # Nbins_z = int((z_max - z_min) / z_binsize)
    # z_bins = np.linspace(z_min, z_max, Nbins_z)

    avg_sBHAR_log_enhancement = []
    std_sBHAR_log_enhancement = []

    # Loop through redshift bins
    for i in range(len(z_bins) - 1):
        # Create masks for merging and control populations within each redshift bin
        merger_z_mask = (control_obj.z_merging_pop >= z_bins[i]) & (control_obj.z_merging_pop < z_bins[i+1])
        control_z_mask = (control_obj.z_control_pop >= z_bins[i]) & (control_obj.z_control_pop < z_bins[i+1])

        sBHAR_merging_pop_filtered = control_obj.sBHAR_merging_pop[merger_z_mask]
        sBHAR_control_pop_filtered = control_obj.sBHAR_control_pop[control_z_mask]

        sBHAR_log_enhancement = []
        for i in range(len(sBHAR_control_pop_filtered)):
            if sBHAR_merging_pop_filtered[i]>0 and sBHAR_control_pop_filtered[i]>0:
                sBHAR_log_enhancement.append(np.log10(sBHAR_merging_pop_filtered[i]) - np.log10(sBHAR_control_pop_filtered[i]))
        
        avg_sBHAR_log_enhancement.append(np.mean(sBHAR_log_enhancement))
        std_sBHAR_log_enhancement.append(np.std(sBHAR_log_enhancement)/ np.sqrt(len(sBHAR_log_enhancement)))

    return np.array(avg_sBHAR_log_enhancement),np.array(std_sBHAR_log_enhancement)

def match_z_Mstar_plot(ax,control_obj,Mstar_binsize = 0.5,Mstar_min = 7,Mstar_max = 12,z_binsize = 0.8,z_min = 0,z_max = 15):

        Nbins_Ms = int((Mstar_max-Mstar_min)/Mstar_binsize)
        Mstar_bins = np.linspace(Mstar_min,Mstar_max,Nbins_Ms)

        Nbins_z = int((z_max - z_min) / z_binsize)
        z_bins = np.linspace(z_min, z_max, Nbins_z)

        #control_sample_ids = np.array(self.control_indices).flatten()

        #fig,ax = plt.subplots(1,2,figsize=(10,4))

        # merging_z = control_obj.pop['merging_population']['z'][:][control_obj.valid_control_mask]
        # control_z = control_obj.pop['non_merging_population']['z'][:][control_obj.control_indices[0][control_obj.valid_control_mask]]
        ax[0].hist(control_obj.z_control_pop, bins=z_bins, color="black", histtype="step",linewidth=2,density=True)
        ax[0].hist(control_obj.z_merging_pop, bins=z_bins, histtype="step",color="Darkorange",linestyle="--",linewidth=2,density=True)
        ax[0].set_xlabel("z",fontsize=25)
        ax[0].set_ylabel("pdf",fontsize=25)
        #ax[0].set_xticks([0,1,2,3,4,5])
        # merging_Mstar = control_obj.pop['merging_population']['Mstar'][:][control_obj.valid_control_mask]
        # control_Mstar = control_obj.pop['non_merging_population']['Mstar'][:][control_obj.control_indices[0][control_obj.valid_control_mask]]
        ax[1].hist(np.log10(control_obj.Mstar_control_pop), bins=Mstar_bins,histtype="step",color="black",label="Control",linewidth=2,density=True)
        ax[1].hist(np.log10(control_obj.Mstar_merging_pop),bins=Mstar_bins,histtype="step",label="PM",color="Darkorange",linestyle="--",linewidth=2,density=True)
        #ax[1].set_xticks([7,8,9,10,11,12])
        ax[1].legend(fontsize=15)
        ax[1].set_xlabel("$\log(M_{\star}/M_{\odot})$",fontsize=25)

        #fig_name = fig_loc+"control-pm-z-Mstar-match.pdf"
        #fig.show()
        #fig.savefig(fig_name)
        #print("Figure saved in %s"%(fig_name))

        return ax

def set_plot_style(linewidth=3, titlesize=20,labelsize=25,ticksize=20,legendsize=20):
    '''Set the plotting style for matplotlib plots.
    Parameters:
    -----------
    linewidth : float
        The line width for plot lines.
    titlesize : int
        The font size for plot titles.
    labelsize : int
        The font size for axis labels.
    ticksize : int
        The font size for tick labels.
    legendsize : int
        The font size for legend text.
    Returns:
    --------'''
    
    plt.rcParams.update({
            'lines.linewidth': linewidth,
            'axes.labelsize': labelsize,
            'axes.titlesize': titlesize,
            'xtick.labelsize': ticksize,
            'ytick.labelsize': ticksize,
            'legend.fontsize': legendsize,
            'figure.titlesize': titlesize,
            "font.family": "Serif",  # Replace "Arial" with your desired font name
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"]
        })
    
    return None

def calculate_sSFR_enhancements(merging_pop,control_pop):
    sSFR_enhancements = []
    sSFR_enhancement_z = []
    sSFR_enhancement_Mstar = []
    
    for pair in tqdm(merger_control_index_pairs):
        merger_index = pair[0]
        control_index = pair[1]
        
        if control_index == -1: #no suitable control found
            continue
        
        sSFR_merger = merging_pop['SFR'][merger_index]/merging_pop['Mstar'][merger_index]
        sSFR_control = non_merging_pop['SFR'][control_index]/non_merging_pop['Mstar'][control_index]
        
        if sSFR_control > 0:
            sSFR_enhancement = sSFR_merger/sSFR_control
            sSFR_enhancements.append(sSFR_enhancement)
            sSFR_enhancement_z.append(merging_pop['z'][merger_index])
            sSFR_enhancement_Mstar.append(merging_pop['Mstar'][merger_index])
    
    return np.array(sSFR_enhancements),np.array(sSFR_enhancement_z),np.array(sSFR_enhancement_Mstar)

def sSFR_returns(z_l,z_u,Mstar_l,Mstar_u):
    sSFR_enhancements_per_zbins = sSFR_enhancements[(sSFR_enhancement_z>=z_l)&(sSFR_enhancement_z<z_u)&(sSFR_enhancement_Mstar>=Mstar_l)&(sSFR_enhancement_Mstar<Mstar_u)]
    return sSFR_enhancements_per_zbins

def sBHAR_z_evolve_plot(ax,z_bins,brahma_sim_obj,brahma_simName_array,brahma_sim_colors):

        for i,sim in enumerate(brahma_simName_array): 
            avg_sBHAR_merger = []
            std_sBHAR_merger = []

            # Loop through redshift bins
            for i in range(len(z_bins) - 1):

                merger_z_mask = (brahma_sim_obj[sim].z_merging_pop > z_bins[i]) & (brahma_sim_obj[sim].z_merging_pop < z_bins[i+1])
                sBHAR_merging_pop_filtered = brahma_sim_obj[sim].Mdot_merging_pop[merger_z_mask]
            
                avg_sBHAR_merger.append(np.mean(sBHAR_merging_pop_filtered))
                std_sBHAR_merger.append(np.std(sBHAR_merging_pop_filtered)/ np.sqrt(len(sBHAR_merging_pop_filtered)))

            avg_sBHAR_merger = np.array(avg_sBHAR_merger)
            std_sBHAR_merger = np.array(std_sBHAR_merger)

            ax.plot(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_sBHAR_merger), label=sim, color=brahma_sim_colors[sim])
            ax.fill_between(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_sBHAR_merger-std_sBHAR_merger), np.log10(avg_sBHAR_merger+std_sBHAR_merger), alpha=0.1,color=brahma_sim_colors[sim])
        
        return ax

def Mgas_z_evolve_plot(ax,z_bins,brahma_sim_obj,brahma_simName_array,brahma_sim_colors):

        for i,sim in enumerate(brahma_simName_array): 
            avg_Mgas_merger = []
            std_Mgas_merger = []

            # Loop through redshift bins
            for i in range(len(z_bins) - 1):

                merger_z_mask = (brahma_sim_obj[sim].z_merging_pop > z_bins[i]) & (brahma_sim_obj[sim].z_merging_pop < z_bins[i+1])
                Mgas_merging_pop_filtered = brahma_sim_obj[sim].Mgas_merging_pop[merger_z_mask]
            
                avg_Mgas_merger.append(np.mean(Mgas_merging_pop_filtered))
                std_Mgas_merger.append(np.std(Mgas_merging_pop_filtered)/ np.sqrt(len(Mgas_merging_pop_filtered)))

            avg_Mgas_merger = np.array(avg_Mgas_merger)
            std_Mgas_merger = np.array(std_Mgas_merger)

            ax.plot(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_Mgas_merger), label=sim, color=brahma_sim_colors[sim])
            ax.fill_between(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_Mgas_merger-std_Mgas_merger), np.log10(avg_Mgas_merger+std_Mgas_merger), alpha=0.1,color=brahma_sim_colors[sim])
        
        return ax

def sSFR_z_evolve_plot(ax,z_bins,brahma_sim_obj,brahma_simName_array,brahma_sim_colors):

        for i,sim in enumerate(brahma_simName_array): 
            avg_SFR_merger = []
            std_SFR_merger = []

            # Loop through redshift bins
            for i in range(len(z_bins) - 1):

                merger_z_mask = (brahma_sim_obj[sim].z_merging_pop > z_bins[i]) & (brahma_sim_obj[sim].z_merging_pop < z_bins[i+1])
                sSFR_merging_pop_filtered = brahma_sim_obj[sim].sSFR_merging_pop[merger_z_mask]
            
                avg_SFR_merger.append(np.mean(sSFR_merging_pop_filtered))
                std_SFR_merger.append(np.std(sSFR_merging_pop_filtered)/ np.sqrt(len(sSFR_merging_pop_filtered)))

            avg_SFR_merger = np.array(avg_SFR_merger)
            std_SFR_merger = np.array(std_SFR_merger)

            ax.plot(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_SFR_merger), label=sim, color=brahma_sim_colors[sim])
            ax.fill_between(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_SFR_merger-std_SFR_merger), np.log10(avg_SFR_merger+std_SFR_merger), alpha=0.1,color=brahma_sim_colors[sim])
        
        return ax

def Mgas_z_evolve_plot_TNG(ax,z_bins,tng_obj,TNG50color='purple'):


        avg_Mgas_merger = []
        std_Mgas_merger = []

        # Loop through redshift bins
        for i in range(len(z_bins) - 1):

            merger_z_mask = (tng_obj.z_merging_pop > z_bins[i]) & (tng_obj.z_merging_pop < z_bins[i+1])
            Mgas_merging_pop_filtered = tng_obj.Mgas_merging_pop[merger_z_mask]

            avg_Mgas_merger.append(np.mean(Mgas_merging_pop_filtered))
            std_Mgas_merger.append(np.std(Mgas_merging_pop_filtered)/ np.sqrt(len(Mgas_merging_pop_filtered)))

        avg_Mgas_merger = np.array(avg_Mgas_merger)
        std_Mgas_merger = np.array(std_Mgas_merger)

        ax.plot(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_Mgas_merger), label='TNG-50', color=TNG50color)
        ax.fill_between(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_Mgas_merger-std_Mgas_merger), np.log10(avg_Mgas_merger+std_Mgas_merger), alpha=0.1,color=TNG50color)

        return ax

def sSFR_z_evolve_plot_TNG(ax,z_bins,tng_obj,TNG50color='purple'):

    avg_sSFR_merger = []
    std_sSFR_merger = []

    # Loop through redshift bins
    for i in range(len(z_bins) - 1):

        merger_z_mask = (tng_obj.z_merging_pop > z_bins[i]) & (tng_obj.z_merging_pop < z_bins[i+1])
        sSFR_merging_pop_filtered = tng_obj.sSFR_merging_pop[merger_z_mask]

        avg_sSFR_merger.append(np.mean(sSFR_merging_pop_filtered))
        std_sSFR_merger.append(np.std(sSFR_merging_pop_filtered)/ np.sqrt(len(sSFR_merging_pop_filtered)))

    avg_sSFR_merger = np.array(avg_sSFR_merger)
    std_sSFR_merger = np.array(std_sSFR_merger)

    ax.plot(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_sSFR_merger), label='TNG-50', color=TNG50color)
    ax.fill_between(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_sSFR_merger-std_sSFR_merger), np.log10(avg_sSFR_merger+std_sSFR_merger), alpha=0.1,color=TNG50color)

    return ax

def sBHAR_z_evolve_plot_TNG(ax,z_bins,tng_obj,TNG50color='purple'):

         
    avg_sBHAR_merger = []
    std_sBHAR_merger = []

    # Loop through redshift bins
    for i in range(len(z_bins) - 1):

        merger_z_mask = (tng_obj.z_merging_pop > z_bins[i]) & (tng_obj.z_merging_pop < z_bins[i+1])
        sBHAR_merging_pop_filtered = tng_obj.Mdot_merging_pop[merger_z_mask]
    
        avg_sBHAR_merger.append(np.mean(sBHAR_merging_pop_filtered))
        std_sBHAR_merger.append(np.std(sBHAR_merging_pop_filtered)/ np.sqrt(len(sBHAR_merging_pop_filtered)))

    avg_sBHAR_merger = np.array(avg_sBHAR_merger)
    std_sBHAR_merger = np.array(std_sBHAR_merger)

    ax.plot(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_sBHAR_merger), label='TNG50', color=TNG50color)
    ax.fill_between(z_bins[:-1] + np.diff(z_bins) / 2, np.log10(avg_sBHAR_merger-std_sBHAR_merger), np.log10(avg_sBHAR_merger+std_sBHAR_merger), alpha=0.1,color=TNG50color)

    return ax

def sSFR_dist_brahma(ax,brahma_simName_array,brahma_sim_obj,brahma_sim_colors, SFR_log_min = -13,SFR_log_max = -7,N_bins=15):
    
    SFR_log_bins = np.linspace(SFR_log_min,SFR_log_max,N_bins)

    print("Median sSFR in mergers:")
    for i,sim in enumerate(brahma_simName_array):
        SFR_mergers = brahma_sim_obj[sim].sSFR_merging_pop
        print(f"{sim},{np.median(SFR_mergers[SFR_mergers>0]):2.2e}")
        ax.hist(np.log10(SFR_mergers[SFR_mergers>0]),bins=SFR_log_bins,label=sim,histtype='step',density=True,color=brahma_sim_colors[sim])
    ax.set_xlabel("log sSFR")
    #ax.legend()
    #plt.show()
    return ax

def sBHAR_dist_brahma(ax,brahma_simName_array,brahma_sim_obj,brahma_sim_colors, sBHAR_log_min = -15,sBHAR_log_max = -7,N_bins=15):
    
    sBHAR_log_bins = np.linspace(sBHAR_log_min,sBHAR_log_max,N_bins)

    print("Median sBHAR in mergers:")
    for i,sim in enumerate(brahma_simName_array):
        sBHAR_mergers = brahma_sim_obj[sim].sBHAR_merging_pop
        print(f"{sim},{np.median(sBHAR_mergers[sBHAR_mergers>0]):2.2e}")
        ax.hist(np.log10(sBHAR_mergers),bins=sBHAR_log_bins,label=sim,histtype='step',density=True,color=brahma_sim_colors[sim])
    ax.set_xlabel("log sBHAR")
    #ax.legend()
    #plt.show()
    return ax

def Mgas_dist_brahma(ax,brahma_simName_array,brahma_sim_obj,brahma_sim_colors,Mgas_log_min = 7,Mgas_log_max = 11,N_bins=15):
    
    Mgas_log_bins = np.linspace(Mgas_log_min,Mgas_log_max,N_bins)

    print("Median Mgas in mergers:")
    for i,sim in enumerate(brahma_simName_array):
        Mgas_mergers = brahma_sim_obj[sim].Mgas_merging_pop
        print(f"{sim},{np.median(Mgas_mergers[Mgas_mergers>0]):2.2e}")
        ax.hist(np.log10(Mgas_mergers[Mgas_mergers>0]),bins=Mgas_log_bins,label=sim,histtype='step',density=True,color=brahma_sim_colors[sim])
    ax.set_xlabel("log Mgas")
    #ax.legend()
    #plt.show()
    return ax
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


def sSFR_evolution_comparison_plot(ax,control_obj,z_min=0,z_max=8,z_binsize=1):

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
        ax.plot(z_bins[:-1] + z_binsize / 2, np.log10(avg_sSFR_merger[avg_sSFR_merger>0]), label='Merger host', color="dodgerblue")
        ax.fill_between(z_bins[:-1] + z_binsize / 2, np.log10(avg_sSFR_merger-std_sSFR_merger), np.log10(avg_sSFR_merger+std_sSFR_merger), alpha=0.3,color='dodgerblue')
        ax.plot(z_bins[:-1] + z_binsize / 2, np.log10(avg_sSFR_control[avg_sSFR_control>0]), label='Control', color='orange')
        ax.fill_between(z_bins[:-1] + z_binsize / 2, np.log10(avg_sSFR_control-std_sSFR_control), np.log10(avg_sSFR_control+std_sSFR_control), alpha=0.3,color='orange')
        #ax[0].legend()
        
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

    return avg_sSFR_log_enhancement,std_sSFR_log_enhancement,z_bins

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
            'figure.titlesize': titlesize
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
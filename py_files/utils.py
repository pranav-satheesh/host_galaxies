import control_sample as control
import host_galaxy_enhancement_plots as hostplot
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
from astropy import units as u


pop_file_path = '/home/pranavsatheesh/host_galaxies/data/population_files/' 
TNG_basepath = '/orange/lblecha/IllustrisTNG/Runs/TNG50-1/output'
brahma_basepath="/orange/lblecha/aklantbhowmick/GAS_BASED_SEED_MODEL_UNIFORM_RUNS/L12p5n512/AREPO/"

brahma_simName_array = [
    'SM5_TNG', 
    'SM5_LW10_TNG', 
    'SM5_LW10_LOWSPIN_TNG', 
    'SM5_LW10_LOWSPIN_RICH_TNG', 
    'SM5_DFD_3_TNG', 
    'SM5_LW10_DFD_3_TNG'
]

brahma_sim_colors = {
    brahma_simName_array[0]: '#bdc9e1', 
    brahma_simName_array[1]: '#67a9cf', 
    brahma_simName_array[2]: '#1c9099', 
    brahma_simName_array[3]: '#016c59',
    brahma_simName_array[4]: '#3690c0', 
    brahma_simName_array[5]: '#014636'  
}

TNG_color = '#d95f02'



def load_control_brahma(sim_path,pop_file_path,Nlist):
    pop = control.load_pop_file(sim_path,pop_file_path,Nlist)
    return control.control_sample_brahma(pop)

def load_control_TNG(sim_path,pop_file_path,Nlist):
    pop = control.load_pop_file(sim_path,pop_file_path,Nlist)
    return control.control_samples_TNG(pop)

def load_all_sim_control_objs():
    #this function by default loads all brahma sims and TNG40 sim objects
    
    sim_objs = {}
    TNG_50_control = load_control_TNG(TNG_basepath,pop_file_path,[0,0,1000,1])
    brahma_sim_obj = {}
    brahma_sim_zbins = {}
    for i,sim in enumerate(brahma_simName_array):
        simPath = brahma_basepath + sim + '/'
        brahma_control = load_control_brahma(simPath,pop_file_path,[0,0,10,1])
        brahma_sim_obj[sim] = brahma_control
        #brahma_sim_zbin_width,brahma_sim_zbins[sim] =hostplot.find_best_z_width(brahma_control.z_merging_pop,z_min=0,z_max=10,z_width_initial=0.2)
        brahma_sim_zbins[sim] = hostplot.find_adaptive_z_bins(brahma_control.z_merging_pop,z_min=0,z_max=12,zbin_width=0.3,min_N_values=5)


    
    TNG_50_zbins = hostplot.find_adaptive_z_bins(TNG_50_control.z_merging_pop,z_min=0,z_max=10,zbin_width=0.3,min_N_values=40)
    print('TNG50',TNG_50_zbins )
    brahma_zbins = hostplot.find_brahma_adaptive_z_bins(brahma_sim_obj,brahma_simName_array,z_lower=0,z_max=15,zbin_width=0.4,min_N_values=10)
    print('Brahma common zbins:',brahma_zbins)

    sim_names = ['TNG50'] + brahma_simName_array
    sim_colors = [TNG_color] + [brahma_sim_colors[sim] for sim in brahma_simName_array] 
    sim_objs = brahma_sim_obj
    sim_objs['TNG50'] = TNG_50_control
    sim_zbins_list = [TNG_50_zbins] + [brahma_zbins for _ in brahma_simName_array] 


    return sim_objs,sim_names,sim_colors,sim_zbins_list

def hellinger_distance(data1, data2, bins=30, range=None):
    """
    Calculate Hellinger distance between two distributions.
    
    Parameters:
    -----------
    data1 : array-like
        First dataset (e.g., merger population)
    data2 : array-like
        Second dataset (e.g., control population)
    bins : int or array-like
        Number of bins or bin edges for histogram
    range : tuple, optional
        (min, max) range for histogram
        
    Returns:
    --------
    H : float
        Hellinger distance (0 = identical, 1 = no overlap)
    """
    # Create histograms with same bins
    if range is None:
        range = (min(np.min(data1), np.min(data2)), max(np.max(data1), np.max(data2)))
    
    hist1, bin_edges = np.histogram(data1, bins=bins, range=range, density=True)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
    
    # Normalize to probability distributions (sum to 1)
    bin_width = bin_edges[1] - bin_edges[0]
    p = hist1 * bin_width
    q = hist2 * bin_width
    
    # Hellinger distance
    H = np.sqrt(1 - np.sum(np.sqrt(p * q)))
    
    return H

def merger_enhancement_calc(sim_obj,quantity,zbins,log=True,major_merger_mask=False):



    avg_quantity_enhancement = []
    std_quantity_enhancement = []

    for i in range(len(zbins)-1):
        merger_z_mask = (sim_obj.z_merging_pop >= zbins[i]) & (sim_obj.z_merging_pop < zbins[i+1])
        if major_merger_mask == True:
            merger_z_mask = merger_z_mask & sim_obj.major_major_merger_mask

        if quantity == 'Mgas':
            merging_pop_quantity = getattr(sim_obj,'MgasInRad')[merger_z_mask]
            control_pop_quantity = getattr(sim_obj,'Mgas_control_pop')[merger_z_mask]
            
           
        elif quantity == 'fgas':
            #merging_pop_quantity = getattr(sim_obj,'fgas_post_merger')[merger_z_mask]/getattr(sim_obj,'MstarInRad')[merger_z_mask]
            merging_pop_quantity = getattr(sim_obj,'fgas_progs')[merger_z_mask]/getattr(sim_obj,'MstarInRad')[merger_z_mask]
            control_pop_quantity = getattr(sim_obj,'fgas_control')[merger_z_mask]/getattr(sim_obj,'Mstar_control_pop')[merger_z_mask]
        
        elif quantity == 'StellarHalfmassRad':
            scale_factor_mergers = 1/(1+sim_obj.z_merging_pop[merger_z_mask])
            scale_factor_controls = 1/(1+sim_obj.z_control_pop[merger_z_mask])
            merging_pop_quantity = getattr(sim_obj,'StellarHalfmassRad_merging_pop')[merger_z_mask]
            control_pop_quantity = getattr(sim_obj,'StellarHalfmassRad_control_pop')[merger_z_mask]
            merging_pop_quantity = merging_pop_quantity * scale_factor_mergers
            control_pop_quantity = control_pop_quantity * scale_factor_controls        
            
        else:
            merging_pop_quantity = getattr(sim_obj,quantity+"_merging_pop")[merger_z_mask]
            control_pop_quantity = getattr(sim_obj,quantity+"_control_pop")[merger_z_mask]

        if log == True:
            quantity_log_enhancement = []
            for i in range(len(control_pop_quantity)):
                if control_pop_quantity[i]>0 and merging_pop_quantity[i]>0:
                    quantity_log_enhancement.append(np.log10(merging_pop_quantity[i]/control_pop_quantity[i]))
            avg_quantity_enhancement.append(np.mean(quantity_log_enhancement))
            #avg_quantity_enhancement.append(np.median(quantity_log_enhancement))
            std_quantity_enhancement.append(np.std(quantity_log_enhancement)/np.sqrt(len(quantity_log_enhancement)))
            #std_quantity_enhancement.append(stats.sem(quantity_log_enhancement))
        else:
            quantity_enhancement = merging_pop_quantity - control_pop_quantity
            avg_quantity_enhancement.append(np.mean(quantity_enhancement))
            #avg_quantity_enhancement.append(np.median(quantity_log_enhancement))
            std_quantity_enhancement.append(np.std(quantity_enhancement)/np.sqrt(len(quantity_enhancement)))
            #std_quantity_enhancement.append(stats.sem(quantity_enhancement))

    avg_quantity_enhancement = np.array(avg_quantity_enhancement)
    std_quantity_enhancement = np.array(std_quantity_enhancement)

    return avg_quantity_enhancement,std_quantity_enhancement

def enhancement_vs_mstar_for_z(ax,sim_obj,quantity,mstar_bins,zlow=0,zupper=1,log=True,major_merger_flag=False,minor_merger_flag=False,min_N=5):
    
    median_enhancement = []
    sem_enhancement = []

    for i in range(len(mstar_bins)-1):
        merger_z_mask = (sim_obj.z_merging_pop >= zlow) & (sim_obj.z_merging_pop < zupper)
        merger_mstar_mask = (sim_obj.Mstar_merging_pop >= mstar_bins[i]) & (sim_obj.Mstar_merging_pop < mstar_bins[i+1])

        if major_merger_flag == False and minor_merger_flag == False:
            all_relevant_merger_mask = merger_z_mask & merger_mstar_mask

        elif major_merger_flag == True:
            all_relevant_merger_mask = merger_z_mask & merger_mstar_mask & sim_obj.major_major_merger_mask

        elif minor_merger_flag == True:
            all_relevant_merger_mask = merger_z_mask & merger_mstar_mask & (sim_obj.q_merger < 0.25) & (sim_obj.q_merger >= 0.1)

        merging_pop_quantity = getattr(sim_obj,quantity+"_merging_pop")[all_relevant_merger_mask]
        control_pop_quantity = getattr(sim_obj,quantity+"_control_pop")[all_relevant_merger_mask]

        if len(merging_pop_quantity) < min_N:
            median_enhancement.append(np.nan)
            sem_enhancement.append(np.nan)
            continue

        if log == True:
            quantity_log_enhancement = []
            for i in range(len(control_pop_quantity)):
                if control_pop_quantity[i]>0 and merging_pop_quantity[i]>0:
                    quantity_log_enhancement.append(np.log10(merging_pop_quantity[i]/control_pop_quantity[i]))
            median_enhancement.append(np.median(quantity_log_enhancement))
            sem_enhancement.append(stats.sem(quantity_log_enhancement))
        else:
            quantity_enhancement = merging_pop_quantity/control_pop_quantity
            median_enhancement.append(np.median(quantity_enhancement))
            sem_enhancement.append(stats.sem(quantity_enhancement))
        
    median_enhancement = np.array(median_enhancement)
    sem_enhancement = np.array(sem_enhancement)

    return median_enhancement,sem_enhancement

def enhancement_vs_mbh_for_z(ax,sim_obj,quantity,mbh_bins,zlow=0,zupper=1,log=True,major_merger_flag=False,minor_merger_flag=False,min_N=5):
    
    median_enhancement = []
    sem_enhancement = []

    for i in range(len(mbh_bins)-1):
        merger_z_mask = (sim_obj.z_merging_pop >= zlow) & (sim_obj.z_merging_pop < zupper)
        merger_mbh_mask = (sim_obj.MBH_merging_pop >= mbh_bins[i]) & (sim_obj.MBH_merging_pop < mbh_bins[i+1])

        if major_merger_flag == False and minor_merger_flag == False:
            all_relevant_merger_mask = merger_z_mask & merger_mbh_mask

        elif major_merger_flag == True:
            all_relevant_merger_mask = merger_z_mask & merger_mbh_mask & sim_obj.major_major_merger_mask

        elif minor_merger_flag == True:
            all_relevant_merger_mask = merger_z_mask & merger_mbh_mask & (sim_obj.q_merger < 0.25) & (sim_obj.q_merger >= 0.1)

        merging_pop_quantity = getattr(sim_obj,quantity+"_merging_pop")[all_relevant_merger_mask]
        control_pop_quantity = getattr(sim_obj,quantity+"_control_pop")[all_relevant_merger_mask]

        if len(merging_pop_quantity) < min_N:
            median_enhancement.append(np.nan)
            sem_enhancement.append(np.nan)
            continue

        if log == True:
            quantity_log_enhancement = []
            for j in range(len(control_pop_quantity)):
                if control_pop_quantity[j]>0 and merging_pop_quantity[j]>0:
                    quantity_log_enhancement.append(np.log10(merging_pop_quantity[j]/control_pop_quantity[j]))
            median_enhancement.append(np.median(quantity_log_enhancement))
            sem_enhancement.append(stats.sem(quantity_log_enhancement))
        else:
            quantity_enhancement = merging_pop_quantity/control_pop_quantity
            median_enhancement.append(np.median(quantity_enhancement))
            sem_enhancement.append(stats.sem(quantity_enhancement))
        
    median_enhancement = np.array(median_enhancement)
    sem_enhancement = np.array(sem_enhancement)

    return median_enhancement,sem_enhancement

def make_hex_plot_quantity_vs_black_hole_mass(ax,sim_obj,quantity='sBHAR',zlow=0,zupper=1,
                                               gridsize=50,cmap='Blues',xmin=5,xmax=9,
                                               ymin=-2,ymax=2,mincnt=1,alpha=0.5,
                                               color_quantity='count',vmin=None,vmax=None):
    merger_z_mask = (sim_obj.z_merging_pop >= zlow) & (sim_obj.z_merging_pop < zupper)

    if quantity == 'fgas':
        merging_pop_quantity = getattr(sim_obj,'fgas_progs')[merger_z_mask]
        control_pop_quantity = getattr(sim_obj,'fgas_control')[merger_z_mask]
    else:
        merging_pop_quantity = getattr(sim_obj,quantity+"_merging_pop")[merger_z_mask]
        control_pop_quantity = getattr(sim_obj,quantity+"_control_pop")[merger_z_mask]

    quantity_enhancement = merging_pop_quantity / control_pop_quantity
    MBH_mergers = np.log10(sim_obj.MBH_merging_pop[merger_z_mask])

    x = MBH_mergers
    y = np.log10(quantity_enhancement)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if color_quantity == 'fgas':
        # color by mean gas fraction in each hex
        fgas_val = getattr(sim_obj, 'fgas_progs')[merger_z_mask][valid]
        hb = ax.hexbin(x, y, C=fgas_val, reduce_C_function=np.nanmean,
                       gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                       extent=[xmin,xmax,ymin,ymax], alpha=alpha,vmin=vmin,vmax=vmax)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label(r'$f_{\mathrm{gas}}$')
    elif color_quantity == 'mstar':
        # color by mean stellar mass in each hex
        mstar_val = np.log10(getattr(sim_obj, 'Mstar_merging_pop')[merger_z_mask][valid])
        hb = ax.hexbin(x, y, C=mstar_val, reduce_C_function=np.nanmean,
                       gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                       extent=[xmin,xmax,ymin,ymax], alpha=alpha,vmin=vmin,vmax=vmax)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label(r'$\log(M_{\star}/M_{\odot})$')
    else:
        # default: counts
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                       extent=[xmin,xmax,ymin,ymax], alpha=alpha)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Number')

    return ax

def make_hex_plot_quantity_vs_stellar_mass(ax,sim_obj,quantity='sSFR',zlow=0,zupper=1,
                                           
                                           gridsize=50,cmap='Blues',xmin=7,xmax=12,
                                           ymin=-2,ymax=2,mincnt=1,alpha=0.5,
                                           color_quantity='count',vmin=None,vmax=None):
    merger_z_mask = (sim_obj.z_merging_pop >= zlow) & (sim_obj.z_merging_pop < zupper)

    if quantity == 'fgas':
        merging_pop_quantity = getattr(sim_obj,'fgas_progs')[merger_z_mask]
        control_pop_quantity = getattr(sim_obj,'fgas_control')[merger_z_mask]
    else:
        merging_pop_quantity = getattr(sim_obj,quantity+"_merging_pop")[merger_z_mask]
        control_pop_quantity = getattr(sim_obj,quantity+"_control_pop")[merger_z_mask]

    quantity_enhancement = merging_pop_quantity / control_pop_quantity
    Mstar_mergers = np.log10(sim_obj.Mstar_merging_pop[merger_z_mask])

    x = Mstar_mergers
    y = np.log10(quantity_enhancement)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if color_quantity == 'fgas':
        # color by mean gas fraction in each hex
        fgas_val = getattr(sim_obj, 'fgas_progs')[merger_z_mask][valid]
        hb = ax.hexbin(x, y, C=fgas_val, reduce_C_function=np.nanmean,
                       gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                       extent=[xmin,xmax,ymin,ymax], alpha=alpha,vmin=vmin,vmax=vmax)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label(r'$f_{\mathrm{gas}}$')
    else:
        # default: counts
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                       extent=[xmin,xmax,ymin,ymax], alpha=alpha)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Number')

    return ax

def z_to_tlookback(z):
    """Convert redshift to lookback time in Gyr using Planck15 cosmology."""
    return Planck15.lookback_time(z).value  # returns time in Gyr

def find_merger_time_delays(sim_objs, sim_names):
    merger_time_delay = {}
    for sim in sim_names:
        merger_time_delay[sim] = {'td':[], 'snap1':[], 'snap2':[], 'snap_f':[]}
        for i, z_prog in enumerate(sim_objs[sim].z_progs):
            z_post_merger = sim_objs[sim].z_merging_pop[i]
            if z_prog[0] != z_prog[1]:
                z_merger_time = min(z_prog[0], z_prog[1])
            else:
                z_merger_time = z_prog[0]
            tdelay = z_to_tlookback(z_merger_time) - z_to_tlookback(z_post_merger)
            
            # Get snapshot numbers
            snap1 = sim_objs[sim].snap_progs[i][0] if hasattr(sim_objs[sim], 'snap_progs') else None
            snap2 = sim_objs[sim].snap_progs[i][1] if hasattr(sim_objs[sim], 'snap_progs') else None
            snap_f = sim_objs[sim].snap_merging_pop[i] if hasattr(sim_objs[sim], 'snap_merging_pop') else None
            
            merger_time_delay[sim]['td'].append(tdelay)
            merger_time_delay[sim]['snap1'].append(snap1)
            merger_time_delay[sim]['snap2'].append(snap2)
            merger_time_delay[sim]['snap_f'].append(snap_f)
    return merger_time_delay

def merger_enhancement_calc_for_small_td(sim_name,sim_objs,merger_time_delay,td_threshold=0.5,quantity='sSFR',min_count=5):
    
    merger_time_delay_array = np.array(merger_time_delay[sim_name]['td'])
    small_td_mask = merger_time_delay_array < td_threshold

    z_with_small_td = sim_objs[sim_name].z_merging_pop[small_td_mask]
    unique_zs, counts_per_z = np.unique(z_with_small_td,return_counts=True)
    filtered_unique_zs = unique_zs[counts_per_z>=min_count]
    unique_zs = filtered_unique_zs

    avg_quantity_enhancement = []
    std_quantity_enhancement = []

    for z in unique_zs:
        if quantity == 'fgas':
            merging_pop_quantity = getattr(sim_objs[sim_name],'fgas_progs')[(sim_objs[sim_name].z_merging_pop==z) & small_td_mask]
            control_pop_quantity = getattr(sim_objs[sim_name],'fgas_control')[(sim_objs[sim_name].z_merging_pop==z) & small_td_mask]
        else:
            merging_pop_quantity = getattr(sim_objs[sim_name],quantity+"_merging_pop")[(sim_objs[sim_name].z_merging_pop==z) & small_td_mask]
            control_pop_quantity = getattr(sim_objs[sim_name],quantity+"_control_pop")[(sim_objs[sim_name].z_merging_pop==z) & small_td_mask]

        quantity_enhancement = []
        for i in range(len(control_pop_quantity)):
            if control_pop_quantity[i]>0 and merging_pop_quantity[i]>0:
                quantity_enhancement.append(np.log10(merging_pop_quantity[i]/control_pop_quantity[i]))
        
        avg_quantity_enhancement.append(np.mean(quantity_enhancement))
        std_quantity_enhancement.append(np.std(quantity_enhancement)/np.sqrt(len(quantity_enhancement)))
    avg_quantity_enhancement = np.array(avg_quantity_enhancement)
    std_quantity_enhancement = np.array(std_quantity_enhancement) 
    
    return filtered_unique_zs,avg_quantity_enhancement, std_quantity_enhancement
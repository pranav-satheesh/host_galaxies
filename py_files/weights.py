import numpy as np
from scipy.stats import ks_2samp
from scipy import stats
import matplotlib.pyplot as plt
import h5py
import time

def weight_Mstar(i,pop,control):
    return 1  - np.abs(np.log10(pop["merging_population"]["Mstar"][i])- np.log10(control["Mstar"][i]))/control["Mstar_dex_tol"][i]

def weight_z(i,pop,control):
    return 1  - np.abs(np.log10(pop["merging_population"]["z"][i])- np.log10(control["z"][i]))/control["z_tol"][i]

def weight_tot(i,pop,control):
    return weight_z(i,pop,control)*weight_Mstar(i,pop,control)

def control_prop_avg(prop_key,index,pop,control):
    control_pop = control[prop_key][index]
    control_weights = weight_tot(index,pop,control)
    control_avg = np.sum(control_weights*control_pop)/np.sum(control_weights)
    return control_avg



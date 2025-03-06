import illustris_python as il
import matplotlib.pyplot as plt
import h5py
import numpy as np
import scienceplots
plt.style.use('science')


pop_file_loc = "/home/pranavsatheesh/host_galaxies/merger_file_runs/"
pop_file = pop_file_loc + "population_sort_gas-100_dm-100_star-100_bh-001.hdf5"
pop = h5py.File(pop_file, 'r')
Mstar_merging = pop['merging_population']['Mstar'][:]
Mstar_non_merging = pop['non_merging_population']['Mstar'][:]

plt.figure(figsize=(6,6))
plt.hist(np.log10(Mstar_merging), bins=10, label='Merging',histtype='step',density=True)
plt.hist(np.log10(Mstar_non_merging), bins=10, label='Non Merging',histtype='step',density=True)
plt.xlabel('$\log_{10}(M_{*})$')
plt.ylabel('Density')
plt.legend()
plt.savefig('Mstar_hist.png')
plt.show()
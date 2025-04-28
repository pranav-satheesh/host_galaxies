import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt
import h5py
import sys
from scipy.spatial import cKDTree

import sys
sys.path.append('../BH_dynamics_analysis')
sys.path.append('/home/pranavsatheesh/arepo_package/')
import arepo_package as arepo



MSOL = 1.988409870698051e+33
h = 0.6774

class population_generator:
     
    def __init__(self,merger_file,basePath,minN_values):
        
        self.base_path = basePath
        self.minN_values = minN_values
        fmergers = h5py.File(merger_file, 'r')
        self.h = fmergers.attrs['HubbleParam']
        self.box_length = fmergers.attrs['box_volume_mpc']**(1/3) * 1000 #in comoving units ckpc

        self.subhaloidxs_galaxy_mergers = fmergers["shids_subf"][:,2] # the subfind subhalo ID of the descendent correspond to the index in the subhalo catalogue
        self.snaps_galaxy_mergers = fmergers['snaps'][:,2]
        self.z_galaxy_mergers = 1/(fmergers['time'][:][:,2])-1
        snap_list = np.arange(np.min(self.snaps_galaxy_mergers),100)

        BHMasses_merger = fmergers['SubhaloBHMass'][:]/MSOL
        q_mergers, major_merger_mask = self.find_major_mergers(BHMasses_merger)
        self.major_merger_mask = major_merger_mask
        self.q_mergers = q_mergers
        self.M1_prog = BHMasses_merger[:,0]
        self.M2_prog = BHMasses_merger[:,1]
        self.M_final = BHMasses_merger[:,2]

        self.all_merging_population,self.non_merging_population = self.generate_population(snap_list)

        #self.all_merging_population,self.major_merger_population,self.non_merging_population = self.generate_population(snap_list)
        #generate_pop_dict_for_all_cases()
          
        #generate_pop_dict_for_major_mergers()  
     
    def find_major_mergers(self,BHMasses):
        q_merger_list = []
        for mass in BHMasses:
            if mass[0] > mass[1]:
                 q_merger = mass[1]/mass[0]
            else:
                q_merger = mass[0]/mass[1]
            q_merger_list.append(q_merger)
               
        q_merger_list = np.array(q_merger_list)
        major_merger_mask = [True if qi > 0.1 else False for qi in q_merger_list]
        
        return q_merger_list,major_merger_mask
    
    def generate_population(self, snap_population):
        redshifts = np.array([il.groupcat.loadHeader(self.base_path, snap)['Redshift'].item() 
                              for snap in snap_population])
        all_merging_population = self.initialize_population_dict(key="M")
        #major_merger_population = self.initialize_population_dict(key="M")
        non_merging_population = self.initialize_population_dict(key="N")

        for i, snap in enumerate(snap_population):
            subhalos = il.groupcat.loadSubhalos(
                self.base_path, snap, 
                fields=['SubhaloLenType', 'SubhaloMassType', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloGasMetallicity','SubhaloStarMetallicity','SubhaloPos','SubhaloHalfmassRadType']
            )
            valid_subhalo_ids,rsep_subhalos = self.get_valid_subhalo_ids(subhalos)
            
            # Find merger events corresponding to this snapshot from the merger file.
            merger_indices = np.where(self.snaps_galaxy_mergers == snap)[0]
            if merger_indices.size > 0:
                # All merging events for this snapshot.
                #subhalo_ids_merging_all = self.subhaloidxs_galaxy_mergers[merger_indices]
                # Only major mergers (using the mask).
                major_mask = np.array(self.major_merger_mask)[merger_indices]
                major_indices = merger_indices[major_mask]
                #major_indices = merger_indices[self.major_merger_mask[merger_indices]]
                #subhalo_ids_merging_major = self.subhaloidxs_galaxy_mergers[major_indices]
                subhalo_ids_merging = self.subhaloidxs_galaxy_mergers[major_indices]
            else:
                subhalo_ids_merging = np.array([], dtype=int)
                #subhalo_ids_merging_major = np.array([], dtype=int)
            
            # Non-merging population: valid subhalos excluding any merging events (all cases).
            subhalo_ids_non_merging = np.setdiff1d(valid_subhalo_ids, subhalo_ids_merging)

            # Update population dictionaries.
            self.update_population_dict(all_merging_population, subhalos, subhalo_ids_merging, snap, redshifts[i])
            #self.update_population_dict(major_merger_population, subhalos, subhalo_ids_merging_major, snap, redshifts[i])
            self.update_population_dict(non_merging_population, subhalos, subhalo_ids_non_merging, snap, redshifts[i])

        # Convert units for each population.
        self.convert_units(all_merging_population)
        #self.convert_units(major_merger_population)
        self.convert_units(non_merging_population)

        all_merging_population["MBH_1"] = np.append(all_merging_population["MBH_1"], self.M1_prog[self.major_merger_mask])
        all_merging_population["MBH_2"] = np.append(all_merging_population["MBH_2"], self.M2_prog[self.major_merger_mask])
        all_merging_population["q_merger"] = np.append(all_merging_population["q_merger"], self.q_mergers[self.major_merger_mask])

        # major_merger_population["MBH_1"] = np.append(major_merger_population["MBH_1"], self.M1_prog[self.major_merger_mask])
        # major_merger_population["MBH_2"] = np.append(major_merger_population["MBH_2"], self.M2_prog[self.major_merger_mask])
        # major_merger_population["q_merger"] = np.append(major_merger_population["q_merger"], self.q_mergers[self.major_merger_mask])

        # return all_merging_population, major_merger_population, non_merging_population
        return all_merging_population, non_merging_population
          
    def initialize_population_dict(self,key="M"):

        if(key=="M"): #for merging population
            return {
                "snap": np.array([], dtype=int),
                "z": np.array([], dtype=float),
                "subhalo_ids": np.array([], dtype=int),
                "Mstar": np.array([], dtype=float),
                "Mgas": np.array([], dtype=float),
                "MBH": np.array([], dtype=float),
                "MBH_1":np.array([], dtype=float),
                "MBH_2":np.array([], dtype=float),
                "q_merger":np.array([], dtype=float),
                "Mdot": np.array([], dtype=float),
                "SFR": np.array([], dtype=float)
            }
        else:
            return {
                "snap": np.array([], dtype=int),
                "z": np.array([], dtype=float),
                "subhalo_ids": np.array([], dtype=int),
                "Mstar": np.array([], dtype=float),
                "Mgas": np.array([], dtype=float),
                "MBH": np.array([], dtype=float),
                "Mdot": np.array([], dtype=float),
                "SFR": np.array([], dtype=float)
            }
        
    def get_valid_subhalo_ids(self, subhalos):

        #applying the minimum particle cut

        Ngas = subhalos['SubhaloLenType'][:, 0]
        Ndm = subhalos['SubhaloLenType'][:, 1]
        Nstar = subhalos['SubhaloLenType'][:, 4]
        Nbh = subhalos['SubhaloLenType'][:, 5]
        subhalo_ids = np.arange(len(Ngas))
        #particle_cut_mask = (Ngas >= self.minN_values[1]) & (Ndm >= self.minN_values[0]) & (Nstar >= self.minN_values[2]) & (Nbh >= 0)
        particle_cut_mask = (Ngas >= self.minN_values[1]) & (Ndm >= self.minN_values[0]) & (Nstar >= self.minN_values[2]) & (Nbh >= self.minN_values[3])

        #applying the condition that the nearest neighbour sep rsep >2 for non interacting galaxies

        valid_subhalo_ids = subhalo_ids[particle_cut_mask]
        valid_subhalo_pos = subhalos['SubhaloPos'][particle_cut_mask]

        tree = cKDTree(valid_subhalo_pos, boxsize=self.box_length)
        distances, nearest_neighbors = tree.query(valid_subhalo_pos,k=2)

        nearest_subhalo_ids = nearest_neighbors[:,-1]
        r_subhalos = distances[:,-1]

        stellar_half_mass_radius = subhalos['SubhaloHalfmassRadType'][:,4]/h

        r_sep = r_subhalos/(stellar_half_mass_radius[valid_subhalo_ids]+stellar_half_mass_radius[nearest_subhalo_ids])

        valid_sep_mask = r_sep>2

        return valid_subhalo_ids[valid_sep_mask],r_sep

    def update_population_dict(self, population, subhalos, subhalo_ids, snap, redshift):
        if len(subhalo_ids) == 0:
            return
        population["snap"] = np.append(population["snap"], snap * np.ones(len(subhalo_ids)))
        population["z"] = np.append(population["z"], redshift * np.ones(len(subhalo_ids)))
        population["subhalo_ids"] = np.concatenate((population["subhalo_ids"], subhalo_ids))
        population["Mstar"] = np.concatenate((population["Mstar"], subhalos['SubhaloMassType'][subhalo_ids, 4]))
        population["Mgas"] = np.concatenate((population["Mgas"], subhalos['SubhaloMassType'][subhalo_ids, 0]))
        population["MBH"] = np.concatenate((population["MBH"], subhalos['SubhaloBHMass'][subhalo_ids]))
        population["Mdot"] = np.concatenate((population["Mdot"], subhalos['SubhaloBHMdot'][subhalo_ids]))
        population["SFR"] = np.concatenate((population["SFR"], subhalos['SubhaloSFR'][subhalo_ids]))
        
        # population["SubhaloPos"] = np.concatenate((population["SubhaloPos"], subhalos['SubhaloPos'][subhalo_ids]))
        # population["SubhaloHalfmassRadType"] = np.concatenate((population["SubhaloHalfmassRadType"], subhalos['SubhaloHalfmassRadType'][subhalo_ids]))

    def convert_units(self, population):
        population['Mstar'] *= 1e10 / self.h
        population['Mgas'] *= 1e10 / self.h
        population['MBH'] *= 1e10 / self.h
        population['Mdot'] *= (1e10 / self.h) / (0.978 * 1e9 / self.h)
          
    def save_population_to_file(self, filepath):
        """Save all cases (all merging and non-merging populations) to file."""
        outfilename = filepath + f"/population_sort_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}_w_rsep_cut_1bh.hdf5"

        with h5py.File(outfilename, 'w') as f:
            all_merge_grp = f.create_group('merging_population')
            for key, value in self.all_merging_population.items():
                all_merge_grp.create_dataset(key, data=value)
            
            non_merge_grp = f.create_group('non_merging_population')
            for key, value in self.non_merging_population.items():
                non_merge_grp.create_dataset(key, data=value)
            
            # major_grp = f.create_group('major_merging_population')
            # for key, value in self.major_merger_population.items():
            #     major_grp.create_dataset(key, data=value)

        print(f"All-cases population saved to {outfilename}")


class population_generator_for_brahma:

    def __init__(self,basePath,merger_file_name):

        self.basePath = basePath
        self.h = arepo.get_cosmology(basePath)[2]  #h parameter associated with this run
        
        bh_merger_properties = np.load(basePath + merger_file_name,allow_pickle=True)[0]
        true_merger_flag = np.array(bh_merger_properties['merger_flag'])==1
        print("The number of BH mergers in this run is %d"%(np.sum(true_merger_flag)))
        self.N_mergers = np.sum(true_merger_flag)

        brahma_merger_data = {}

        for key in bh_merger_properties.keys():
            brahma_merger_data[key] = np.array(bh_merger_properties[key])[true_merger_flag]

        self.brahma_merger_data = brahma_merger_data
        self.z_bh_mergers = brahma_merger_data['remnant_redshift']

        #get the snapshots for the BH mergers (remnant subhalo)
        self.snaps_bh_mergers = []
        snap_list, z_list=arepo.get_snapshot_redshift_correspondence(basePath)
        self.snap_list = snap_list
        self.z_list = z_list

        for remnant_z in self.z_bh_mergers:
            idx = np.where(z_list == remnant_z)[0]
            self.snaps_bh_mergers.append(snap_list[idx][0])
        
        self.merging_population,self.non_merging_population = self.generate_population()


    def generate_population(self):

        merging_population = self.initialize_population_dict(key='M')
        non_merging_population = self.initialize_population_dict(key='N')

        fields=['SubhaloLenType', 'SubhaloMassType', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloGasMetallicity','SubhaloStarMetallicity','SubhaloPos','SubhaloHalfmassRadType']

        for i,redshift in enumerate(self.z_list):
            subhalos = arepo.get_subhalo_property(self.basePath,fields,redshift,postprocessed=1)
            subhalo_ids = np.arange(len(subhalos['SubhaloLenType'][:,0]))

            #find all mergers within this snapshot/redshift
            merger_indices = np.where(self.brahma_merger_data['remnant_redshift']==redshift)
            if merger_indices.size > 0:
                subhalo_ids_merging = self.brahma_merger_data['remnant_SubhaloID'][merger_indices]

            else:
                subhalo_ids_merging = np.array([], dtype=int)

            #TO-DO: change this to include the subhalo cuts to select only valid subhalos
            valid_subhalo_ids = subhalo_ids
            subhalo_ids_non_merging = np.setdiff1d(valid_subhalo_ids, subhalo_ids_merging)

            self.update_population_dict(merging_population, subhalos, subhalo_ids_merging,self.snap_list[i],redshift)
            #self.update_population_dict(major_merger_population, subhalos, subhalo_ids_merging_major, snap, redshifts[i])
            self.update_population_dict(non_merging_population, subhalos, subhalo_ids_non_merging,self.snap_list[i],redshift)
        
        # Convert units for each population.
        self.convert_units(merging_population)
        self.convert_units(non_merging_population)

        return merging_population, non_merging_population
            
    def initialize_population_dict(self,key="M"):

        if(key=="M"): #for merging population
            return {
                "snap": np.array([], dtype=int),
                "z": np.array([], dtype=float),
                "subhalo_ids": np.array([], dtype=int),
                "Mstar": np.array([], dtype=float),
                "Mgas": np.array([], dtype=float),
                "MBH": np.array([], dtype=float),
                #"MBH_1":np.array([], dtype=float),
                #"MBH_2":np.array([], dtype=float),
                # "q_merger":np.array([], dtype=float),
                "Mdot": np.array([], dtype=float),
                "SFR": np.array([], dtype=float)
            }
        else:
            return {
                "snap": np.array([], dtype=int),
                "z": np.array([], dtype=float),
                "subhalo_ids": np.array([], dtype=int),
                "Mstar": np.array([], dtype=float),
                "Mgas": np.array([], dtype=float),
                "MBH": np.array([], dtype=float),
                "Mdot": np.array([], dtype=float),
                "SFR": np.array([], dtype=float)
            }

    def update_population_dict(self, population, subhalos, subhalo_ids, snap, redshift):
        if len(subhalo_ids) == 0:
            return
        population["snap"] = np.append(population["snap"], snap * np.ones(len(subhalo_ids)))
        population["z"] = np.append(population["z"], redshift * np.ones(len(subhalo_ids)))
        population["subhalo_ids"] = np.concatenate((population["subhalo_ids"], subhalo_ids))
        population["Mstar"] = np.concatenate((population["Mstar"], subhalos['SubhaloMassType'][subhalo_ids, 4]))
        population["Mgas"] = np.concatenate((population["Mgas"], subhalos['SubhaloMassType'][subhalo_ids, 0]))
        population["MBH"] = np.concatenate((population["MBH"], subhalos['SubhaloBHMass'][subhalo_ids]))
        population["Mdot"] = np.concatenate((population["Mdot"], subhalos['SubhaloBHMdot'][subhalo_ids]))
        population["SFR"] = np.concatenate((population["SFR"], subhalos['SubhaloSFR'][subhalo_ids]))

    def convert_units(self, population):
        population['Mstar'] *= 1e10 / self.h
        population['Mgas'] *= 1e10 / self.h
        population['MBH'] *= 1e10 / self.h
        population['Mdot'] *= (1e10 / self.h) / (0.978 * 1e9 / self.h)
          
    def save_population_to_file(self, filepath):
        """Save all cases (all merging and non-merging populations) to file."""
        outfilename = filepath + f"/population_from_brahma.hdf5"

        with h5py.File(outfilename, 'w') as f:
            all_merge_grp = f.create_group('merging_population')
            for key, value in self.all_merging_population.items():
                all_merge_grp.create_dataset(key, data=value)
            
            non_merge_grp = f.create_group('non_merging_population')
            for key, value in self.non_merging_population.items():
                non_merge_grp.create_dataset(key, data=value)
            
            # major_grp = f.create_group('major_merging_population')
            # for key, value in self.major_merger_population.items():
            #     major_grp.create_dataset(key, data=value)

        print(f"All-cases population saved to {outfilename}")

if __name__ == "__main__":

    if(sys.argv[1] == "brahma"):
        basePath = sys.argv[2]
        merger_file_name = sys.argv[3]
        population_file_path = sys.argv[4]

        pop_gen = population_generator_for_brahma(basePath,merger_file_name)
        pop_gen.save_population_to_file(population_file_path)
        exit()

    else:
        basePath = sys.argv[1]
        merger_file_path = sys.argv[2]
        population_file_path = sys.argv[3]

        minN_values = np.array([int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])])

        merger_file_1bh = merger_file_path + f'/galaxy-mergers_TNG50-1_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5'

        pop_gen = population_generator(merger_file_1bh,basePath,minN_values)

        pop_gen.save_population_to_file(population_file_path)
    
    # fmergers = h5py.File(merger_file_1bh, 'r')
    # subhaloidxs_galaxy_mergers = fmergers["shids_subf"][:,2]
    # snaps_galaxy_mergers = fmergers['snaps'][:,2]
    # z_galaxy_mergers = 1/(fmergers['time'][:][:,2])-1
    # h = fmergers.attrs['HubbleParam']

    # snap_list = np.arange(np.min(snaps_galaxy_mergers),100)
    # #the minimum value will be the snapshot where the max redshift for gaalxy mergers is reached
    
    # population_sort_file_name = merger_file_path
    # write_population_to_file(population_sort_file_name, basePath, snap_list, snaps_galaxy_mergers, subhaloidxs_galaxy_mergers, minN_values)

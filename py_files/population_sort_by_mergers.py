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
import BRAHMA_python as il_brahma
from tqdm import tqdm

MSOL = 1.988409870698051e+33
h = 0.6774

class population_generator:
     
    def __init__(self,merger_file,mergerfilePath,basePath,minN_values):
        
        self.base_path = basePath
        self.minN_values = minN_values
        fmergers = h5py.File(merger_file, 'r')
        self.h = fmergers.attrs['HubbleParam']
        self.box_length = fmergers.attrs['box_volume_mpc']**(1/3) * 1000 #in comoving units ckpc

        self.subhaloidxs_galaxy_mergers = fmergers["shids_subf"][:,2] # the subfind subhalo ID of the descendent correspond to the index in the subhalo catalogue
        self.snaps_galaxy_mergers = fmergers['snaps'][:,2]
        self.z_galaxy_mergers = 1/(fmergers['time'][:][:,2])-1
        snap_list = np.arange(np.min(self.snaps_galaxy_mergers),100)

        prog_mass_ratio = fmergers['ProgMassRatio_mod'][:]
        prog_mass_ratio[prog_mass_ratio>1] = 1/prog_mass_ratio[prog_mass_ratio>1]
        self.prog_mass_ratio = prog_mass_ratio
        self.major_merger_mask = self.prog_mass_ratio > 0.1


        BHMasses_merger = fmergers['SubhaloBHMass'][:]/MSOL
        #q_mergers, major_merger_mask = self.find_major_mergers(BHMasses_merger)
        
        #elf.q_mergers = q_mergers
        self.M1_prog = BHMasses_merger[:,0]
        self.M2_prog = BHMasses_merger[:,1]
        self.M_final = BHMasses_merger[:,2]

        descendant_file = mergerfilePath+'/descendants_after_2Gyr_of_mergers.hdf5'
        self.fdescendant = h5py.File(descendant_file, 'r')

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
                fields=['SubhaloLenType', 'SubhaloMassType', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloGasMetallicity','SubhaloStarMetallicity','SubhaloPos','SubhaloHalfmassRadType','SubhaloMassInRadType','SubhaloStellarPhotometrics']
            )

            valid_subhalo_ids,r_sep_valid_subhalos = self.get_valid_subhalo_ids(subhalos,snap)

            subhalo_ids = np.arange(len(subhalos['SubhaloLenType'][:, 0]))
            #valid_subhalo_ids = subhalo_ids[valid_subhalo_mask]
    
            # Find merger events corresponding to this snapshot from the merger file.
            merger_indices = np.where(self.snaps_galaxy_mergers == snap)[0]

            if merger_indices.size > 0:
                # All merging events for this snapshot.
                #subhalo_ids_merging_all = self.subhaloidxs_galaxy_mergers[merger_indices]
                # Only major mergers (using the mask).
                major_mask = np.array(self.major_merger_mask)[merger_indices]
                major_indices = merger_indices[major_mask]
                subhalo_ids_merging = self.subhaloidxs_galaxy_mergers[major_indices]
                
            else:
                subhalo_ids_merging = np.array([], dtype=int)
                #subhalo_ids_merging_major = np.array([], dtype=int)
            
            #merging_subhalo_ids_mask = np.isin(valid_subhalo_ids,subhalo_ids_merging)
            #r_sep_merging = r_sep_all_subhalos[valid_subhalo_ids][merging_subhalo_ids_mask]

            r_sep_merging = self.get_rsep_for_mergers(subhalos,subhalo_ids_merging)
            #get rsep for merging subhalos
            #merging_subhalo_ids_mask = np.isin(valid_subhalo_ids, subhalo_ids_merging)
            #r_sep_merging = r_sep_all_subhalos_w_particle_cut[merging_subhalo_ids_mask]

            # Non-merging population: valid subhalos excluding any merging events (all cases).
            subhalo_ids_non_merging = np.setdiff1d(valid_subhalo_ids, subhalo_ids_merging)

            non_merging_mask = np.isin(valid_subhalo_ids, subhalo_ids_non_merging)

            r_sep_non_merging = r_sep_valid_subhalos[non_merging_mask]
           
            # Update population dictionaries.
            self.update_population_dict(all_merging_population, subhalos, subhalo_ids_merging, snap, redshifts[i],r_sep_merging)
            #self.update_population_dict(major_merger_population, subhalos, subhalo_ids_merging_major, snap, redshifts[i])
            self.update_population_dict(non_merging_population, subhalos, subhalo_ids_non_merging, snap, redshifts[i],r_sep_non_merging)

        # Convert units for each population.
        self.convert_units(all_merging_population)
        #self.convert_units(major_merger_population)
        self.convert_units(non_merging_population)

        all_merging_population["MBH_1"] = np.append(all_merging_population["MBH_1"], self.M1_prog[self.major_merger_mask])
        all_merging_population["MBH_2"] = np.append(all_merging_population["MBH_2"], self.M2_prog[self.major_merger_mask])
        all_merging_population["MBH_final"] = np.append(all_merging_population["MBH_final"], self.M_final[self.major_merger_mask])
        all_merging_population["q_merger"] = np.append(all_merging_population["q_merger"], self.prog_mass_ratio[self.major_merger_mask])

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
                "MBH_final": np.array([], dtype=float),
                "q_merger":np.array([], dtype=float),
                "Mdot": np.array([], dtype=float),
                "SFR": np.array([], dtype=float),
                "rsep": np.array([], dtype=float),
                "Mstar-half": np.array([], dtype=float),
                "Mgas-half": np.array([], dtype=float)
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
                "SFR": np.array([], dtype=float),
                "rsep": np.array([], dtype=float),
                "Mstar-half": np.array([], dtype=float),
                "Mgas-half": np.array([], dtype=float)
            }
        
    def get_valid_subhalo_ids(self,subhalos,snap):

        #applying the minimum particle cut

        Ngas = subhalos['SubhaloLenType'][:, 0]
        Ndm = subhalos['SubhaloLenType'][:, 1]
        Nstar = subhalos['SubhaloLenType'][:, 4]
        Nbh = subhalos['SubhaloLenType'][:, 5]
        subhalo_ids = np.arange(len(Ngas))
        #particle_cut_mask = (Ngas >= self.minN_values[1]) & (Ndm >= self.minN_values[0]) & (Nstar >= self.minN_values[2]) & (Nbh >= 0)
        particle_cut_mask = (Ngas >= self.minN_values[1]) & (Ndm >= self.minN_values[0]) & (Nstar >= self.minN_values[2]) & (Nbh >= self.minN_values[3])

        #remove subhalos that are descendant of a recent major galaxy merger (within the past 2 Gyr)
        
        #relaxing this requirement for now
        #descendant_of_major_mergers_snap = self.fdescendant['snap'][:]
        #descendant_of_major_mergers_subhalo_ids = self.fdescendant['shids_subfind'][:]

        #descend_snaps = np.where(descendant_of_major_mergers_snap==snap)[0]
        #descend_subhalo_ids = descendant_of_major_mergers_subhalo_ids[descend_snaps]
        #non_merger_descendant_mask = ~np.isin(subhalo_ids, descend_subhalo_ids)
        #particle_and_non_merger_descendent_mask = particle_cut_mask & non_merger_descendant_mask
        #valid_subhalo_pos = subhalos['SubhaloPos'][particle_and_non_merger_descendent_mask]
        
      
        #valid_subhalo_ids = subhalo_ids[particle_cut_mask & non_merger_descendant_mask]
        #valid_subhalo_pos = subhalos['SubhaloPos'][particle_cut_mask & non_merger_descendant_mask] 

        #valid_subhalo_ids = subhalo_ids[particle_cut_mask]
        valid_subhalo_pos = subhalos['SubhaloPos'][particle_cut_mask]  #ckpc/h
    
        
        #applying the condition that the nearest neighbour sep rsep >2 for non interacting galaxies
        #subhalo_pos = subhalos['SubhaloPos']
        #tree = cKDTree(subhalo_pos,boxsize=self.box_length)
        #distances, nearest_neighbors = tree.query(subhalo_pos,k=2)

        tree = cKDTree(valid_subhalo_pos, boxsize=self.box_length)
        distances, nearest_neighbors = tree.query(valid_subhalo_pos,k=2)

        nearest_subhalo_ids = nearest_neighbors[:,-1]
        r_subhalos = distances[:,-1]

        stellar_half_mass_radius = subhalos['SubhaloHalfmassRadType'][:,4] #ckpc/h

        r_sep = r_subhalos/(stellar_half_mass_radius[particle_and_non_merger_descendent_mask]+stellar_half_mass_radius[nearest_subhalo_ids])

        #r_sep = r_subhalos/(stellar_half_mass_radius[valid_subhalo_ids]+stellar_half_mass_radius[nearest_subhalo_ids])

        valid_sep_mask = r_sep>2
        valid_rsep = r_sep[valid_sep_mask]
        #print(len(r_sep),len(valid_sep_mask),len(particle_and_non_merger_descendent_mask))

        #valid_subhalo_mask = particle_and_non_merger_descendent_mask& valid_sep_mask

        #valid_subhalo_mask[np.where(particle_and_non_merger_descendent_mask)[0][valid_sep_mask]] = True

        valid_subhalo_ids = subhalo_ids[particle_cut_mask][valid_sep_mask]
        #valid_subhalo_ids = subhalo_ids[particle_and_non_merger_descendent_mask][valid_sep_mask]

        return valid_subhalo_ids,valid_rsep

        #return valid_subhalo_mask,valid_sep_mask,non_merger_descendant_mask,particle_cut_mask,r_sep

    def get_rsep_for_mergers(self, subhalos, subhalo_ids_merging):

        if len(subhalo_ids_merging) == 0:
            return np.array([])
        
        else:
            subhalo_pos_mergers = subhalos['SubhaloPos'][subhalo_ids_merging]/h #ckpc

            tree = cKDTree(subhalo_pos_mergers, boxsize=self.box_length)
            distances, nearest_neighbors = tree.query(subhalo_pos_mergers,k=2)

            nearest_subhalo_ids = nearest_neighbors[:,-1]
            r_subhalos = distances[:,-1]

            stellar_half_mass_radius = subhalos['SubhaloHalfmassRadType'][:,4]/h #ckpc

            r_sep = r_subhalos/(stellar_half_mass_radius[subhalo_ids_merging]+stellar_half_mass_radius[nearest_subhalo_ids])

            return r_sep

    def update_population_dict(self, population, subhalos, subhalo_ids, snap, redshift,r_sep):
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
        population["rsep"] = np.concatenate((population["rsep"], r_sep))
        population["Mstar-half"]=np.concatenate((self.non_merging_pop["Mstar-half"], subhalos['SubhaloMassInHalfRadType'][:,4][subhalo_ids_non_merging]))
        population["Mgas-half"] = np.concatenate((self.non_merging_pop["Mgas-half"], subhalos['SubhaloMassInHalfRadType'][:,0][subhalo_ids_non_merging]))

        
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

    def __init__(self,basePath,merger_file_path):
        
        self.basePath = basePath
        self.h = il_brahma.groupcat.loadHeader(self.basePath,0)['HubbleParam']
        self.runName = basePath.split('/')[-1]
        #self.runName = runname
        self.merger_file_path = merger_file_path 
        #self.h = arepo.get_cosmology(self.basePath)[2]  #h parameter associated with this run
        self.N_snaps = 32 #should change this later for different runs
        self.mrg_file = h5py.File(self.merger_file_path, 'r')
        self.box_length = mrg_file.attrs['box_volume_mpc']**(1/3) * 1000 #in comoving units ckpc
        
        brahma_snapshots,brahma_redshifts = arepo.get_snapshot_redshift_correspondence(self.basePath)

        self.unique_merger_snap = np.unique(self.mrg_file['snaps'][:,2])
        self.unique_merger_z = brahma_redshifts[self.unique_merger_snap]

        self.generate_population()

    def generate_population(self):

        self.merging_pop = self.initialize_population_dict(key='M')
        self.non_merging_pop = self.initialize_population_dict(key='N')

        fields=['SubhaloLenType', 'SubhaloMassType','SubhaloMass', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloGasMetallicity','SubhaloStarMetallicity','SubhaloPos','SubhaloHalfmassRadType','SubhaloMassInHalfRadType','SubhaloMassInRadType','SubhaloStellarPhotometrics']

        for i in tqdm(range(len(self.unique_merger_z)), desc="Processing unique merger redshifts"):

            # merger_idxs = np.where(self.brahma_merger_data['remnant_redshift'] == self.unique_merger_z[i])[0]

            merger_idxs = np.where(self.mrg_file['snaps'][:,2] == self.unique_merger_snap[i])[0]

            subhalos = il_brahma.groupcat.loadSubhalos_postprocessed(self.basePath, self.unique_merger_snap[i], fields)
            subhalo_ids_merging = self.mrg_file['shids_subf'][merger_idxs,2]

            Ngas = subhalos['SubhaloLenType'][:, 0]
            Ndm = subhalos['SubhaloLenType'][:, 1]
            Nstar = subhalos['SubhaloLenType'][:, 4]
            Nbh = subhalos['SubhaloLenType'][:, 5]
            subhalo_ids = np.arange(len(Ngas))
            particle_cut_mask = (Ngas >= minN_values[1]) & (Ndm >= minN_values[0]) & (Nstar >= minN_values[2]) & (Nbh >= minN_values[3])
            
            valid_subhalo_pos = subhalos['SubhaloPos'][particle_cut_mask]  #ckpc/h
            tree = cKDTree(valid_subhalo_pos, boxsize=self.box_length)
            distances, nearest_neighbors = tree.query(valid_subhalo_pos,k=2)
            nearest_subhalo_ids = nearest_neighbors[:,-1]
            r_subhalos = distances[:,-1]
            stellar_half_mass_radius = subhalos['SubhaloHalfmassRadType'][:,4] #ckpc/h
            r_sep = r_subhalos/(stellar_half_mass_radius[particle_cut_mask]+stellar_half_mass_radius[nearest_subhalo_ids])
            valid_sep_mask = r_sep>2
            r_sep_valid_subhalos = r_sep[valid_sep_mask]
            
            valid_subhalo_ids = subhalo_ids[particle_cut_mask][valid_sep_mask]
            subhalo_ids_non_merging = np.setdiff1d(valid_subhalo_ids, subhalo_ids_merging)

            self.update_merger_details(subhalos, subhalo_ids_merging, self.unique_merger_z[i], self.unique_merger_snap[i])
            self.update_non_merger_details(subhalos, subhalo_ids_non_merging, self.unique_merger_z[i], self.unique_merger_snap[i])

        self.update_pop_units(self.merging_pop)
        self.update_pop_units(self.non_merging_pop)
            
    def initialize_population_dict(self,key="M"):

        if(key=="M"): #for merging population
            return {
                "snap": np.array([], dtype=int),
                "z": np.array([], dtype=float),
                "subhalo_ids": np.array([], dtype=int),
                "Mstar": np.array([], dtype=float),
                "Mgas": np.array([], dtype=float),
                "MBH": np.array([], dtype=float),
                "Mdot": np.array([], dtype=float),
                "SFR": np.array([], dtype=float),
                "Mstar-half": np.array([], dtype=float),
                "Mgas-half": np.array([], dtype=float),
                "Msubhalo": np.array([], dtype=float)
                # "MBH1": np.array([], dtype=float),
                # "MBH2": np.array([], dtype=float)
            }
        else:
            return {
                "snap": [],
                "z": [],
                "subhalo_ids": [],
                "Mstar": [],
                "Mgas": [],
                "MBH": [],
                "Mdot": [],
                "SFR": [],
                "Mstar-half": [],
                "Mgas-half": [],
                "Msubhalo": []
            }

    def update_merger_details(self,subhalos,subhalo_ids_merging,redshift,snapnum):

        self.merging_pop["subhalo_ids"] = np.concatenate((self.merging_pop["subhalo_ids"], subhalo_ids_merging))
        self.merging_pop["snap"] = np.append(self.merging_pop["snap"], snapnum*np.ones(len(subhalo_ids_merging)))
        self.merging_pop["z"] = np.append(self.merging_pop["z"], redshift*np.ones(len(subhalo_ids_merging)))
        self.merging_pop["Mstar"] = np.concatenate((self.merging_pop["Mstar"], subhalos['SubhaloMassType'][:,4][subhalo_ids_merging]))
        # self.merging_pop["Mgas"] = np.concatenate((self.merging_pop["Mgas"], subhalos['SubhaloMassType'][:,0][subhalo_ids_merging]))
        self.merging_pop["Mgas"] = np.concatenate((self.merging_pop["Mgas"], subhalos['SubhaloMassInRadType'][:,0][subhalo_ids_merging]))
        self.merging_pop["MBH"] = np.concatenate((self.merging_pop["MBH"], subhalos['SubhaloBHMass'][subhalo_ids_merging]))
        self.merging_pop["Mdot"] = np.concatenate((self.merging_pop["Mdot"], subhalos['SubhaloBHMdot'][subhalo_ids_merging]))
        self.merging_pop["SFR"] = np.concatenate((self.merging_pop["SFR"], subhalos['SubhaloSFR'][subhalo_ids_merging]))
        self.merging_pop["Mstar-half"] = np.concatenate((self.merging_pop["Mstar-half"], subhalos['SubhaloMassInHalfRadType'][:,4][subhalo_ids_merging]))
        self.merging_pop["Mgas-half"] = np.concatenate((self.merging_pop["Mgas-half"], subhalos['SubhaloMassInHalfRadType'][:,0][subhalo_ids_merging]))
        self.merging_pop["Msubhalo"] = np.concatenate((self.merging_pop["Msubhalo"], subhalos['SubhaloMass'][subhalo_ids_merging]))

    def update_non_merger_details(self,subhalos,subhalo_ids_non_merging,redshift,snapnum):

            self.non_merging_pop["subhalo_ids"] = np.concatenate((self.non_merging_pop["subhalo_ids"], subhalo_ids_non_merging))
            self.non_merging_pop["snap"] = np.append(self.non_merging_pop["snap"], snapnum*np.ones(len(subhalo_ids_non_merging)))
            self.non_merging_pop["z"] = np.append(self.non_merging_pop["z"], redshift*np.ones(len(subhalo_ids_non_merging)))
            self.non_merging_pop["Mstar"] = np.concatenate((self.non_merging_pop["Mstar"], subhalos['SubhaloMassType'][:,4][subhalo_ids_non_merging]))
            #self.non_merging_pop["Mgas"] = np.concatenate((self.non_merging_pop["Mgas"], subhalos['SubhaloMassType'][:,0][subhalo_ids_non_merging]))
            self.non_merging_pop["Mgas"] = np.concatenate((self.non_merging_pop["Mgas"], subhalos['SubhaloMassInRadType'][:,0][subhalo_ids_non_merging]))
            self.non_merging_pop["MBH"] = np.concatenate((self.non_merging_pop["MBH"], subhalos['SubhaloBHMass'][subhalo_ids_non_merging]))
            self.non_merging_pop["Mdot"] = np.concatenate((self.non_merging_pop["Mdot"], subhalos['SubhaloBHMdot'][subhalo_ids_non_merging]))
            self.non_merging_pop["SFR"] = np.concatenate((self.non_merging_pop["SFR"], subhalos['SubhaloSFR'][subhalo_ids_non_merging]))
            self.non_merging_pop["Mstar-half"]=np.concatenate((self.non_merging_pop["Mstar-half"], subhalos['SubhaloMassInHalfRadType'][:,4][subhalo_ids_non_merging]))
            self.non_merging_pop["Mgas-half"] = np.concatenate((self.non_merging_pop["Mgas-half"], subhalos['SubhaloMassInHalfRadType'][:,0][subhalo_ids_non_merging]))
            self.non_merging_pop["Msubhalo"] = np.concatenate((self.non_merging_pop["Msubhalo"], subhalos['SubhaloMass'][subhalo_ids_non_merging]))   
    
    def update_pop_units(self,pop):
        pop["Mstar"] *= 1e10/self.h
        pop["Mgas"] *= 1e10/self.h
        pop["MBH"] *= 1e10/self.h
        pop["Mdot"] *= (1e10/self.h)/ ( 0.978 * 1e9 /self.h)
        pop["Mstar-half"] *= 1e10/self.h
        pop["Mgas-half"] *= 1e10/self.h
        pop["Msubhalo"] *= 1e10/self.h
        # pop["MBH1"] *= 1e10/h
        # pop["MBH2"] *= 1e10/h
    

    def get_rsep_for_mergers(self, subhalos, subhalo_ids_merging):

            if len(subhalo_ids_merging) == 0:
                return np.array([])
            
            else:
                subhalo_pos_mergers = subhalos['SubhaloPos'][subhalo_ids_merging]/h #ckpc

                tree = cKDTree(subhalo_pos_mergers, boxsize=self.box_length)
                distances, nearest_neighbors = tree.query(subhalo_pos_mergers,k=2)

                nearest_subhalo_ids = nearest_neighbors[:,-1]
                r_subhalos = distances[:,-1]

                stellar_half_mass_radius = subhalos['SubhaloHalfmassRadType'][:,4]/h #ckpc

                r_sep = r_subhalos/(stellar_half_mass_radius[subhalo_ids_merging]+stellar_half_mass_radius[nearest_subhalo_ids])

                return r_sep


    def save_population_to_file(self, filepath):
        """Save all cases (all merging and non-merging populations) to file."""
        outfilename = filepath + f"/{self.runName}_population_sort_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}_brahma.hdf5"
        
        with h5py.File(outfilename, 'w') as f:
            all_merge_grp = f.create_group('merging_population')
            for key, value in self.merging_pop.items():
                all_merge_grp.create_dataset(key, data=value)
            
            non_merge_grp = f.create_group('non_merging_population')
            for key, value in self.non_merging_pop.items():
                non_merge_grp.create_dataset(key, data=value)
            
            # major_grp = f.create_group('major_merging_population')
            # for key, value in self.major_merger_population.items():
            #     major_grp.create_dataset(key, data=value)

        print(f"All-cases population saved to {outfilename}")

if __name__ == "__main__":

    if(sys.argv[1] == "brahma"):

        basePath = sys.argv[2]
        merger_file_loc = sys.argv[3]
        pop_file_save_loc = sys.argv[4]

        minN_values = np.array([int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])])

        merger_file_name = f'/galaxy-mergers_brahma_{basePath.split("/")[-1]}_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5'
        merger_file_path = merger_file_loc + merger_file_name

        pop_gen = population_generator_for_brahma(basePath,merger_file_path)
        pop_gen.save_population_to_file(pop_file_save_loc)

        exit()

    else:
        basePath = sys.argv[1]
        merger_file_path = sys.argv[2]
        population_file_path = sys.argv[3]

        minN_values = np.array([int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])])

        merger_file_1bh = merger_file_path + f'/galaxy-mergers_TNG50-1_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5'

        pop_gen = population_generator(merger_file_1bh,merger_file_path,basePath,minN_values)

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

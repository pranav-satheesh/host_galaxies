import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt
import h5py
import sys
from scipy.spatial import cKDTree
sys.path.append('../BH_dynamics_analysis')
sys.path.append('/home/pranavsatheesh/arepo_package/')
import arepo_package as arepo
import BRAHMA_python as il_brahma
from tqdm import tqdm

MSOL = 1.988409870698051e+33
h = 0.6774


class pop_generator:

    def __init__(self,basePath,merger_file_loc,minN_values,brahma_key):

        self.basePath = basePath
        self.brahma_key = brahma_key
        self.merger_file_loc = merger_file_loc

        if self.brahma_key:
            merger_file_name = f'/galaxy-mergers_brahma_{basePath.split("/")[-1]}_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5'
        else:
            merger_file_name = f'/galaxy-mergers_TNG50-1_gas-{minN_values[0]:03d}_dm-{minN_values[1]:03d}_star-{minN_values[2]:03d}_bh-{minN_values[3]:03d}.hdf5'

        self.merger_file_path = merger_file_loc + merger_file_name
        fmergers = h5py.File(self.merger_file_path, 'r')
        self.simName = basePath.split('/')[-2] 
        self.box_length = fmergers.attrs['box_volume_mpc']**(1/3) * 1000
        self.minN_values = minN_values

        #galaxy merger snapshots and snapshots

        self.snap_prog1 = fmergers['snaps'][:,0]
        self.snap_prog2 = fmergers['snaps'][:,1]
        self.snap_remnant = fmergers['snaps'][:,2]
        self.subhaloidx_prog1 = fmergers['shids_subf'][:,0]
        self.subhaloidx_prog2 = fmergers['shids_subf'][:,1]
        self.subhaloidx_remnant = fmergers['shids_subf'][:,2]

        if(brahma_key):
            self.h = fmergers.attrs['hubbleParam']
            self.N_last_snap = fmergers.attrs['snapshots'][-1]
            self.brahma_snapshots,self.brahma_redshifts = arepo.get_snapshot_redshift_correspondence(self.basePath)
            self.z_prog1 = self.brahma_redshifts[self.snap_prog1]
            self.z_prog2 = self.brahma_redshifts[self.snap_prog2]
            self.z_remnant = self.brahma_redshifts[self.snap_remnant]
            #self.prog_mass_ratio = fmergers['ProgMassRatio_mod'][:]

        else:
            self.h = fmergers.attrs['HubbleParam'] 
            self.z_prog1 = 1/(fmergers['time'][:][:,0])-1
            self.z_prog2 = 1/(fmergers['time'][:][:,1])-1
            self.z_remnant = 1/(fmergers['time'][:][:,2])-1
            prog_mass_ratio = fmergers['ProgMassRatio_mod'][:]
            prog_mass_ratio[prog_mass_ratio>1] = 1/prog_mass_ratio[prog_mass_ratio>1]
            self.prog_mass_ratio = prog_mass_ratio

        self.generate_population()
        
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
                    "MgasInRad": np.array([], dtype=float),
                    "MstarInRad": np.array([], dtype=float),
                    "Msubhalo": np.array([], dtype=float),
                    "StellarHalfmassRad": np.array([], dtype=float),
                    "q_merger": np.array([], dtype=float),
                    "prog_snap": np.array([], dtype=int),
                    "prog_subhalo_id": np.array([], dtype=int),
                    "prog_redshift": np.array([], dtype=float),
                    "prog_Mstar": np.array([], dtype=float),
                    "prog_Mgas": np.array([], dtype=float),
                    "prog_MBH": np.array([], dtype=float),
                    "prog_Mdot": np.array([], dtype=float),
                    "prog_SFR": np.array([], dtype=float),
                    "prog_MgasInRad": np.array([], dtype=float),
                    "prog_MstarInRad": np.array([], dtype=float),
                    "prog_StellarHalfmassRad": np.array([], dtype=float),
                    "SubhaloPhotoMag": np.empty((0, 8), dtype=float)
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
                    "Msubhalo": np.array([], dtype=float),
                    "SFR": np.array([], dtype=float),
                    "MgasInRad": np.array([], dtype=float),
                    "MstarInRad": np.array([], dtype=float),
                    "StellarHalfmassRad": np.array([], dtype=float),
                    "SubhaloPhotoMag": np.empty((0, 8), dtype=float)
                }
      
    def generate_population(self):
        self.unique_snaps = np.unique(self.snap_remnant)

        self.merging_pop = self.initialize_population_dict(key="M")
        self.non_merging_pop = self.initialize_population_dict(key="N")

        fields=['SubhaloLenType', 'SubhaloMassType','SubhaloMass', 'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloGasMetallicity','SubhaloStarMetallicity','SubhaloPos','SubhaloHalfmassRadType','SubhaloMassInHalfRadType',
                'SubhaloMassInRadType','SubhaloStellarPhotometrics']

        if(self.brahma_key==False):
            self.unique_redshifts = np.array([il.groupcat.loadHeader(self.basePath, snap)['Redshift'].item() 
                              for snap in self.unique_snaps])
        else:
            self.unique_redshifts = self.brahma_redshifts[self.unique_snaps]

        for i, snap in enumerate(self.unique_snaps):
            if(self.brahma_key==False):
                subhalos = il.groupcat.loadSubhalos(self.basePath, snap,fields)
            else:
                brahma_snap = np.where(self.brahma_snapshots == snap)[0][0]
                print(self.unique_redshifts[i], snap, brahma_snap,self.brahma_redshifts[brahma_snap])
                subhalos = il_brahma.groupcat.loadSubhalos_postprocessed(self.basePath,snap,fields)
            
            merger_indices = np.where(self.snap_remnant == snap)[0]
            print(len(merger_indices))
    

            if merger_indices.size > 0:
                subhalo_ids_merger_remnants = self.subhaloidx_remnant[merger_indices]
            else:
                subhalo_ids_merger_remnants = np.array([], dtype=int)

            Ngas = subhalos['SubhaloLenType'][:, 0]
            Ndm = subhalos['SubhaloLenType'][:, 1]
            Nstar = subhalos['SubhaloLenType'][:, 4]
            Nbh = subhalos['SubhaloLenType'][:, 5]
            subhalo_ids = np.arange(len(Ngas))
            print("subhalos in this snap:", len(subhalo_ids))
            print("minN_values:", self.minN_values)
            particle_cut_mask = (Ngas >= self.minN_values[0]) & (Ndm >= self.minN_values[1]) & (Nstar >= self.minN_values[2]) & (Nbh >= 0) #changed to relax the black hole cut in non mergers
            print("subhalos with particle cut:", np.sum(particle_cut_mask))

            valid_subhalo_pos = subhalos['SubhaloPos'][particle_cut_mask]  #ckpc/h
            tree = cKDTree(valid_subhalo_pos, boxsize=self.box_length)
            distances, nearest_neighbors = tree.query(valid_subhalo_pos,k=2)
            nearest_subhalo_ids = nearest_neighbors[:,-1]
            r_subhalos = distances[:,-1]
            stellar_half_mass_radius = subhalos['SubhaloHalfmassRadType'][:,4] #ckpc/h
            r_sep = r_subhalos/(stellar_half_mass_radius[particle_cut_mask]+stellar_half_mass_radius[nearest_subhalo_ids])
            valid_sep_mask = r_sep>2
            r_sep_valid_subhalos = r_sep[valid_sep_mask]

            print(len(subhalo_ids), np.sum(particle_cut_mask), np.sum(valid_sep_mask))
            valid_subhalo_ids = subhalo_ids[particle_cut_mask][valid_sep_mask]
            print(len(valid_subhalo_ids), len(subhalo_ids_merger_remnants))
            subhalo_ids_non_merging = np.setdiff1d(valid_subhalo_ids, subhalo_ids_merger_remnants)

            self.update_merger_details(subhalos, subhalo_ids_merger_remnants, self.unique_redshifts[i], snap)
            self.update_non_merger_details(subhalos, subhalo_ids_non_merging, self.unique_redshifts[i], snap)
        
        self.update_pop_units(self.merging_pop)
        self.update_pop_units(self.non_merging_pop)

    def update_merger_details(self,subhalos,subhalo_ids_merging,redshift,snapnum):

        self.merging_pop["subhalo_ids"] = np.concatenate((self.merging_pop["subhalo_ids"], subhalo_ids_merging))
        self.merging_pop["snap"] = np.append(self.merging_pop["snap"], snapnum*np.ones(len(subhalo_ids_merging)))
        self.merging_pop["z"] = np.append(self.merging_pop["z"], redshift*np.ones(len(subhalo_ids_merging)))
        self.merging_pop["Mstar"] = np.concatenate((self.merging_pop["Mstar"], subhalos['SubhaloMassType'][:,4][subhalo_ids_merging]))
        self.merging_pop["Mgas"] = np.concatenate((self.merging_pop["Mgas"], subhalos['SubhaloMassType'][:,0][subhalo_ids_merging]))
        self.merging_pop["MBH"] = np.concatenate((self.merging_pop["MBH"], subhalos['SubhaloBHMass'][subhalo_ids_merging]))
        self.merging_pop["Mdot"] = np.concatenate((self.merging_pop["Mdot"], subhalos['SubhaloBHMdot'][subhalo_ids_merging]))
        self.merging_pop["SFR"] = np.concatenate((self.merging_pop["SFR"], subhalos['SubhaloSFR'][subhalo_ids_merging]))
        self.merging_pop["Msubhalo"] = np.concatenate((self.merging_pop["Msubhalo"], subhalos['SubhaloMass'][subhalo_ids_merging]))
        self.merging_pop["MgasInRad"] = np.concatenate((self.merging_pop["MgasInRad"], subhalos['SubhaloMassInRadType'][:,0][subhalo_ids_merging]))
        self.merging_pop["MstarInRad"] = np.concatenate((self.merging_pop["MstarInRad"], subhalos['SubhaloMassInRadType'][:,4][subhalo_ids_merging]))
        self.merging_pop["StellarHalfmassRad"] = np.concatenate((self.merging_pop["StellarHalfmassRad"], subhalos['SubhaloHalfmassRadType'][:,4][subhalo_ids_merging]))
        self.merging_pop["SubhaloPhotoMag"] = np.concatenate((self.merging_pop["SubhaloPhotoMag"], subhalos['SubhaloStellarPhotometrics'][subhalo_ids_merging])) 

        merger_indices = np.where(self.snap_remnant == snapnum)[0]
        for merger_idx in merger_indices:
            prog1_info = self.load_progenitor_info(merger_idx, 1)
            prog2_info = self.load_progenitor_info(merger_idx, 2)

            if(prog1_info["Mstar"]>=prog2_info["Mstar"]):
                q_merger = prog2_info["Mstar"]/prog1_info["Mstar"]
            else:
                q_merger = prog1_info["Mstar"]/prog2_info["Mstar"]

            self.merging_pop["q_merger"] = np.append(self.merging_pop["q_merger"], q_merger)
            self.merging_pop["prog_snap"] = np.append(self.merging_pop["prog_snap"], [prog1_info["snap"], prog2_info["snap"]])
            self.merging_pop["prog_subhalo_id"] = np.concatenate((self.merging_pop["prog_subhalo_id"], [prog1_info["subhalo_id"], prog2_info["subhalo_id"]]))
            self.merging_pop["prog_redshift"] = np.append(self.merging_pop["prog_redshift"], [prog1_info["redshift"], prog2_info["redshift"]])
            self.merging_pop["prog_Mstar"] = np.append(self.merging_pop["prog_Mstar"], [prog1_info["Mstar"], prog2_info["Mstar"]])
            self.merging_pop["prog_Mgas"] = np.append(self.merging_pop["prog_Mgas"], [prog1_info["Mgas"], prog2_info["Mgas"]])
            self.merging_pop["prog_MBH"] = np.append(self.merging_pop["prog_MBH"], [prog1_info["MBH"], prog2_info["MBH"]])
            self.merging_pop["prog_Mdot"] = np.append(self.merging_pop["prog_Mdot"], [prog1_info["Mdot"], prog2_info["Mdot"]])
            self.merging_pop["prog_SFR"] = np.append(self.merging_pop["prog_SFR"], [prog1_info["SFR"], prog2_info["SFR"]])
            self.merging_pop["prog_MstarInRad"] = np.append(self.merging_pop["prog_MstarInRad"], [prog1_info["MstarInRad"], prog2_info["MstarInRad"]])
            self.merging_pop["prog_MgasInRad"] = np.append(self.merging_pop["prog_MgasInRad"], [prog1_info["MgasInRad"], prog2_info["MgasInRad"]])
            self.merging_pop["prog_StellarHalfmassRad"] = np.append(self.merging_pop["prog_StellarHalfmassRad"], [prog1_info["StellarHalfmassRad"], prog2_info["StellarHalfmassRad"]])  

    def update_non_merger_details(self,subhalos,subhalo_ids_non_merging,redshift,snapnum):
        
            self.non_merging_pop["subhalo_ids"] = np.concatenate((self.non_merging_pop["subhalo_ids"], subhalo_ids_non_merging))
            self.non_merging_pop["snap"] = np.append(self.non_merging_pop["snap"], snapnum*np.ones(len(subhalo_ids_non_merging)))
            self.non_merging_pop["z"] = np.append(self.non_merging_pop["z"], redshift*np.ones(len(subhalo_ids_non_merging)))
            self.non_merging_pop["Mstar"] = np.concatenate((self.non_merging_pop["Mstar"], subhalos['SubhaloMassType'][:,4][subhalo_ids_non_merging]))
            self.non_merging_pop["Mgas"] = np.concatenate((self.non_merging_pop["Mgas"], subhalos['SubhaloMassType'][:,0][subhalo_ids_non_merging]))
            self.non_merging_pop["MBH"] = np.concatenate((self.non_merging_pop["MBH"], subhalos['SubhaloBHMass'][subhalo_ids_non_merging]))
            self.non_merging_pop["Mdot"] = np.concatenate((self.non_merging_pop["Mdot"], subhalos['SubhaloBHMdot'][subhalo_ids_non_merging]))
            self.non_merging_pop["SFR"] = np.concatenate((self.non_merging_pop["SFR"], subhalos['SubhaloSFR'][subhalo_ids_non_merging]))
            self.non_merging_pop["Msubhalo"] = np.concatenate((self.non_merging_pop["Msubhalo"], subhalos['SubhaloMass'][subhalo_ids_non_merging]))
            self.non_merging_pop["MgasInRad"] = np.concatenate((self.non_merging_pop["MgasInRad"], subhalos['SubhaloMassInRadType'][:,0][subhalo_ids_non_merging]))
            self.non_merging_pop["MstarInRad"] = np.concatenate((self.non_merging_pop["MstarInRad"], subhalos['SubhaloMassInRadType'][:,4][subhalo_ids_non_merging]))
            self.non_merging_pop["StellarHalfmassRad"] = np.concatenate((self.non_merging_pop["StellarHalfmassRad"], subhalos['SubhaloHalfmassRadType'][:,4][subhalo_ids_non_merging])) 
            self.non_merging_pop["SubhaloPhotoMag"] = np.concatenate((self.non_merging_pop["SubhaloPhotoMag"], subhalos['SubhaloStellarPhotometrics'][subhalo_ids_non_merging]))

    def load_progenitor_info(self, merger_idx, prog_number):

        if prog_number == 1:
            snap = self.snap_prog1[merger_idx]
            subhalo_id = self.subhaloidx_prog1[merger_idx]
            redshift = self.z_prog1[merger_idx]
        elif prog_number == 2:
            snap = self.snap_prog2[merger_idx]
            subhalo_id = self.subhaloidx_prog2[merger_idx]
            redshift = self.z_prog2[merger_idx]
        else:
            raise ValueError("prog_number must be 1 or 2")

        fields=['SubhaloMassType','SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloSFR','SubhaloHalfmassRadType',
                'SubhaloMassInRadType']
        
        if(self.brahma_key==False):
            subhalos = il.groupcat.loadSubhalos(
            self.basePath, snap,fields)
        else:
            subhalos = il_brahma.groupcat.loadSubhalos_postprocessed(
            self.basePath,snap,fields)

        prog_dict = {
            "snap": snap,
            "subhalo_id": subhalo_id,
            "redshift": redshift,
            "Mstar": subhalos['SubhaloMassType'][subhalo_id, 4],
            "Mgas": subhalos['SubhaloMassType'][subhalo_id, 0],
            "MBH": subhalos['SubhaloBHMass'][subhalo_id],
            "Mdot": subhalos['SubhaloBHMdot'][subhalo_id],
            "SFR": subhalos['SubhaloSFR'][subhalo_id],
            "MstarInRad":subhalos['SubhaloMassInRadType'][subhalo_id,4],
            "MgasInRad": subhalos['SubhaloMassInRadType'][subhalo_id,0],
            "StellarHalfmassRad": subhalos['SubhaloHalfmassRadType'][subhalo_id,4]
        }

        return prog_dict

    def update_pop_units(self,pop_dict):
        pop_dict["Mstar"] = pop_dict["Mstar"]*1e10/self.h #MSOL
        pop_dict["Mgas"] = pop_dict["Mgas"]*1e10/self.h #MSOL
        pop_dict["MBH"] = pop_dict["MBH"]*1e10/self.h #MSOL
        pop_dict["Mdot"] = pop_dict["Mdot"]*1e10*self.h/(0.978e9/self.h) #MSOL/yr
        pop_dict["MgasInRad"] = pop_dict["MgasInRad"]*1e10/self.h #MSOL
        pop_dict["MstarInRad"] = pop_dict["MstarInRad"]*1e10/self.h #MSOL
        pop_dict["StellarHalfmassRad"] = pop_dict["StellarHalfmassRad"]/self.h #ckpc. Multiply by scale factor (can be obtained from snapnum or redshift to get physical kpc)

        if "prog_Mstar" in pop_dict:
            pop_dict["prog_Mstar"] = pop_dict["prog_Mstar"]*1e10/self.h #MSOL
            pop_dict["prog_Mgas"] = pop_dict["prog_Mgas"]*1e10/self.h #MSOL
            pop_dict["prog_MBH"] = pop_dict["prog_MBH"]*1e10/self.h #MSOL
            pop_dict["prog_Mdot"] = pop_dict["prog_Mdot"]*1e10*self.h/(0.978e9/self.h) #MSOL/yr
            pop_dict["prog_MgasInRad"] = pop_dict["prog_MgasInRad"]*1e10/self.h #MSOL
            pop_dict["prog_MstarInRad"] = pop_dict["prog_MstarInRad"]*1e10/self.h #MSOL
            pop_dict["prog_StellarHalfmassRad"] = pop_dict["prog_StellarHalfmassRad"]/self.h #ckpc. Multiply by scale factor (can be obtained from snapnum or redshift to get physical kpc)

    def save_population_to_file(self,save_path):
        if self.brahma_key:
            outfilename = save_path + f'/{self.simName}_population_sort_gas-{self.minN_values[0]:03d}_dm-{self.minN_values[1]:03d}_star-{self.minN_values[2]:03d}_bh-{self.minN_values[3]:03d}.hdf5'
        else:
            outfilename = save_path + f'/{self.simName}_population_sort_gas-{self.minN_values[0]:03d}_dm-{self.minN_values[1]:03d}_star-{self.minN_values[2]:03d}_bh-{self.minN_values[3]:03d}.hdf5'

        with h5py.File(outfilename, 'w') as f:
            all_merge_grp = f.create_group('merging_population')
            for key, value in self.merging_pop.items():
                all_merge_grp.create_dataset(key, data=value)
            
            non_merge_grp = f.create_group('non_merging_population')
            for key, value in self.non_merging_pop.items():
                non_merge_grp.create_dataset(key, data=value)
        
            if self.brahma_key:
                all_merge_grp.attrs['simulation'] = f'Brahma_{self.simName}'
                all_merge_grp.attrs['hubble_param'] = self.h
                non_merge_grp.attrs['hubble_param'] = self.h
            else:
                all_merge_grp.attrs['simulation'] = f'TNG50-1_{self.simName}'
                all_merge_grp.attrs['prog_mass_ratio']= self.prog_mass_ratio
                all_merge_grp.attrs['hubble_param'] = self.h
                non_merge_grp.attrs['hubble_param'] = self.h
        
        print(f"Saved populations to {outfilename}")



if __name__ == "__main__":

    if len(sys.argv) < 9:
        print("Usage: python population_sort_mergers_and_non_mergers.py <brahma_or_other> <basePath> <merger_file_loc> <pop_file_save_loc> <minN1> <minN2> <minN3> <minN4>")
        sys.exit(1)
        
    if(sys.argv[1] == "brahma"):
        brahma_key = True
    else:
        brahma_key = False
    
    basePath = sys.argv[2]
    merger_file_loc = sys.argv[3]
    print(basePath,merger_file_loc)
    pop_file_save_loc = sys.argv[4]
    minN_values = np.array([int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])])
    
    print(f"Creating merger and non merger populations for sim run"+basePath.split("/")[-2])
    pop_gen = pop_generator(basePath,merger_file_loc,minN_values,brahma_key)
    pop_gen.save_population_to_file(pop_file_save_loc)

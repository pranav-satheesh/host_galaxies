import numpy as np
import h5py
import sys

class ControlSampleGenerator:
    def __init__(self, population, z_tol_default=0.01, Mstar_dex_tol_default=0.1):
        self.pop = population
        self.z_tol_default = z_tol_default
        self.Mstar_dex_tol_default = Mstar_dex_tol_default

        self.control_sample = {
            "idx": np.array([], dtype=int),
            "subhalo_ids": [],
            "snap": [],
            "z": [],
            "Mstar": [],
            "Mgas": [],
            "MBH": [],
            "Mdot": [],
            "SFR": [],
            "z_tol": np.array([], dtype=float),
            "Mstar_dex_tol": np.array([], dtype=float)
        }

    def generate_control_sample(self):
        for i in range(len(self.pop['merging_population']["z"])):
            z_mrg = self.pop['merging_population']["z"][i]
            Mstar_mrg = self.pop['merging_population']["Mstar"][i]

            z_tol = self.z_tol_default
            Mstar_dex_tol = self.Mstar_dex_tol_default

            idxs = np.where((np.abs(self.pop['non_merging_population']["z"] - z_mrg) <= z_tol) &
                            (np.abs(np.log10(self.pop['non_merging_population']["Mstar"]) - np.log10(Mstar_mrg)) <= Mstar_dex_tol))

            while (np.size(idxs) < 10):
                z_tol *= 1.5  # increase the tolerances by 50 percent
                Mstar_dex_tol *= 1.5

                idxs = np.where((np.abs(self.pop['non_merging_population']["z"] - z_mrg) <= z_tol) & 
                                (np.abs(np.log10(self.pop['non_merging_population']["Mstar"]) - np.log10(Mstar_mrg)) <= Mstar_dex_tol))

            self.control_sample["idx"] = np.append(self.control_sample["idx"], i)
            self.control_sample["subhalo_ids"].append(self.pop['non_merging_population']["subhalo_ids"][idxs])
            self.control_sample["snap"].append(self.pop['non_merging_population']["snap"][idxs])
            self.control_sample["z"].append(self.pop['non_merging_population']["z"][idxs])
            self.control_sample["Mstar"].append(self.pop['non_merging_population']["Mstar"][idxs])
            self.control_sample["Mgas"].append(self.pop['non_merging_population']["Mgas"][idxs])
            self.control_sample["MBH"].append(self.pop['non_merging_population']["MBH"][idxs])
            self.control_sample["Mdot"].append(self.pop['non_merging_population']["Mdot"][idxs])
            self.control_sample["SFR"].append(self.pop['non_merging_population']["SFR"][idxs])
            self.control_sample["z_tol"]=np.append(self.control_sample["z_tol"],z_tol)
            self.control_sample["Mstar_dex_tol"]=np.append(self.control_sample["Mstar_dex_tol"],Mstar_dex_tol)
            #self.control_sample["z_tol"].append(z_tol)
            #self.control_sample["Mstar_dex_tol"].append(Mstar_dex_tol)

        return self.control_sample

    def write_control_sample(self, pop_file_loc):
        control_sample_file = pop_file_loc + "control_sample.hdf5"
        with h5py.File(control_sample_file, 'w') as f:
            for key, value in self.control_sample.items():
                # Define a variable-length data type
                vlen_dtype = h5py.special_dtype(vlen=np.dtype('float64'))
                
                # Create a dataset with the variable-length data type
                if isinstance(value, list):
                    dset = f.create_dataset(key, (len(value),), dtype=vlen_dtype)
                    dset[:] = value
                else:
                    f.create_dataset(key, data=value)
        print(f"Control sample saved to {control_sample_file}")

if __name__ == "__main__":
    
    # Get the population file location from the command-line arguments
    pop_file_loc = sys.argv[1]
    pop_file = pop_file_loc + "population_sort_gas-100_dm-100_star-100_bh-001.hdf5"

    # Open the population file
    pop = h5py.File(pop_file, 'r')

    control_generator = ControlSampleGenerator(pop)
    control_sample = control_generator.generate_control_sample()

    print(f"The maximum tolerance in z is {np.max(control_sample['z_tol']):.3f}")
    print(f"The maximum tolerance in $M_{{\star}}$ is {np.max(control_sample['Mstar_dex_tol']):.3f}")
    lengths_subhalo_ids = [len(sublist) for sublist in control_sample['subhalo_ids']]
    print(f"The average number of control samples is {np.mean(lengths_subhalo_ids):.0f}")
    print(f"The minimum number of control samples is {np.min(lengths_subhalo_ids):.0f}")

    control_sample_file_loc = sys.argv[2]
    # Write the control sample to a file
    control_generator.write_control_sample(control_sample_file_loc)


# def generate_control_sample(pop,z_tol_default=0.01,Mstar_dex_tol_default=0.1):
    
#     control_sample = {
#         "idx": np.array([], dtype=int),
#         "subhalo_ids": [],
#         "snap": [],
#         "z": [],
#         "Mstar": [],
#         "Mgas": [],
#         "MBH": [],
#         "Mdot": [],
#         "SFR": []
#     }


#     for i in range(len(pop['merging_population']["z"])):
#         z_mrg = pop['merging_population']["z"][i]
#         Mstar_mrg = pop['merging_population']["Mstar"][i]

#         z_tol = z_tol_default
#         Mstar_dex_tol = Mstar_dex_tol_default


#         idxs=np.where((np.abs(pop['non_merging_population']["z"] - z_mrg) <= z_tol)&(np.abs(np.log10(pop['non_merging_population']["Mstar"]) - np.log10(Mstar_mrg)) <= Mstar_dex_tol))

#         while (np.size(idxs) < 10):
#             z_tol *= 1.5  # increase the tolerances by 50 percent
#             Mstar_dex_tol *= 1.5

#             idxs = np.where((np.abs(pop['non_merging_population']["z"] - z_mrg) <= z_tol) & 
#                             (np.abs(np.log10(pop['non_merging_population']["Mstar"]) - np.log10(Mstar_mrg)) <= Mstar_dex_tol))

#         control_sample["idx"] = np.append(control_sample["idx"], i)
#         control_sample["subhalo_ids"].append(pop['non_merging_population']["subhalo_ids"][idxs])
#         control_sample["snap"].append(pop['non_merging_population']["snap"][idxs])
#         control_sample["z"].append(pop['non_merging_population']["z"][idxs])
#         control_sample["Mstar"].append(pop['non_merging_population']["Mstar"][idxs])
#         control_sample["Mgas"].append(pop['non_merging_population']["Mgas"][idxs])
#         control_sample["MBH"].append(pop['non_merging_population']["MBH"][idxs])
#         control_sample["Mdot"].append(pop['non_merging_population']["Mdot"][idxs])
#         control_sample["SFR"].append(pop['non_merging_population']["SFR"][idxs])

#     return control_sample

# def write_control_sample(control_sample, pop_file_loc):
#     control_sample_file = pop_file_loc + "control_sample.hdf5"

#     with h5py.File(control_sample_file, 'w') as f:
#         for key, value in control_sample.items():

#             vlen_dtype = h5py.special_dtype(vlen=np.dtype('float64'))

#             # Create a dataset with the variable-length data type
#             if isinstance(value, list):
#                 dset = f.create_dataset(key, (len(value),), dtype=vlen_dtype)
#                 dset[:] = value
#             else:
#                 f.create_dataset(key, data=value)
#     print(f"Control sample saved to {control_sample_file}")
#     return None


# if __name__ == "__main__":
#     pop_file_loc = sys.argv[1]
#     pop_file = pop_file_loc + "population_sort_gas-100_dm-100_star-100_bh-001.hdf5"
#     pop = h5py.File(pop_file, 'r')
#     control_sample = generate_control_sample(pop)
#     write_control_sample(control_sample, pop_file_loc)
    

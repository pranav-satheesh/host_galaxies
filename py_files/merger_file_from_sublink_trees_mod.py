import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from collections import Counter
import h5py
import datetime
import sys

# load constants manually if running w/o holodeck / astropy dependencies
try:
    from holodeck.constants import PC, MSOL, YR
except:
    PC = 3.0856775814913674e+18
    MSOL = 1.988409870698051e+33
    YR = 31557600.0

def get_scale_factors(basePath, filename="output_scale_factors.txt"):
    path = basePath.split('/output')[0]
    #print(path)
    f = open(path+"/"+filename,'r')
    snaptimes = np.array([float(line) for line in f.readlines()])
    f.close()
    print(f"snapshot scale factors in {path}:")
    print(snaptimes)
    return snaptimes

def get_simname_from_basepath(basePath):
    tmp = basePath.split('/')
    print(tmp)
    for t in tmp:
        if 'Illustris' in t:
            simname = t
        elif 'TNG' in t:
            simname = t 
        elif ('L' in t) and ('FP' in t):
            simname = t
    return simname

def numMergers(tree, minMassRatio=1e-10, massPartType='stars', index=0, getFullTree=True):
    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 'FirstProgenitorID', 'SubhaloMassType']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    MergerCount   = 0
    invMinMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]
    
    while fpID != -1:
        fpIndex = index + (fpID - rootID)
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)

        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        while npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)

            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass

                if ratio >= minMassRatio and ratio <= invMinMassRatio:
                    MergerCount += 1

            npID = tree['NextProgenitorID'][npIndex]

            # explore full merger history of next progenitors & add their mergers to the total
            if getFullTree and tree['FirstProgenitorID'][npIndex] != -1:
                nmrg_sub = numMergers(tree, minMassRatio=minMassRatio, massPartType=massPartType, index=npIndex)
                MergerCount += nmrg_sub

        fpID = tree['FirstProgenitorID'][fpIndex]

    return MergerCount

# class for extracting merger info from SubLink tree
class MergerInfo:
    
    def __init__(self, *args, **kwargs): 
        """ Initialize class instance for extracting merger info from SubLink tree. """


        if len(args)!=1:
            print("Error: class MererInfo takes exactly 1 argument (tree).")
            return -1

        tree = args[0]

        self.minMassRatio = kwargs.get('minMassRatio', 1.0e-10)
        self.massPartType = kwargs.get('massPartType', 'dm')
        index = kwargs.get('index', 0)
        getFullTree = kwargs.get('getFullTree', True)
        self.minNdm = kwargs.get('minNdm', 0)
        self.minNgas = kwargs.get('minNgas', 0)
        self.minNstar = kwargs.get('minNstar', 0)
        self.minNbh = kwargs.get('minNbh', 0)
        verbose = kwargs.get('verbose', False)
        
        #print(f"in init: {self.minNdm} {self.minNgas} {self.minNstar} {self.minNbh}")
        
        # verify the input sub-tree has the required fields
        reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 
                     'LastProgenitorID', 'RootDescendantID',
                     'FirstProgenitorID', 'SubhaloMassType','SnapNum','DescendantID','FirstSubhaloInFOFGroupID']

        if not set(reqFields).issubset(tree.keys()):
            raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

        self.lastProgenitorID = tree['LastProgenitorID']
        self.rootDescendantID = tree['RootDescendantID']
        idx = np.where(self.lastProgenitorID==self.lastProgenitorID.max())
        #self.nextTreeSubfindID = tree['SubfindID'][
        
        self.count   = 0
        self.progMassRatio = np.array([])
        self.fpSnap = np.array([]).astype('int64')
        self.npSnap = np.array([]).astype('int64')
        self.fpMass = np.array([])
        self.npMass = np.array([])

        self.fpMass_mod = np.array([])
        self.npMass_mod = np.array([])
        self.fpinfallMass = np.array([])
        self.npinfallMass = np.array([])
        self.progMassRatio_mod = np.array([])
        self.infallMassRatio = np.array([])

        self.fpMasshistory = []
        self.npMasshistory = []
        self.fpsnaphistory = []
        self.npsnaphistory = []

        self.descSnap = np.array([]).astype('int64')
        self.fpSubhaloID = np.array([]).astype('int64')
        self.fpSubfindID = np.array([]).astype('int64')
        self.fpSubhaloMassType = np.array([]).reshape(0,6)
        self.fpSubhaloLenType = np.array([]).astype('int64').reshape(0,6)
        self.npSubhaloID = np.array([]).astype('int64')
        self.npSubfindID = np.array([]).astype('int64')
        self.npSubhaloMassType = np.array([]).reshape(0,6)
        self.npSubhaloLenType = np.array([]).astype('int64').reshape(0,6)
        self.descSubhaloID = np.array([]).astype('int64')
        self.descSubfindID = np.array([]).astype('int64')
        self.descSubhaloMassType = np.array([]).reshape(0,6)
        self.descSubhaloLenType = np.array([]).astype('int64').reshape(0,6)
        
        self.countDupDesc = self.countMrgFromDupDesc(tree)
        
        self.getMergerInfoSubtree(tree, minMassRatio=self.minMassRatio, massPartType=self.massPartType, 
                                  index=index, getFullTree=getFullTree, minNdm=self.minNdm, 
                                  minNgas=self.minNgas, minNstar=self.minNstar, verbose=verbose)
        
        
    def countMrgFromDupDesc(self, tree):
        """ Returns total # of mergers in full tree by counting duplicate descendant IDs. """

        dup = Counter(tree['DescendantID'])
        #print(dup)
    
        for item in dup:
            if dup[item] > 1:
                idx = np.where(tree['DescendantID']==item)
                #print(dup[item], item, tree['SubhaloID'][idx], tree['NextProgenitorID'][idx], tree['MainLeafProgenitorID'][idx], tree['FirstProgenitorID'][idx], tree['SnapNum'][idx], tree['SubfindID'][idx])
        #print(np.array([d-1 for d in dup.values()]))
    
        return np.array([d-1 for d in dup.values()]).sum()

    
    def getMergerInfoSubtree(self, tree, minMassRatio=1e-10, massPartType='stars', index=0, 
                             getFullTree=True, minNdm=0, minNgas=0, minNstar=0, minNbh=0, verbose=False):
        """ Return merger snapshots in this sub-tree (optionally above some mass ratio threshold). """

        ## (we're doing this in the init function now)
        ## verify the input sub-tree has the required fields
        #reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 
        #             'LastProgenitorID', 'RootDescendantID',
        #             'FirstProgenitorID', 'SubhaloMassType','SnapNum','DescendantID']

        #if not set(reqFields).issubset(tree.keys()):
        #    raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

        #print(f"in getMergerInfoSubtree: {self.minNdm} {self.minNgas} {self.minNstar} {self.minNbh}")

        invMinMassRatio = 1.0 / self.minMassRatio
    
        # walk back main progenitor branch
        rootID = tree['SubhaloID'][index]
        rootSubfindID = tree['SubfindID'][index]
        fpID   = tree['FirstProgenitorID'][index]
        rootSnap = tree['SnapNum'][index]
        fpSnap = rootSnap #tree['SnapNum'][index]
        #print(f"rootID = {rootID}, fpID={fpID}")
        fdesID = tree['DescendantID'][index]
        
        while fpID != -1:
            fpIndex = index + (fpID - rootID)
            fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)
            fpNpart = tree['SubhaloLenType'][fpIndex,:]

            #for ptype in ['stars','gas','dm']:
            #    tmp, altfpMass = idOfMaxPastMass(tree, fpIndex, ptype)
            #    if altfpMass > fpMass:
            #        print("WARNING!! mass of type {ptype} greater than {massPartType} fpMass!!")
            #fpMass = maxPastMass(tree, fpIndex, massPartType)

            # explore breadth
            npID = tree['NextProgenitorID'][fpIndex]
            npSnap = tree['SnapNum'][fpIndex]
            ndesID = tree['DescendantID'][fpIndex]

            fdesIndex = index + (fdesID - rootID)

            while npID != -1:

                npIndex = index + (npID - rootID)
                npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)
                npNpart = tree['SubhaloLenType'][npIndex,:]

                #for ptype in ['stars','gas','dm']:
                #    tmp, altnpMass = idOfMaxPastMass(tree, npIndex, ptype)
                #    if altnpMass > npMass:
                #        print("WARNING!! mass of type {ptype} greater than {massPartType} npMass!!")
                #npMass = maxPastMass(tree, npIndex, massPartType)
            
                ndesIndex = index + (ndesID - rootID)
            
                # count if both masses are non-zero, and ratio exceeds threshold
                if fpMass > 0.0 and npMass > 0.0:
                    ratio = npMass / fpMass
                    
                    if (ratio >= self.minMassRatio and ratio <= invMinMassRatio and 
                        min(npNpart[0], fpNpart[0]) >= self.minNgas and 
                        min(npNpart[1], fpNpart[1]) >= self.minNdm and
                        min(npNpart[4], fpNpart[4]) >= self.minNstar and
                        min(npNpart[5], fpNpart[5]) >= self.minNbh):
                        
                        self.count += 1

                        # make sure descendant snap is progenitor snap + 1
                        if fpSnap == (npSnap+2):
                            if verbose: print(f"NOTE: SubLink skipped snap {npSnap+1} in finding descendant.")
                        elif fpSnap != (npSnap+1) and fpSnap != (npSnap+2):
                            raise Exception(f'ERROR: snaps not contiguous b/t prog ({npSnap}) & desc ({fpSnap}).')


                        if verbose:
                            print(f"MERGER {self.count}:")
                            print(f" desc {tree['SubfindID'][ndesIndex]} in snap {fpSnap} has progs"
                                  " {tree['SubfindID'][fpIndex]} & {tree['SubfindID'][npIndex]} in snap {npSnap}")
                            print(f" root id={rootID}, rootSnap={rootSnap}, rootSubfindID={rootSubfindID},"
                                  " fpMass={fpMass}, npMass={npMass}, ratio={ratio}")
                            print(f" npIndex={npID}, fpIndex={fpID}, ndesIndex={ndesID}, fdesIndex={fdesID}") 
                            print(" subfind IDs: ", 
                                  [tree['SubfindID'][k] for k in (npIndex,fpIndex,ndesIndex,fdesIndex)])
                            print(" snaps: ", [tree['SnapNum'][k] for k in (npIndex,fpIndex,ndesIndex,fdesIndex)])
                            #print(" subhalo IDs in tree: ", [tree['SubhaloID'][k] for k in (npIndex,fpIndex,ndesIndex,fdesIndex)])
                            print(f"minNdm={minNdm}, minNgas={minNgas}, minNstar={minNstar}, minNbh={minNbh}")
                            print(" np len types:", [tree['SubhaloLenType'][npIndex,i] for i in range(6)])
                            print(" fp len types:", [tree['SubhaloLenType'][fpIndex,i] for i in range(6)])
                            print(" desc len types:", [tree['SubhaloLenType'][ndesIndex,i] for i in range(6)])

                        # progenitor snapnum
                        self.fpSnap = np.append(self.fpSnap, tree['SnapNum'][fpIndex])
                        self.npSnap = np.append(self.npSnap, tree['SnapNum'][npIndex])
                        self.descSnap = np.append(self.descSnap, tree['SnapNum'][ndesIndex])
                        # first progenitor subfind ID and masses
                        self.fpSubhaloID = np.append(self.fpSubhaloID, tree['SubhaloID'][fpIndex])
                        self.fpSubfindID = np.append(self.fpSubfindID, tree['SubfindID'][fpIndex])
                        self.fpSubhaloMassType = np.vstack((self.fpSubhaloMassType, 
                                                            tree['SubhaloMassType'][fpIndex,:]))
                        self.fpSubhaloLenType = np.vstack((self.fpSubhaloLenType, tree['SubhaloLenType'][fpIndex,:]))
                        # next progenitor subfind ID and masses
                        self.npSubhaloID = np.append(self.npSubhaloID, tree['SubhaloID'][npIndex])
                        self.npSubfindID = np.append(self.npSubfindID, tree['SubfindID'][npIndex])
                        self.npSubhaloMassType = np.vstack((self.npSubhaloMassType, 
                                                            tree['SubhaloMassType'][npIndex,:]))
                        self.npSubhaloLenType = np.vstack((self.npSubhaloLenType, tree['SubhaloLenType'][npIndex,:]))
                        
                        # fp_fof = tree['FirstSubhaloInFOFGroupID'][fpIndex]
                        # np_fof = tree['FirstSubhaloInFOFGroupID'][npIndex]   

                        fp_maxpast_mass,np_maxpast_mass,fpinfallMass,npinfallMass,fp_mass_history, np_mass_history, fp_snap_history, np_snap_history = find_infall_and_maxpastmass(tree,fpIndex,npIndex)

                        fpMass_mod = fp_maxpast_mass
                        npMass_mod = np_maxpast_mass
                        print("The sizes of fp_mass_history and np_mass_history are: ", fp_mass_history.size, np_mass_history.size)
                        

                        # mass ratio (defined to be nextProg / firstProg *at max past mass*)
                        self.progMassRatio = np.append(self.progMassRatio, ratio)
                        self.progMassRatio_mod = np.append(self.progMassRatio_mod, npMass_mod / fpMass_mod)
                        self.infallMassRatio = np.append(self.infallMassRatio, npinfallMass / fpinfallMass)

                        self.fpMass = np.append(self.fpMass, fpMass)
                        self.npMass = np.append(self.npMass, npMass)
                        self.fpMass_mod = np.append(self.fpMass_mod, fpMass_mod)
                        self.npMass_mod = np.append(self.npMass_mod, npMass_mod)
                        self.fpinfallMass = np.append(self.fpinfallMass, fpinfallMass)
                        self.npinfallMass = np.append(self.npinfallMass, npinfallMass)

                        self.fpMasshistory.append(fp_mass_history)
                        self.npMasshistory.append(np_mass_history)
                        self.fpsnaphistory.append(fp_snap_history)
                        self.npsnaphistory.append(np_snap_history)

                        # descendant subfind ID and masses
                        self.descSubhaloID = np.append(self.descSubhaloID, tree['SubhaloID'][ndesIndex])
                        self.descSubfindID = np.append(self.descSubfindID, tree['SubfindID'][ndesIndex])
                        self.descSubhaloMassType = np.vstack((self.descSubhaloMassType, 
                                                              tree['SubhaloMassType'][ndesIndex,:]))
                        self.descSubhaloLenType = np.vstack((self.descSubhaloLenType, 
                                                             tree['SubhaloLenType'][ndesIndex,:]))

                    #else:
                    #    print("WARNING!! merger didnt meet mass ratio &/or Npart criteria!")
                        
                npID = tree['NextProgenitorID'][npIndex]
                npSnap = tree['SnapNum'][npIndex]
                ndesID = tree['DescendantID'][npIndex]

                #check for sub-trees and add their mergers
                if getFullTree and tree['FirstProgenitorID'][npIndex] != -1:
                    if verbose: print(f"tracing subtree...")
                    self = self.getMergerInfoSubtree(tree, minMassRatio=self.minMassRatio, 
                                                     massPartType=massPartType, index=npIndex, 
                                                     minNdm=self.minNdm, minNgas=self.minNgas, 
                                                     minNstar=self.minNstar, minNbh=self.minNbh,
                                                     verbose=verbose)
                
                #print(f"status: {tree['SubhaloID'][fpIndex]} {tree['NextProgenitorID'][fpIndex]} {tree['MainLeafProgenitorID'][fpIndex]} {tree['FirstProgenitorID'][fpIndex]} {tree['SnapNum'][fpIndex]} {tree['DescendantID'][fpIndex]} {tree['SubfindID'][fpIndex]}")

            fpID = tree['FirstProgenitorID'][fpIndex]
            fpSnap = tree['SnapNum'][fpIndex]
            fdesID = tree['DescendantID'][fpIndex]

            #print(f"status: {tree['SubhaloID'][fpIndex]} {tree['NextProgenitorID'][fpIndex]} {tree['MainLeafProgenitorID'][fpIndex]} {tree['FirstProgenitorID'][fpIndex]} {tree['SnapNum'][fpIndex]} {tree['DescendantID'][fpIndex]} {tree['SubfindID'][fpIndex]}")

        return self

    def loadExtraSubhaloData(self, basePath, fpSnap, npSnap, descSnap, 
                             fpSubfindID, npSubfindID, descSubfindID):

        try:
            fpsub = il.groupcat.loadSingle(basePath, fpSnap, subhaloID=fpSubfindID)
            npsub = il.groupcat.loadSingle(basePath, npSnap, subhaloID=npSubfindID)
            descsub = il.groupcat.loadSingle(basePath, descSnap, subhaloID=descSubfindID)
        except:
            print(f"Error loading extra subhalo data in {basePath}.")
            print(f"fpSnap={fpSnap}, npSnap={npSnap}, descSnap={descSnap}")
            print(f"fpSubfindID={fpSubfindID}, npSubfindID={npSubfindID}, descSubfindID={descSubfindID}")
            
        return fpsub, npsub, descsub


def find_infall_and_maxpastmass(tree, fp_index, np_index):

    fp_mpb_size = tree['MainLeafProgenitorID'][fp_index] - tree['SubhaloID'][fp_index] + 1
    np_mpb_size = tree['MainLeafProgenitorID'][np_index] - tree['SubhaloID'][np_index] + 1

    first_subs_fp = tree['FirstSubhaloInFOFGroupID'][fp_index: fp_index + fp_mpb_size]
    first_subs_np = tree['FirstSubhaloInFOFGroupID'][np_index: np_index + np_mpb_size]

    fp_masses = tree['SubhaloMassType'][fp_index: fp_index + fp_mpb_size , 4]
    np_masses = tree['SubhaloMassType'][np_index: np_index + np_mpb_size , 4]

    fp_snapshpot_num = tree['SnapNum'][fp_index: fp_index + fp_mpb_size]
    np_snapshpot_num = tree['SnapNum'][np_index: np_index + np_mpb_size]


    first_mismatch = 0
    # fp_infall_mass = -1
    # np_infall_mass = -1

    if len(first_subs_np)==1:
        fp_infall_mass = fp_masses[0]
        np_infall_mass = np_masses[0]
    elif len(first_subs_fp)==1:
        fp_infall_mass = fp_masses[0]
        np_infall_mass = np_masses[0]
    else:
    #finding the infall mass by checking when the first subhalos in the fof group are different

        if first_subs_fp[0] == first_subs_np[0]:
            for i, (a, b) in enumerate(zip(first_subs_fp[0:], first_subs_np[0:])):
                if a != b:
                    first_mismatch = i
                    break

            if first_mismatch == 0:
                fp_infall_mass = fp_masses[0]
                np_infall_mass = np_masses[0]
            else:   
                fp_infall_mass = fp_masses[first_mismatch-1]
                np_infall_mass = np_masses[first_mismatch-1]
            
        elif first_subs_np[0]==first_subs_fp[1]:
                for i, (a, b) in enumerate(zip(first_subs_fp[1:], first_subs_np[0:])):
                    if a != b:
                        first_mismatch = i
                        break
                if first_mismatch == 0:
                    fp_infall_mass = fp_masses[0]
                    np_infall_mass = np_masses[0]
                else:
                    fp_infall_mass = fp_masses[first_mismatch]
                    np_infall_mass = np_masses[first_mismatch-1]
            
        elif first_subs_fp[0]==first_subs_np[1]:
                for i, (a, b) in enumerate(zip(first_subs_fp[0:], first_subs_np[1:])):
                    if a != b:
                        first_mismatch = i
                        break

                if first_mismatch == 0:
                    fp_infall_mass = fp_masses[0]
                    np_infall_mass = np_masses[0]
                else:   
                    fp_infall_mass = fp_masses[first_mismatch-1]
                    np_infall_mass = np_masses[first_mismatch]

        else:
            fp_infall_mass = fp_masses[0]
            np_infall_mass = np_masses[0]

    #max past mass for first progenitor and next progenitor
    fp_past_max_mass = np.max(fp_masses)
    np_past_max_mass = np.max(np_masses)

    return fp_past_max_mass,np_past_max_mass,fp_infall_mass,np_infall_mass,fp_masses, np_masses, fp_snapshpot_num, np_snapshpot_num


def writeMergerFile(savePath,basePath, snapNum, minNdm=0, minNgas=0, minNstar=0, minNbh=0, 
                    subLinkMassType='stars', verbose=False):
    """Walk thru full sublink merger tree and output key merger catalog data to file.
    
    Parameters
    ----------
    savepath: str
                file path to merger file ouput
    basePath : str
               file path to simulation output
    snapNum : int
              simulation snapshot number from which to load merger trees
    minNdm : int, default=0
             minimum number of DM particles for each progenitor subhalo in a merger
    minNgas : int, default=0
             minimum number of gas cells for each progenitor subhalo in a merger
    minNstar : int, default=0
             minimum number of star particles for each progenitor subhalo in a merger
    minNbh : int, default=0
             minimum number of BH particles for each progenitor subhalo in a merger
    subLinkMassType : str, default='stars'
             particle mass type used for mass ratio criterion in merger trees; possible values: 'dm', 'stars'
    verbose : bool, default=False
    
    """
    simName = get_simname_from_basepath(basePath)
    print(f"{simName=}")
    
    if subLinkMassType not in ('dm', 'stars'):
        err = "Error: keyword `subLinkMassType` must be 'stars' or 'dm'."
        raise ValueError(err)
    
    # load header info and snapshot times (scale factors)
    sub_hdr = il.groupcat.loadHeader(basePath, snapNum)
    nsubs = sub_hdr['Nsubgroups_Total']
    snaptimes = get_scale_factors(basePath)
    
    # load subhalo len types & find those that meet criteria
    sub_lentype = il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloLenType'])
    Ngas = sub_lentype[:,0]
    Ndm = sub_lentype[:,1]
    Nstar = sub_lentype[:,4]
    Nbh = sub_lentype[:,5]
    idx = np.where((Ngas >= minNgas) & (Ndm >= minNdm) & (Nstar >= minNstar) & (Nbh >= minNbh))[0]
    nselect = idx.size
    ncheck = 10**np.int64(np.log10(nselect)-1)
    print(f"{nselect} subhalos meet criteria: {minNdm=}, {minNgas=},"
          +f"{minNstar=}, {minNbh=}")
    print(f"Total number of subhalos in snap {snapNum}: {nsubs}. {ncheck=}.")
    sys.stdout.flush()
    
    # create output file and write initial header data
    
    outfilename = f"galaxy-mergers_{simName}_gas-{minNgas:03d}_dm" \
                  f"-{minNdm:03d}_star-{minNstar:03d}_bh-{minNbh:03d}.hdf5"
    #print(f"Output filepath: {basePath}/{outfilename}")
    print(f"Output filepath: {savePath}/{outfilename}")

    with h5py.File(f"{savePath}/{outfilename}",'w') as mf:
        now = datetime.datetime.now()
        mf.attrs['created'] = str(now)
        if 'TNG' in simName:
            hubbleParam = sub_hdr['HubbleParam']
            mf.attrs['Omega0'] = sub_hdr['Omega0']
            mf.attrs['OmegaLambda'] = sub_hdr['OmegaLambda']
            ##print(f"from sub_hdr: UnitMass_in_g = {sub_hdr['UnitMass_in_g']}.")
            ##print(f"from sub_hdr: UnitLength_in_g = {sub_hdr['UnitLength_in_g']}.")
            ##print(f"from sub_hdr: UnitVelocity_in_cm_per_s = {sub_hdr['UnitVelocity_in_cm_per_s']}.")
        else:
            # hard-coding for Illustris
            hubbleParam = 0.704
            mf.attrs['Omega0'] = 0.2726
            mf.attrs['OmegaLambda'] = 0.7274
        mf.attrs['HubbleParam'] = hubbleParam
        mf.attrs['box_volume_mpc'] = (sub_hdr['BoxSize']/1000.0 / hubbleParam)**3 
        mf.attrs['min_parts'] = np.array([minNgas, minNdm, minNstar, minNbh])
        mf.attrs['part_names'] = ['gas', 'dm', 'star', 'bh']
        mf.attrs['part_types'] = np.array([0,1,4,5])
        mf.attrs['snaptimes'] = snaptimes
        print(f"Finished writing initial header data to file.\n")
        sys.stdout.flush()
    
    # list of required subhalo fields
    fields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 'FirstProgenitorID',
              'LastProgenitorID', 'RootDescendantID', 'SubhaloLenType', 'SubhaloMassType',
              'SnapNum', 'DescendantID', 'SubfindID','FirstSubhaloInFOFGroupID']

    # initialize arrays for output data
    allMrgSubhID = np.array([]).astype('int64').reshape(0,3)
    allMrgSubfID = np.array([]).astype('int64').reshape(0,3)
    allMrgSnaps = np.array([]).astype('int64').reshape(0,3)
    allMrgTimes = np.array([]).reshape(0,3)
    allMrgProgMassRatio = np.array([])
    allMrgfpMass = np.array([])
    allMrgnpMass = np.array([])
    allMrgSubhMassType = np.array([]).reshape(0,6,3) ### CHANGED ARRAY SHAPE
    allMrgSubhLenType = np.array([]).astype('int64').reshape(0,6,3) ### CHANGED ARRAY SHAPE
    allMrgSubhBHMass = np.array([]).reshape(0,3)
    allMrgSubhBHMdot = np.array([]).reshape(0,3)
    allMrgSubhCM = np.array([]).reshape(0,3,3) ### NO CHANGE HERE, ONLY BELOW
    allMrgSubhGrNr = np.array([]).astype('int64').reshape(0,3) 
    allMrgSubhHalfmassRadType = np.array([]).reshape(0,6,3) ### CHANGED ARRAY SHAPE
    allMrgSubhMassInHalfRadType = np.array([]).reshape(0,6,3) ### CHANGED ARRAY SHAPE
    allMrgSubhMassInRadType = np.array([]).reshape(0,6,3) ### CHANGED ARRAY SHAPE
    allMrgSubhPos = np.array([]).reshape(0,3,3) ### NO CHANGE HERE, ONLY BELOW
    allMrgSubhSFR = np.array([]).reshape(0,3)
    allMrgSubhVel = np.array([]).reshape(0,3,3) ### NO CHANGE HERE, ONLY BELOW
    allMrgSubhVelDisp = np.array([]).reshape(0,3)

    allMrgfpMass_mod = np.array([]) #first progenitor past max mass
    allMrgnpMass_mod = np.array([]) #next progenitor past max mass
    allMrgfpinfallMass = np.array([]) #first progenitor infall mass
    allMrgnpinfallMass = np.array([]) #next progenitor infall mass
    allMrgProgMassRatio_mod = np.array([]) # mass ratio at max past mass
    allMrgInfallMassRatio = np.array([]) # mass ratio at infall mass
    allMrgfpMasshistory = [] #first progenitor mass history
    allMrgnpMasshistory = [] #next progenitor mass history
    allMrgfpsnaphistory = [] #first progenitor snap history
    allMrgnpsnaphistory = [] #next progenitor snap history

        
    total_mrg_count = 0
    
    # loop over all subhalos meeting length criteria
    for k,isub in enumerate(idx):
        if k%ncheck == 0 or verbose: 
            print(f"processing sub {isub} ({k} of {nselect} meeting criteria)...")
            sys.stdout.flush()
        
        if (sub_lentype[isub,0] < minNgas or sub_lentype[isub,1] < minNdm
            or sub_lentype[isub,4] < minNstar or sub_lentype[isub,5] < minNbh):
            err = f"Error! subhalo {isub} does not meet length criteria."
            raise ValueError(err)

        if verbose: 
            print(f"Ngas={sub_lentype[isub,0]}, Ndm={sub_lentype[isub,1]},"
                  f"Nstar={sub_lentype[isub,4]}, Nbh={sub_lentype[isub,5]}")

        # load the sublink merger tree for this subhalo
        tree = il.sublink.loadTree(basePath, snapNum, isub, fields=fields, onlyMPB=False)
        if tree == None:
            continue
            
        if k%ncheck == 0 or verbose: 
            print(f"Total tree entries for subhalo {isub}: {len(tree['SubhaloID'])}")
            sys.stdout.flush()

        # instantiatite MergerInfo class (walks thru merger tree)
        mrg = MergerInfo(tree, massPartType=subLinkMassType, verbose=verbose, minNgas=minNgas, 
                         minNdm=minNdm, minNstar=minNstar, minNbh=minNbh)

        if mrg.count > 0:
            if k%ncheck == 0 or verbose: 
                print(f" # mergers from dup DescendantIDs: {mrg.countDupDesc}, from MergerInfo class: {mrg.count}")
                sys.stdout.flush()
            total_mrg_count = total_mrg_count + mrg.count
                
            #d_subhmt.resize(total_mrg_count, axis=0)
            #d_subhmt[-mrg.count:] = mrg.Subhalo
            #print(mrg.fpSubhaloID.shape, mrg.npSubhaloID.shape, mrg.descSubhaloID.shape)
            #print(np.array([mrg.fpSubhaloID, mrg.npSubhaloID,mrg.descSubhaloID]).transpose().shape, allMrgSubhID.shape)
            allMrgSubhID = np.vstack( (allMrgSubhID, np.array([mrg.fpSubhaloID, mrg.npSubhaloID, 
                                                               mrg.descSubhaloID]).T) )
            allMrgSubfID = np.vstack( (allMrgSubfID, np.array([mrg.fpSubfindID, mrg.npSubfindID, 
                                                               mrg.descSubfindID]).T) )
            #allMergersNextProgSubhID = np.append(allMergersNextProgSubhID, mrg.npSubhaloID)
            #allMergersDescSubhID = np.append(allMergersDescSubhID, mrg.descSubhaloID)
            #allMergersFirstProgSubfID = np.append(allMergersFirstProgSubfID, mrg.fpSubfindID)
            #allMergersNextProgSubfID = np.append(allMergersNextProgSubfID, mrg.npSubfindID)
            #allMergersDescSubfID = np.append(allMergersDescSubfID, mrg.descSubfindID)
            allMrgSnaps = np.vstack( (allMrgSnaps, np.array([mrg.fpSnap, mrg.npSnap, mrg.descSnap]).T) )
            allMrgTimes = np.vstack( (allMrgTimes, np.array([snaptimes[mrg.fpSnap], 
                                                             snaptimes[mrg.npSnap], 
                                                             snaptimes[mrg.descSnap]]).T) )
            allMrgProgMassRatio = np.append( allMrgProgMassRatio, mrg.progMassRatio )
            allMrgfpMass = np.append( allMrgfpMass, mrg.fpMass )
            allMrgnpMass = np.append( allMrgnpMass, mrg.npMass )

            #new modified informations added to the merger file
            allMrgProgMassRatio_mod = np.append( allMrgProgMassRatio_mod, mrg.progMassRatio_mod )
            allMrgfpMass_mod = np.append( allMrgfpMass_mod, mrg.fpMass_mod )
            allMrgnpMass_mod = np.append( allMrgnpMass_mod, mrg.npMass_mod )
            allMrgfpinfallMass = np.append( allMrgfpinfallMass, mrg.fpinfallMass )
            allMrgnpinfallMass = np.append( allMrgnpinfallMass, mrg.npinfallMass )
            allMrgInfallMassRatio = np.append( allMrgInfallMassRatio, mrg.infallMassRatio )
            allMrgfpMasshistory.extend(mrg.fpMasshistory )
            allMrgnpMasshistory.extend(mrg.npMasshistory )
            allMrgfpsnaphistory.extend(mrg.fpsnaphistory )
            allMrgnpsnaphistory.extend(mrg.npsnaphistory )

            # loop thru mergers in this subhalo, load extra subhalo data, append to arrays
            for j in range(mrg.count):
                
                # length-6 vector variables
                allMrgSubhMassType = np.vstack( (allMrgSubhMassType, 
                                                 np.vstack([mrg.fpSubhaloMassType[j,:], 
                                                            mrg.npSubhaloMassType[j,:],
                                                            mrg.descSubhaloMassType[j,:]]
                                                          ).T.reshape(1,6,3)) ) ### CHANGED ARRAY SHAPE
                allMrgSubhLenType = np.vstack( (allMrgSubhLenType, 
                                                np.vstack([mrg.fpSubhaloLenType[j,:], 
                                                           mrg.npSubhaloLenType[j,:], 
                                                           mrg.descSubhaloLenType[j,:]]
                                                         ).T.reshape(1,6,3)) ) ### CHANGED ARRAY SHAPE

                # load extra data from subhalo catalog
                fps, nps, ds = mrg.loadExtraSubhaloData(basePath, mrg.fpSnap[j], 
                                                        mrg.npSnap[j], mrg.descSnap[j], 
                                                        mrg.fpSubfindID[j], mrg.npSubfindID[j],
                                                        mrg.descSubfindID[j]) 
                
                # scalar variables
                allMrgSubhBHMass = np.vstack( (allMrgSubhBHMass, 
                                               np.array([fps['SubhaloBHMass'], 
                                                         nps['SubhaloBHMass'], 
                                                         ds['SubhaloBHMass']]).T) ) 
                allMrgSubhBHMdot = np.vstack( (allMrgSubhBHMdot, 
                                               np.array([fps['SubhaloBHMdot'], 
                                                         nps['SubhaloBHMdot'], 
                                                         ds['SubhaloBHMdot']]).T) ) 
                allMrgSubhGrNr = np.vstack( (allMrgSubhGrNr, 
                                             np.array([fps['SubhaloGrNr'], 
                                                       nps['SubhaloGrNr'], 
                                                       ds['SubhaloGrNr']]).T) )
                allMrgSubhSFR = np.vstack( (allMrgSubhSFR, 
                                            np.array([fps['SubhaloSFR'], 
                                                      nps['SubhaloSFR'],
                                                      ds['SubhaloSFR']]).T) ) 
                allMrgSubhVelDisp = np.vstack( (allMrgSubhVelDisp, 
                                                np.array([fps['SubhaloVelDisp'], 
                                                          nps['SubhaloVelDisp'], 
                                                          ds['SubhaloVelDisp']]).T) ) 
                
                # length-3 vector variables
                allMrgSubhCM = np.vstack( (allMrgSubhCM, 
                                           np.vstack([fps['SubhaloCM'],
                                                      nps['SubhaloCM'],
                                                      ds['SubhaloCM']]
                                                    ).T.reshape(1,3,3)) ) ### CHANGED ARRAY SHAPE
                allMrgSubhPos = np.vstack( (allMrgSubhPos, 
                                            np.vstack([fps['SubhaloPos'],
                                                       nps['SubhaloPos'],
                                                       ds['SubhaloPos']]
                                                     ).T.reshape(1,3,3)) ) ### CHANGED ARRAY SHAPE
                allMrgSubhVel = np.vstack( (allMrgSubhVel, 
                                            np.vstack([fps['SubhaloVel'],
                                                       nps['SubhaloVel'],
                                                       ds['SubhaloVel']]
                                                     ).T.reshape(1,3,3)) ) ### CHANGED ARRAY SHAPE

                # remaining length-6 vector variables
                allMrgSubhHalfmassRadType = np.vstack( (allMrgSubhHalfmassRadType, 
                                                        np.vstack([fps['SubhaloHalfmassRadType'], 
                                                                   nps['SubhaloHalfmassRadType'], 
                                                                   ds['SubhaloHalfmassRadType']]
                                                                 ).T.reshape(1,6,3)) ) ### CHANGED ARRAY SHAPE
                allMrgSubhMassInHalfRadType = np.vstack( (allMrgSubhMassInHalfRadType, 
                                                          np.vstack([fps['SubhaloMassInHalfRadType'], 
                                                                     nps['SubhaloMassInHalfRadType'], 
                                                                     ds['SubhaloMassInHalfRadType']]
                                                                   ).T.reshape(1,6,3)) ) ### CHANGED ARRAY SHAPE
                allMrgSubhMassInRadType = np.vstack( (allMrgSubhMassInRadType, 
                                                      np.vstack([fps['SubhaloMassInRadType'], 
                                                                 nps['SubhaloMassInRadType'], 
                                                                 ds['SubhaloMassInRadType']]
                                                               ).T.reshape(1,6,3)) ) ### CHANGED ARRAY SHAPE

    print(f"allMrgSubhID shape: {allMrgSubhID.shape}")
    sys.stdout.flush()

    with h5py.File(f"{savePath}/{outfilename}",'a') as mf:

        mf.attrs['num_mergers'] = total_mrg_count
        mf.attrs['merger_components_in_data_arrays'] = '(FirstProg, NextProg, Descendant)'

        mf.create_dataset('SubhaloMassType', data=allMrgSubhMassType * 1.0e10 / hubbleParam * MSOL) #: [cm]
        mf['SubhaloMassType'].attrs['dataShape'] = '(Nmrg, Nparttypes, 3)'
        mf['SubhaloMassType'].attrs['units'] = '[g]'

        mf.create_dataset('SubhaloLenType', data=allMrgSubhLenType)
        mf['SubhaloLenType'].attrs['dataShape'] = '(Nmrg, Nparttypes, 3)'
        mf['SubhaloLenType'].attrs['units'] = 'none'

        mf.create_dataset('shids_tree', data=allMrgSubhID)
        mf['shids_tree'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['shids_tree'].attrs['units'] = 'none'

        mf.create_dataset('shids_subf', data=allMrgSubfID)
        mf['shids_subf'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['shids_subf'].attrs['units'] = 'none'

        mf.create_dataset('snaps', data=allMrgSnaps)
        mf['snaps'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['snaps'].attrs['units'] = 'none'

        mf.create_dataset('time', data=allMrgTimes)
        mf['time'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['time'].attrs['units'] = 'none'
      
        mf.create_dataset('ProgMassRatio', data=allMrgProgMassRatio)
        mf['ProgMassRatio'].attrs['dataShape'] = 'Nmrg'
        mf['ProgMassRatio'].attrs['units'] = 'none'

        mf.create_dataset('fpMass', data=allMrgfpMass * 1.0e10 / hubbleParam) #: [Msun]
        mf['fpMass'].attrs['dataShape'] = 'Nmrg'
        mf['fpMass'].attrs['units'] = '[Msun]'

        mf.create_dataset('npMass', data=allMrgnpMass * 1.0e10 / hubbleParam) #: [Msun]
        mf['npMass'].attrs['dataShape'] = 'Nmrg'
        mf['npMass'].attrs['units'] = '[Msun]'

        mf.create_dataset('ProgMassRatio_mod', data=allMrgProgMassRatio_mod)
        mf['ProgMassRatio_mod'].attrs['dataShape'] = 'Nmrg'
        mf['ProgMassRatio_mod'].attrs['units'] = 'none'

        mf.create_dataset('fpMass_mod', data=allMrgfpMass_mod * 1.0e10 / hubbleParam) #: [Msun]
        mf['fpMass_mod'].attrs['dataShape'] = 'Nmrg'
        mf['fpMass_mod'].attrs['units'] = '[Msun]'

        mf.create_dataset('npMass_mod', data=allMrgnpMass_mod * 1.0e10 / hubbleParam) #: [Msun]
        mf['npMass_mod'].attrs['dataShape'] = 'Nmrg'
        mf['npMass_mod'].attrs['units'] = '[Msun]'

        mf.create_dataset('fpinfallMass', data=allMrgfpinfallMass * 1.0e10 / hubbleParam) #: [Msun]
        mf['fpinfallMass'].attrs['dataShape'] = 'Nmrg'
        mf['fpinfallMass'].attrs['units'] = '[Msun]'

        mf.create_dataset('npinfallMass', data=allMrgnpinfallMass * 1.0e10 / hubbleParam) #: [Msun]
        mf['npinfallMass'].attrs['dataShape'] = 'Nmrg'
        mf['npinfallMass'].attrs['units'] = '[Msun]'

        mf.create_dataset('InfallMassRatio', data=allMrgInfallMassRatio)
        mf['InfallMassRatio'].attrs['dataShape'] = 'Nmrg'
        mf['InfallMassRatio'].attrs['units'] = 'none'

        dt = h5py.special_dtype(vlen=np.float64)
        mf.create_dataset('fpMasshistory', data=np.array(allMrgfpMasshistory, dtype=object), dtype=dt) #: [Msun]
        mf['fpMasshistory'].attrs['dataShape'] = 'var'
        mf['fpMasshistory'].attrs['units'] = 'code'

        mf.create_dataset('npMasshistory', data=np.array(allMrgnpMasshistory, dtype=object), dtype=dt)#code units
        mf['npMasshistory'].attrs['dataShape'] = 'var'
        mf['npMasshistory'].attrs['units'] = 'code'

        mf.create_dataset('fpsnaphistory', data=np.array(allMrgfpsnaphistory, dtype=object), dtype=dt)
        mf['fpsnaphistory'].attrs['dataShape'] = 'var'
        mf['fpsnaphistory'].attrs['units'] = 'none'

        mf.create_dataset('npsnaphistory', data=np.array(allMrgnpsnaphistory, dtype=object), dtype=dt)
        mf['npsnaphistory'].attrs['dataShape'] = 'var'
        mf['npsnaphistory'].attrs['units'] = 'none'

        mf.create_dataset('SubhaloBHMass', data=allMrgSubhBHMass * 1.0e10 / hubbleParam * MSOL) #: [g]
        mf['SubhaloBHMass'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['SubhaloBHMass'].attrs['units'] = '[g]'

        mf.create_dataset('SubhaloBHMdot', data=allMrgSubhBHMdot * 10.2247 * MSOL / YR) #: [g/s]
        mf['SubhaloBHMdot'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['SubhaloBHMdot'].attrs['units'] = '[g/s]'
        
        mf.create_dataset('SubhaloCM', data=allMrgSubhCM * 1.0e3*PC / hubbleParam) #: [cm]
        mf['SubhaloCM'].attrs['dataShape'] = '(Nmrg, Ndims, 3)'
        mf['SubhaloCM'].attrs['units'] = '[cm]'

        mf.create_dataset('SubhaloGrNr', data=allMrgSubhGrNr)
        mf['SubhaloGrNr'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['SubhaloGrNr'].attrs['units'] = 'none'
        
        mf.create_dataset('SubhaloHalfmassRadType', 
                          data=allMrgSubhHalfmassRadType * 1.0e3*PC / hubbleParam) #: [cm]
        mf['SubhaloHalfmassRadType'].attrs['dataShape'] = '(Nmrg, Nparttypes, 3)'
        mf['SubhaloHalfmassRadType'].attrs['units'] = '[cm]'
        
        mf.create_dataset('SubhaloMassInHalfRadType', 
                          data=allMrgSubhMassInHalfRadType * 1.0e10 / hubbleParam * MSOL) #: [g]
        mf['SubhaloMassInHalfRadType'].attrs['dataShape'] = '(Nmrg, Nparttypes, 3)'
        mf['SubhaloMassInHalfRadType'].attrs['units'] = '[g]'
        
        mf.create_dataset('SubhaloMassInRadType', 
                          data=allMrgSubhMassInRadType * 1.0e10 / hubbleParam * MSOL) #: [g]
        mf['SubhaloMassInRadType'].attrs['dataShape'] = '(Nmrg, Nparttypes, 3)'
        mf['SubhaloMassInRadType'].attrs['units'] = '[g]'

        mf.create_dataset('SubhaloPos', data=allMrgSubhPos * 1.0e3*PC / hubbleParam) #: [cm]
        mf['SubhaloPos'].attrs['dataShape'] = '(Nmrg, Ndims, 3)'
        mf['SubhaloPos'].attrs['units'] = '[cm]'

        mf.create_dataset('SubhaloSFR', data=allMrgSubhSFR) # code units
        mf['SubhaloSFR'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['SubhaloSFR'].attrs['units'] = '[code units]'
        
        mf.create_dataset('SubhaloVel', data=allMrgSubhVel * 1.0e5) #: [cm/s]
        mf['SubhaloVel'].attrs['dataShape'] = '(Nmrg, Ndims, 3)'
        mf['SubhaloVel'].attrs['units'] = '[cm/s]'

        mf.create_dataset('SubhaloVelDisp', data=allMrgSubhVelDisp * 1.0e5) #: [cm/s]
        mf['SubhaloVelDisp'].attrs['dataShape'] = '(Nmrg, 3)'
        mf['SubhaloVelDisp'].attrs['units'] = '[cm/s]'
        
    print(f"Finished processing merger trees for {nsubs} subhalos in snap {snapNum}.")
    print(f"Found {total_mrg_count} mergers.")                                                                   

if __name__ == "__main__":

    if len(sys.argv)>6:
        basePath = sys.argv[1]
        snapNum = int(sys.argv[2])
        minNdm = int(sys.argv[3])
        minNgas = int(sys.argv[4])
        minNstar = int(sys.argv[5])
        minNbh = int(sys.argv[6])
        savePath = sys.argv[7]
        if len(sys.argv)>8:
            print("Too many command line args ({sys.argv}).")
            sys.exit()

    else:
        print("expecting 6 command line args: basePath, snapNum, minNdm, minNgas, minNstar, minNbh.")
        sys.exit()

    writeMergerFile(savePath,basePath, snapNum, minNdm=minNdm, minNgas=minNgas, 
                    minNstar=minNstar, minNbh=minNbh, verbose=False)
        
    
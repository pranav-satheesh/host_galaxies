import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from collections import Counter
import h5py
import datetime
import sys


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
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    MergerCount   = 0
    invMassRatio = 1.0 / minMassRatio

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

                if ratio >= minMassRatio and ratio <= invMassRatio:
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
                     'FirstProgenitorID', 'SubhaloMassType','SnapNum','DescendantID']

        if not set(reqFields).issubset(tree.keys()):
            raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

        self.lastProgenitorID = tree['LastProgenitorID']
        self.rootDescendantID = tree['RootDescendantID']
        idx = np.where(self.lastProgenitorID==self.lastProgenitorID.max())
        #self.nextTreeSubfindID = tree['SubfindID'][
        
        self.count   = 0
        self.fpSnap = np.array([]).astype('int64')
        self.npSnap = np.array([]).astype('int64')
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

        invMassRatio = 1.0 / self.minMassRatio
    
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
                    
                    if (ratio >= self.minMassRatio and ratio <= invMassRatio and 
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
                            print(f" desc {tree['SubfindID'][ndesIndex]} in snap {fpSnap} has progs {tree['SubfindID'][fpIndex]} & {tree['SubfindID'][npIndex]} in snap {npSnap}")
                            print(f" root id={rootID}, rootSnap={rootSnap}, rootSubfindID={rootSubfindID}, fpMass={fpMass}, npMass={npMass}, ratio={ratio}")
                            print(f" npIndex={npID}, fpIndex={fpID}, ndesIndex={ndesID}, fdesIndex={fdesID}") 
                            print(" subfind IDs: ", [tree['SubfindID'][k] for k in (npIndex,fpIndex,ndesIndex,fdesIndex)])
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
                        self.fpSubhaloMassType = np.vstack((self.fpSubhaloMassType, tree['SubhaloMassType'][fpIndex,:]))
                        self.fpSubhaloLenType = np.vstack((self.fpSubhaloLenType, tree['SubhaloLenType'][fpIndex,:]))
                        # next progenitor subfind ID and masses
                        self.npSubhaloID = np.append(self.npSubhaloID, tree['SubhaloID'][npIndex])
                        self.npSubfindID = np.append(self.npSubfindID, tree['SubfindID'][npIndex])
                        self.npSubhaloMassType = np.vstack((self.npSubhaloMassType, tree['SubhaloMassType'][npIndex,:]))
                        self.npSubhaloLenType = np.vstack((self.npSubhaloLenType, tree['SubhaloLenType'][npIndex,:]))
                        # descendant subfind ID and masses
                        self.descSubhaloID = np.append(self.descSubhaloID, tree['SubhaloID'][ndesIndex])
                        self.descSubfindID = np.append(self.descSubfindID, tree['SubfindID'][ndesIndex])
                        self.descSubhaloMassType = np.vstack((self.descSubhaloMassType, tree['SubhaloMassType'][ndesIndex,:]))
                        self.descSubhaloLenType = np.vstack((self.descSubhaloLenType, tree['SubhaloLenType'][ndesIndex,:]))

                    #else:
                    #    print("WARNING!! merger didnt meet mass ratio &/or Npart criteria!")
                        
                npID = tree['NextProgenitorID'][npIndex]
                npSnap = tree['SnapNum'][npIndex]
                ndesID = tree['DescendantID'][npIndex]

                #check for sub-trees and add their mergers
                #if getFullTree and tree['FirstProgenitorID'][npIndex] != -1:
                #    if verbose: print(f"tracing subtree...")
                #    self = self.getMergerInfoSubtree(tree, minMassRatio=self.minMassRatio, 
                #                                     massPartType=massPartType, index=npIndex, 
                #                                     minNdm=self.minNdm, minNgas=self.minNgas, 
                #                                     minNstar=self.minNstar, minNbh=self.minNbh,
                #                                     verbose=verbose)
                
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

    
def writeMergerFile(basePath, snapNum, minNdm=0, minNgas=0, minNstar=0, minNbh=0,
                    verbose=False):
        
    simName = get_simname_from_basepath(basePath)
    print(simName)
    
    sub_hdr = il.groupcat.loadHeader(basePath, snapNum)
    box_volume_mpch = (sub_hdr['BoxSize']/1000.0)**3 # box volume in Mpc/h
    nsubs = sub_hdr['Nsubgroups_Total']

    snaptimes = get_scale_factors(basePath)
    sub_lentype = il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloLenType'])
    
    Ngas = sub_lentype[:,0]
    Ndm = sub_lentype[:,1]
    Nstar = sub_lentype[:,4]
    Nbh = sub_lentype[:,5]
    idx = np.where((Ngas >= minNgas)&(Ndm >= minNdm) &(Nstar >= minNstar)&(Nbh >= minNbh))[0]
    nselect = idx.size
    ncheck = 10**np.int64(np.log10(nselect)-1)
    print(f"{nselect} subhalos meet criteria: minNdm={minNdm}, minNgas={minNgas},"+
          f"minNstar={minNstar}, minNbh={minNbh}")
    print(f"Total number of subhalos in snap {snapNum}: {nsubs}. ncheck={ncheck}.")

    outfilename = f"galaxy-mergers_{simName}_gas-{minNgas:03d}_dm-{minNdm:03d}_star-{minNstar:03d}_bh-{minNbh:03d}.hdf5"
    print(f"Output filename: {basePath}/{outfilename}")
    
    mf = h5py.File(f"{basePath}/{outfilename}",'w')
    now = datetime.datetime.now()
    mf.attrs['created'] = str(now)
    mf.attrs['box_volume_mpch'] = box_volume_mpch
    if 'TNG' in simName:
        mf.attrs['HubbleParam'] = sub_hdr['HubbleParam']
        mf.attrs['Omega0'] = sub_hdr['Omega0']
        mf.attrs['OmegaLambda'] = sub_hdr['OmegaLambda']
    else:
        mf.attrs['HubbleParam'] = 0.704
        mf.attrs['Omega0'] = 0.2726
        mf.attrs['OmegaLambda'] = 0.7274
    mf.attrs['min_parts'] = np.array([minNgas, minNdm, minNstar, minNbh])
    mf.attrs['part_names'] = ['gas', 'dm', 'star', 'bh']
    mf.attrs['part_types'] = np.array([0,1,4,5])
    mf.attrs['snaptimes'] = snaptimes
    mf.close()
    print(f"Finished writing initial header data to file.\n")
    
    fields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 'FirstProgenitorID',
              'LastProgenitorID', 'RootDescendantID', 'SubhaloLenType', 'SubhaloMassType',
              'SnapNum', 'DescendantID', 'SubfindID']

    allMrgSubhID = np.array([]).astype('int64').reshape(0,3)
    allMrgSubfID = np.array([]).astype('int64').reshape(0,3)
    allMrgSnaps = np.array([]).astype('int64').reshape(0,3)
    allMrgTimes = np.array([]).reshape(0,3)
    allMrgSubhMassType = np.array([]).reshape(0,3,6)
    allMrgSubhLenType = np.array([]).astype('int64').reshape(0,3,6)
    allMrgSubhBHMass = np.array([]).reshape(0,3)
    allMrgSubhBHMdot = np.array([]).reshape(0,3)
    allMrgSubhCM = np.array([]).reshape(0,3,3)
    allMrgSubhGrNr = np.array([]).astype('int64').reshape(0,3)
    allMrgSubhHalfmassRadType = np.array([]).reshape(0,3,6)
    allMrgSubhMassInHalfRadType = np.array([]).reshape(0,3,6)
    allMrgSubhMassInRadType = np.array([]).reshape(0,3,6)
    allMrgSubhPos = np.array([]).reshape(0,3,3)
    allMrgSubhSFR = np.array([]).reshape(0,3)
    allMrgSubhVel = np.array([]).reshape(0,3,3)
    allMrgSubhVelDisp = np.array([]).reshape(0,3)

    massType = 'dm' 
        
    total_mrg_count = 0
        
    for k,isub in enumerate(idx):
        if k%ncheck == 0 or verbose: print(f"processing sub {isub} ({k} of {nselect} meeting criteria)...")
        
        if (sub_lentype[isub,0] < minNgas or sub_lentype[isub,1] < minNdm or
             sub_lentype[isub,4] < minNstar or sub_lentype[isub,5] < minNbh):
            print(f"Error! subhalo {isub} does not meet length criteria.")
            return -1

        if verbose: 
            print(f"Ngas={sub_lentype[isub,0]}, Ndm={sub_lentype[isub,1]},")
            print(f"Nstar={sub_lentype[isub,4]}, Nbh={sub_lentype[isub,5]}")

        tree = il.sublink.loadTree(basePath, snapNum, isub, fields=fields, onlyMPB=False)
        if tree == None:
            continue
        
        if k%ncheck == 0 or verbose: 
            print(f"Total tree entries for subhalo {isub}: {len(tree['SubhaloID'])}")

        mrg = MergerInfo(tree, massPartType=massType, verbose=verbose, minNgas=minNgas, 
                         minNdm=minNdm, minNstar=minNstar, minNbh=minNbh)

        if mrg.count > 0:
            if k%ncheck == 0 or verbose: 
                print(f" # mergers from dup DescendantIDs: {mrg.countDupDesc},"+
                      f" from MergerInfo class: {mrg.count}")
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
            

            #### and load extra subhalo data here
            for j in range(mrg.count):
                
                ### add subhalomasstype and subhalolentype here
                allMrgSubhMassType = np.vstack( (allMrgSubhMassType, 
                                                 np.array([mrg.fpSubhaloMassType[j,:], 
                                                           mrg.npSubhaloMassType[j,:],
                                                           mrg.descSubhaloMassType[j,:]]
                                                         ).reshape(1,3,6)) )
                allMrgSubhLenType = np.vstack( (allMrgSubhLenType, 
                                                np.array([mrg.fpSubhaloLenType[j,:], 
                                                          mrg.npSubhaloLenType[j,:], 
                                                          mrg.descSubhaloLenType[j,:]]
                                                        ).reshape(1,3,6)) )

                
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
                                           np.array([fps['SubhaloCM'],
                                                     nps['SubhaloCM'],
                                                     ds['SubhaloCM']]
                                                   ).reshape(1,3,3)) )
                allMrgSubhPos = np.vstack( (allMrgSubhPos, 
                                            np.array([fps['SubhaloPos'],
                                                      nps['SubhaloPos'],
                                                      ds['SubhaloPos']]
                                                    ).reshape(1,3,3)) )
                allMrgSubhVel = np.vstack( (allMrgSubhVel, 
                                            np.array([fps['SubhaloVel'],
                                                      nps['SubhaloVel'],
                                                      ds['SubhaloVel']]
                                                    ).reshape(1,3,3)) )

                # length-6 vector variables
                allMrgSubhHalfmassRadType = np.vstack( (allMrgSubhHalfmassRadType, 
                                                        np.array([fps['SubhaloHalfmassRadType'], 
                                                                  nps['SubhaloHalfmassRadType'], 
                                                                  ds['SubhaloHalfmassRadType']]
                                                                ).reshape(1,3,6)) )
                allMrgSubhMassInHalfRadType = np.vstack( (allMrgSubhMassInHalfRadType, 
                                                          np.array([fps['SubhaloMassInHalfRadType'], 
                                                                    nps['SubhaloMassInHalfRadType'], 
                                                                    ds['SubhaloMassInHalfRadType']]
                                                                  ).reshape(1,3,6)) )
                allMrgSubhMassInRadType = np.vstack( (allMrgSubhMassInRadType, 
                                                      np.array([fps['SubhaloMassInRadType'], 
                                                                nps['SubhaloMassInRadType'], 
                                                                ds['SubhaloMassInRadType']]
                                                              ).reshape(1,3,6)) )

    print(f"allMrgSubhID shape: {allMrgSubhID.shape}")

    
    mf = h5py.File(f"{basePath}/{outfilename}",'a')
    mf.attrs['num_mergers'] = total_mrg_count
    mf.create_dataset('SubhaloMassType', data=allMrgSubhMassType)
    mf.create_dataset('SubhaloLenType', data=allMrgSubhLenType)
    mf.create_dataset('shids_tree', data=allMrgSubhID)
    mf.create_dataset('shids_subf', data=allMrgSubfID)
    mf.create_dataset('snaps', data=allMrgSnaps)
    mf.create_dataset('times', data=allMrgTimes)
    mf.create_dataset('SubhaloBHMass', data=allMrgSubhBHMass)
    mf.create_dataset('SubhaloBHMdot', data=allMrgSubhBHMdot)
    mf.create_dataset('SubhaloCM', data=allMrgSubhCM)
    mf.create_dataset('SubhaloGrNr', data=allMrgSubhGrNr)
    mf.create_dataset('SubhaloHalfmassRadType', data=allMrgSubhHalfmassRadType)
    mf.create_dataset('SubhaloMassInHalfRadType', data=allMrgSubhMassInHalfRadType)
    mf.create_dataset('SubhaloMassInRadType', data=allMrgSubhMassInRadType)
    mf.create_dataset('SubhaloPos', data=allMrgSubhPos)
    mf.create_dataset('SubhaloSFR', data=allMrgSubhSFR)
    mf.create_dataset('SubhaloVel', data=allMrgSubhVel)
    mf.create_dataset('SubhaloVelDisp', data=allMrgSubhVelDisp)

    mf.close()
    print(f"Finished processing merger trees for {nsubs} subhalos in snap {snapNum}.")
    print(f"Found {total_mrg_count} mergers.")                                                                   
    

if __name__ == "__main__":

    if len(sys.argv)>6:
        basePath=sys.argv[1]
        snapNum=int(sys.argv[2])
        minNdm=int(sys.argv[3])
        minNgas=int(sys.argv[4])
        minNstar=int(sys.argv[5])
        minNbh=int(sys.argv[6])
        if len(sys.argv)>7:
            print("Too many command line args ({sys.argv}).")
            sys.exit()

    else:
        print("expecting 6 command line args: basePath, snapNum, minNdm, minNgas, minNstar, minNbh.")
        sys.exit()
    
    writeMergerFile(basePath, snapNum, minNdm=minNdm, minNgas=minNgas, 
                    minNstar=minNstar, minNbh=minNbh, verbose=False)
        
    
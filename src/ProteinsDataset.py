
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import math 
import numpy as np
import pandas as pd
# from utils import *
from torch._six import string_classes
import collections
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

def default_collate(batch):
    r"""Puts each data field into a tensor with first dimension batch size. 
    Modified to get 1st dim as batch dim"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 1, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def getUnique(tensor):
    inverseMapping = torch.unique(tensor, dim=1, return_inverse=True)[1]
    dic=defaultdict(lambda:0)
    BooleanKept = torch.tensor([False] * tensor.shape[1])
    for i in range(tensor.shape[1]):
        da = int(inverseMapping[i])
        if dic[da]==0:
            BooleanKept[i]=True
        dic[da] +=1
    return tensor[:, BooleanKept, :], BooleanKept


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, fastaPath,  mapstring = "-ACDEFGHIKLMNPQRSTVWY", transform=None, device=None, batch_first=False, Unalign=False, Align=True, returnlabel=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        seqs = []
        self.names=[]

        alignmap = "-ACDEFGHIKLMNPQRSTVWY"
        gaps =["-."]
        if Align:
            for records in SeqIO.parse(fastaPath,"fasta"):
                self.names.append(records.id )
                seqs.append([x for x in list(records.seq) if x in alignmap])
        elif Unalign:
            for records in SeqIO.parse(fastaPath,"fasta"):
                self.names.append(records.id )
                seqs.append([x.upper() for x in list(records.seq) if x not in gaps])
        else:
            for records in SeqIO.parse(fastaPath,"fasta"):
                self.names.append(records.id )
                seqs.append([x for x in list(records.seq)])


        self.q=len(mapstring)
        self.init_token = "<sos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk="X"
        # self.padIndex = -100

        self.mapstring = mapstring
        self.SymbolMap=dict([(mapstring[i],i) for i in range(len(mapstring))])
        self.SymbolMap[self.unk] = len(mapstring)
        self.SymbolMap[self.init_token] = len(mapstring)+1
        self.SymbolMap[self.eos_token] = len(mapstring)+2
        self.SymbolMap[self.pad_token] = len(mapstring)+3
        self.padIndex = len(mapstring)+3

        # self.inputsize = len(df.iloc[1][0].split(" "))+2
        # self.outputsize = len(df.iloc[1][1].split(" "))+2
        self.gap = "-"
        self.tensorseq=[]#torch.zeros(self.inputsize,len(df), len(self.SymbolMap))

        self.device = device
        self.transform = transform
        self.batch_first = batch_first
        self.returnlabel = returnlabel
        if self.returnlabel:
            self.mutational_effect(names)

        for i in range(seqs):
            inp = torch.tensor([self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in seq[i]]+[self.SymbolMap[self.eos_token]]).to(self.device)
            tensorseq.append(inp)


    def __len__(self):
        return(len(self.tensorseq))


    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.returnlabel:
            return self.tensorseq[idx], self.me[idx]

    def mutational_effect(self, names):
        self.me =  torch.tensor(list(map(lambda x: float(x.split("|")[2]),names))).to(self.device)

    def to(self,device):
        if device != None:
            self.tensorseq= [seq.to(device) for seq in tensorseq]
            
    # def shufflePairs(self,):
    #     self.tensorOUT= self.tensorOUT[:,torch.randperm(self.tensorOUT.size()[1])]
        
    def downsample(self, nsamples):
        idxs = torch.randperm(self.tensorOUT.size()[1])[:nsamples]
        self.tensorIN= self.tensorIN[:,idxs]
        self.tensorOUT= self.tensorOUT[:,idxs]

            

    # def join(self, pds):
    #     if self.device != pds.device:
    #         pds.tensorIN= pds.tensorIN.to(self.device, non_blocking=True)
    #         pds.tensorOUT= pds.tensorOUT.to(self.device, non_blocking=True)
    #     if self.onehot:
    #         if self.inputsize < pds.inputsize:
    #             dif = pds.inputsize - self.inputsize
    #             padIN = torch.zeros(dif, len(self), len(self.SymbolMap)).to(self.device, non_blocking=True)
    #             for i in range(len(self)):
    #                 inp = [self.SymbolMap[self.pad_token]]*dif
    #                 padIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
    #             self.tensorIN = torch.cat([torch.cat([self.tensorIN, padIN],dim=0), pds.tensorIN], dim=1)
    #             self.inputsize = pds.inputsize
    #         elif self.inputsize > pds.inputsize:
    #             dif = self.inputsize - pds.inputsize
    #             padIN = torch.zeros(dif, len(pds), len(self.SymbolMap)).to(self.device, non_blocking=True)
    #             for i in range(len(pds)):
    #                 inp = [self.SymbolMap[self.pad_token]] * dif
    #                 padIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
    #             self.tensorIN = torch.cat([self.tensorIN, torch.cat([pds.tensorIN, padIN],dim=0)], dim=1)
    #             pds.inputsize = self.inputsize
    #         if self.outputsize < pds.outputsize:
    #             dif = pds.outputsize - self.outputsize
    #             padOUT = torch.zeros(dif, self.tensorOUT.shape[1], len(self.SymbolMap)).to(self.device, non_blocking=True)
    #             for i in range(self.tensorOUT.shape[1]):
    #                 inp = [self.SymbolMap[self.pad_token]]*dif
    #                 padOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
    #             self.tensorOUT = torch.cat([torch.cat([self.tensorOUT, padOUT],dim=0), pds.tensorOUT], dim=1)
    #             self.outputsize = pds.outputsize
    #         elif self.outputsize > pds.outputsize:
    #             dif = self.outputsize - pds.outputsize
    #             padOUT = torch.zeros(dif, pds.tensorOUT.shape[1], len(self.SymbolMap)).to(self.device, non_blocking=True)
    #             for i in range(pds.tensorOUT.shape[1]):
    #                 inp = [self.SymbolMap[self.pad_token]] * dif
    #                 padOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
    #             self.tensorOUT = torch.cat([self.tensorOUT, torch.cat([pds.tensorOUT, padOUT],dim=0)], dim=1)
    #             pds.outputsize = self.outputsize
    #     else:
    #         if self.inputsize < pds.inputsize:
    #             dif = pds.inputsize - self.inputsize
    #             padIN = torch.zeros(dif, len(self)).to(self.device, non_blocking=True)
    #             for i in range(len(self)):
    #                 inp = [self.SymbolMap[self.pad_token]]*dif
    #                 padIN[:,i] = torch.tensor(inp)
    #             self.tensorIN = torch.cat([torch.cat([self.tensorIN, padIN],dim=0), pds.tensorIN], dim=1)
    #             self.inputsize = pds.inputsize
    #         elif self.inputsize > pds.inputsize:
    #             dif = self.inputsize - pds.inputsize
    #             padIN = torch.zeros(dif, len(pds)).to(self.device, non_blocking=True)
    #             for i in range(len(pds)):
    #                 inp = [self.SymbolMap[self.pad_token]] * dif
    #                 padIN[:,i] = torch.tensor(inp)
    #             self.tensorIN = torch.cat([self.tensorIN, torch.cat([pds.tensorIN, padIN],dim=0)], dim=1)
    #             pds.inputsize = self.inputsize
    #         if self.outputsize < pds.outputsize:
    #             dif = pds.outputsize - self.outputsize
    #             padOUT = torch.zeros(dif, self.tensorOUT.shape[1]).to(self.device, non_blocking=True)
    #             for i in range(self.tensorOUT.shape[1]):
    #                 inp = [self.SymbolMap[self.pad_token]]*dif
    #                 padOUT[:,i] = torch.tensor(inp)
    #             self.tensorOUT = torch.cat([torch.cat([self.tensorOUT, padOUT],dim=0), pds.tensorOUT], dim=1)
    #             self.outputsize = pds.outputsize
    #         elif self.outputsize > pds.outputsize:
    #             dif = self.outputsize - pds.outputsize
    #             padOUT = torch.zeros(dif, pds.tensorOUT.shape[1]).to(self.device, non_blocking=True)
    #             for i in range(pds.tensorOUT.shape[1]):
    #                 inp = [self.SymbolMap[self.pad_token]] * dif
    #                 padOUT[:,i] = torch.tensor(inp)
    #             self.tensorOUT = torch.cat([self.tensorOUT, torch.cat([pds.tensorOUT, padOUT],dim=0)], dim=1)
    #             pds.outputsize = self.outputsize
            
            
def getPreciseBatch(pds, idxToget):
    data = []
    for idx in idxToget:
        data.append(pds[idx])
    batch = default_collate(data)
    return batch



def getBooleanisRedundant(tensor1, tensor2):
    l1 = tensor1.shape[1]
    l2 = tensor2.shape[1]
    BooleanKept = torch.tensor([True]*l2)
    for i in range(l1):
        protein1 = tensor1[:,i,:]
        for j in range(l2):
            protein2 = tensor2[:,j,:]
            if torch.equal(protein1, protein2):
                BooleanKept[j]=False
    return BooleanKept
    


def deleteRedundancyBetweenDatasets(pds1, pds2):

    filteringOption = pds1.filteringOption
    if filteringOption == "in":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        pds2.tensorIN = pds2.tensorIN[:,a,:]
        pds2.tensorOUT = pds2.tensorOUT[:,a,:]
    elif filteringOption == "out":
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:,b,:]
        pds2.tensorOUT = pds2.tensorOUT[:,b,:]
    elif filteringOption == "and":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:,a*b,:]
        pds2.tensorOUT = pds2.tensorOUT[:,a*b,:]
    elif filteringOption == "or":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:,a+b,:]
        pds2.tensorOUT = pds2.tensorOUT[:,a+b,:]



            

# from Bio import SeqIO
# seqs = []
# names=[]
# alignmap = "-ACDEFGHIKLMNPQRSTVWY"
# gaps = "-."
# for records in SeqIO.parse("AMIE_PSEAE_1_b0.3.a2m","fasta"):
#     names.append(records.id )
#     seqs.append([x.upper() for x in list(records.seq) if x not in gaps])


# alignmap = "-ACDEFGHIKLMNPQRSTVWY"

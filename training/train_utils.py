from collections import namedtuple
import pickle
import torch
import torch.nn as nn
import numpy as np
import edgar
    

class EdgarDataset(torch.utils.data.Dataset):
    def __init__(self, df, target_column=None):
        seqs = []
        structs = []
        na_types = []
        max_slen = df.seq.apply(len).max()
        
        for seq, struct, na_type in zip(df.seq, df.struct, df.na_type):
            seq_vec, struct_vec, na_type = edgar.Model.prepare_data(seq, struct, na_type, slen=max_slen)
            seqs.append(seq_vec)
            structs.append(struct_vec)
            na_types.append(int(na_type=='rna'))
        
        self.SEQ = torch.tensor(np.array(seqs), dtype=torch.int32)
        self.STRUCT = torch.tensor(np.array(structs), dtype=torch.int32)
        self.NATYPE = torch.tensor(np.array(na_types), dtype=torch.int32)
        
        self.Y = None
        if target_column is not None:
            self.Y = torch.tensor(df[target_column], dtype=torch.float32)
        
        
    def __len__(self):
        return len(self.SEQ)
    
    
    def __getitem__(self, idx):
        d = self.SEQ[idx], self.STRUCT[idx], self.NATYPE[idx]
        
        if self.Y is None:
            return d
        
        return d, self.Y[idx]
    
    
def get_lr_scheduler(warmup=2000, peak=5e-4, c=3e-4, min_lr=1.e-4, max_lr=1.e-3):
    
    def lr_scheduler(n):

        if n<warmup:
            lr = ((peak)/warmup)*n
        else:
            lr = peak*np.exp(-c*(n-warmup))

        if lr>max_lr: lr=max_lr
        if n>warmup and lr<min_lr: lr=min_lr

        return lr
    return lr_scheduler
    
    
def input_to_device(X, device):
    tensors = []
    for x in X:
        if isinstance(x, torch.Tensor):
            tensors.append(x.to(device))
        else:
            tensors.append(x)
            
    return tuple(tensors)


def uncertainty_mse(pred, uncertainty, y, loss_lambda=1):
    logvar = uncertainty
    
    var = torch.exp(logvar)
    se = (pred - y)**2
    l = (1-loss_lambda)*se + loss_lambda*(se/var + logvar)
    
    return l.mean()


def RMSE(pred, y):
    return ((pred - y)**2).mean()


def R2(pred, y):
    yvar = ((y-y.mean())**2).sum()
    pvar = ((pred-y)**2).sum()
    return 1 - pvar/yvar

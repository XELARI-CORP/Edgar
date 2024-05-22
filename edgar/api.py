import sys
from collections import namedtuple
import warnings
from typing import Optional

if sys.version_info>=(3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources

import torch
import numpy as np

from .utils import NA_Dict, get_na_pairs
from .exceptions import InvalidSequence, InvalidStructure
from .model import Edgar



class Model:
    MODEL_PATH = 'Pretrained.pth'
    OUTPUT = namedtuple('Output', ['pred', 'uncert', 'div', 'embedding'])
    
    def __init__(self,
                gpu : bool = False, 
                device: Optional[str] = None
                ):
      
        self.device = 'cpu'
        if gpu:
            if torch.cuda.is_available():
                self.device = device if device is not None else 'cuda'
            else:
                warnings.warn('Cuda is not available, Edgar will run on cpu')
                
        model_pkg = pkg_resources.files("edgar")
        model_path = model_pkg.joinpath(self.MODEL_PATH)
        self.load_model(model_path)
        
        
    @staticmethod
    def prepare_data(seq: str, 
                     struct: str, 
                     na_type: Optional[str] = None, 
                     slen: Optional[bool] = None
                    ):
        
        if not seq:
            raise InvalidSequence("Empty sequence")
            
        rn = set(seq) - {'A', 'U', 'G', 'C', 'T'}
        if len(rn)!=0:
            raise InvalidSequence(f'Sequence contains unknown symbols: {tuple(rn)}')
            
        if len(seq)!=len(struct):
            raise InvalidSequence("Sequence and structure must be the same length")
            
        if struct.count('(')==0:
            raise InvalidStructure("Structure has no complementary bonds")
        
        if na_type is None:
            if (('U' in seq) and ('T' in seq)) or (('U' not in seq) and ('T' not in seq)):
                raise InvalidSequence("It is not obvious whether the sequence is RNA or DNA")
            
            na_type = 'rna' if 'U' in seq else 'dna'
            
        slen = len(seq) if slen is None else slen
        
        seq_vec = np.zeros(slen, dtype=np.int32)
        for i, nb in enumerate(seq):
            seq_vec[i] = NA_Dict[nb]
        
        struct_vec = -np.ones(slen, dtype=np.int32)
        pairs = get_na_pairs(struct)
        for a, b in pairs:
            struct_vec[a] = b
            struct_vec[b] = a
    
        return seq_vec, struct_vec, na_type
    
    
    def predict(self, 
                 seq : str,
                 struct: str, 
                 na_type: Optional[str] = None
                ):
        
        seq_vec, struct_vec, na_type = self.prepare_data(seq, struct, na_type, len(seq))
        mean, uncert, emb = self._predict(seq_vec, struct_vec, na_type)
        mean = mean*self.data_std + self.data_mean
        div = 2*np.sqrt(np.exp(uncert))*self.data_std
        
        return self.OUTPUT(pred=mean, uncert=uncert, div=div, embedding=emb)
        
        
    def _predict(self, seq_vec, struct_vec, na_type):
        seq_vec = torch.tensor(seq_vec, dtype=torch.int32, device=self.device)[None, ...]
        struct_vec = torch.tensor(struct_vec, dtype=torch.int32, device=self.device)[None, ...]
        na_type = torch.tensor([int(na_type=='rna')], dtype=torch.int32, device=self.device)[None, ...]
            
        # predict
        with torch.no_grad():
            pred, emb = self.model(seq_vec, struct_vec, na_type)
        
        if self.device!='cpu':
            pred, emb = pred.to('cpu'), emb.to('cpu')
            
        mean, uncert = float(pred[0, 0]), float(pred[0, 1])
        emb = emb.squeeze().numpy()
        
        return mean, uncert, emb
        
        
    def load_model(self, model_path):
        state = torch.load(model_path)
        
        weights = state['model_state_dict']
        hyperparameters = state['hyperparameters']

        model = Edgar(**hyperparameters)
        if self.device!='cpu':
            model = model.to(self.device)
        model.load_state_dict(weights)
        model.eval()
        self.model = model

        self.data_mean = state['data_mean']
        self.data_std = state['data_std']
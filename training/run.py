import sys
import os
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import edgar
from train import train_model
from train_utils import get_lr_scheduler, EdgarDataset



def set_all_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def main(experiment_name: str, 
         results_dir: Path, 
         train_data: str, 
         valid_data: str, 
         target_column: str, 
         device: str, 
         batch_size: int, 
         grad_acum: int
        ):
    
    set_all_seeds()

    # Model
    params = {
                 'vocab':6, 
                 'dim':128, 
                 'conv_layers':1, 
                 'transformer_layers':1, 
                 'heads':8 
                }
        
    model = edgar.model.Edgar(**params)
    # state = torch.load("")['model_state_dict']
    # model.load_state_dict(state)
    model = model.to(device)

    # Optim & lr
    optim = torch.optim.AdamW(model.parameters(), lr=1, weight_decay=1)
    lr_func = get_lr_scheduler(**{
                                'warmup':1500, 
                                'peak':4e-4, 
                                'c':4e-4, 
                                'min_lr':5.e-5, 
                                'max_lr':2e-4 
                                })
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    # Data
    train_df = pd.read_csv(train_data)
    data_mean, data_std = train_df[target_column].mean(), train_df[target_column].std()
    train_df[target_column] = (train_df[target_column] - data_mean)/data_std
    
    valid_df = pd.read_csv(valid_data)
    valid_df[target_column] = (valid_df[target_column] - data_mean)/data_std
    
    train_dataset = EdgarDataset(train_df, target_column)
    valid_dataset = EdgarDataset(valid_df, target_column)

    # Run
    print(f"\n\n------   {experiment_name} ------\n")
    print(f"Data mean: {data_mean:.4f}; data std: {data_std:.4f}")
    print("Model params:")
    for k, v in params.items():
        print(f"\t{k}: {v}")
        
    train_model(
                model, 
                params, 
                train_dataset, 
                valid_dataset, 
                data_mean, 
                data_std, 

                device = device, 
                grad_acum = grad_acum, 
                batch_size = batch_size, 
                valid_batch_size = batch_size, 
                epochs = int(1e4), 
                patience = 20,

                ckp_path = results_dir/experiment_name/"weights.pth", 
                log_path = results_dir/experiment_name/"logs.log", 

                lr_scheduler = lr_scheduler, 
                optim = optim, 

                clip = 0.01, 
                loss_lambda = 1
               )


if __name__=="__main__":
    
    ##################################
    ############  CONFIG  ############
    
    experiment_name = "experiment_1"
    results_dir = "results"
    train_data = "../data/RNA-DNA_train.csv"
    valid_data = "../data/RNA-DNA_valid.csv"
    target_column = "Efold"
    device = 'cuda:0'
    batch_size = 256
    grad_acum = 1
    
    ##################################
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', default=False, action='store_true')
    args = parser.parse_args()
    
    results_dir = Path(results_dir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        
    if os.path.exists(results_dir/experiment_name/"weights.pth"):
        if not args.force:
            raise ValueError(f"Experiment with name: {experiment_name} already exists")
            
    elif not os.path.exists(results_dir/experiment_name):
        os.mkdir(results_dir/experiment_name)
    
    main(experiment_name, results_dir, train_data, valid_data, target_column, device, batch_size, grad_acum)
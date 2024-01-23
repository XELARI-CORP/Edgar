import torch
import math
import numpy as np
from train_utils import input_to_device, uncertainty_mse, RMSE, R2



def inference(model, device, dataset, batch_size):
    y_pred = []
    uncertainty = []
    V = torch.utils.data.DataLoader(dataset, 
                                   batch_size=batch_size, 
                                   shuffle=False, 
                                   drop_last=False)
    
    model.eval()
    with torch.no_grad():
        for X, _ in V:
            X = input_to_device(X, device)
            p, _ = model(*X)
            y_pred.append(p[:, 0])
            uncertainty.append(p[:, 1])
        
    return torch.cat(y_pred, dim=0), torch.cat(uncertainty, dim=0)


def train_model(
                model, 
                params, 
                train_dataset, 
                valid_dataset, 
                data_mean, 
                data_std, 

                device = 'cpu', 
                grad_acum = 1, 
                batch_size = 256, 
                valid_batch_size = 32, 
                epochs = int(1e4), 
                patience = 20,

                ckp_path = None, 
                log_path = None, 

                lr_scheduler = None, 
                optim = None, 

                clip = 0.01, 
                loss_lambda = 1
               ):
    
    TrainDataN = len(train_dataset)
    train_steps = TrainDataN//batch_size
    
    valid_Y = valid_dataset.Y.to(device)
    
    step = 0
    logs = {'rmse':[], 'r2':[], 'loss':[], 'valid_rmse':[], 'valid_r2':[], 'valid_loss':[]}
    best_valid_loss = None
    best_valid_epoch = 0
    freezed_encoder = True
    
    for ep in range(1, epochs+1):
        print(f"Epoch: {ep} | {epochs}")
        
        tmp_logs = {'rmse':[], 'r2':[], 'loss':[]}
        T = torch.utils.data.DataLoader(train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=True, 
                                       drop_last=True)
        
        
        # TRAIN
        model.train()
        for i, (X, Y) in enumerate(T):
            step+=1
            
            X, Y = input_to_device(X, device), Y.to(device)
            output, _ = model(*X)
            y_pred, uncertainty = output[:, 0], output[:, 1]
            
            # calculate loss
            loss = uncertainty_mse(y_pred, uncertainty, Y, loss_lambda)
            loss = loss/grad_acum
            loss.backward()

            # step
            if step%grad_acum==0:
                if clip is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip)

                optim.step()
                optim.zero_grad()
                lr_scheduler.step()

            # metrics + logs
            loss = float(loss)
            y_pred, uncertainty = y_pred.detach(), uncertainty.detach()
            rmse = float(RMSE(y_pred, Y))
            r2 = float(R2(y_pred, Y))
            
            tmp_logs['rmse'].append(rmse)
            tmp_logs['r2'].append(r2)
            tmp_logs['loss'].append(loss)
            
            print((f'\rTrain: |{i+1}/{train_steps}| loss:{loss:.4f}; '
                   f'mse: {rmse:.4f}; r2:{r2:.4f}'), end='')
        print()
            
        # log epoch end
        logs['rmse'].append(np.mean(tmp_logs['rmse']))
        logs['r2'].append(np.mean(tmp_logs['r2']))
        logs['loss'].append(np.mean(tmp_logs['loss']))
            
        # VALIDATE
        y_pred, uncertainty = inference(model, device, valid_dataset, valid_batch_size)
        loss = uncertainty_mse(y_pred, uncertainty, valid_Y, loss_lambda)
        loss = float(loss)
        rmse = float(RMSE(y_pred, valid_Y))
        r2 = float(R2(y_pred, valid_Y))
        
        print(f'Validation: loss:{loss:.4f}; mse: {rmse:.4f}; r2:{r2:.4f}')
    
        # save logs
        logs['valid_rmse'].append(rmse)
        logs['valid_r2'].append(r2)
        logs['valid_loss'].append(loss)
        if log_path is not None:
            with open(log_path, 'w') as f:
                f.write(str(logs))
        
        # checkpoint and patience
        if best_valid_loss is None or loss<best_valid_loss:
            best_valid_loss = loss
            best_valid_epoch = ep
            
            if ckp_path is not None:
                state = {
                    'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'hyperparameters':params,
                    'data_mean':data_mean,
                    'data_std':data_std, 
                        }
                torch.save(state, ckp_path)
                print('New checkpoint')
              
        elif (ep-best_valid_epoch)>=patience:
            print(f"Stop by patience, best validation epoch: {best_valid_epoch}")
            break
            
        print()
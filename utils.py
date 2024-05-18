import torch_geometric
import torch
import os
import pickle as pkl
import numpy as np
from sklearn.metrics import roc_auc_score


def my_collate(data_list):
    ret = torch_geometric.data.Batch().from_data_list(data_list)
    
    boundary_info = []
    sample_pos = [0]
    for d in data_list:
        boundary_info.append(d.boundary_info)
        sample_pos.append(d.num_nodes)
     
    ret.sample_pos = torch.tensor(sample_pos).cumsum(0)
    ret.boundary_info = boundary_info
    
    return ret


def evaluate(dataloader, model, device):
    num_samples = 0
    correct = 0 
    
    model = model.eval().to(device)
    pred_labs = []
    true_labs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if not hasattr(batch, 'node_lab'): batch.node_lab = None
            # batch.boundary_info = [e.to(device) for e in batch.boundary_info]
            
            y_hat = model(batch)
            
            y_pred = y_hat.max(dim=1)[1]  
            pred_labs.append(y_pred.detach().cpu().numpy())
            true_labs.append(batch.y.detach().cpu().numpy())  
            
            correct += (y_pred == batch.y).sum().item()
            num_samples += batch.y.size(0)
    test_acc = float(correct)/ float(num_samples)
    true_labs = np.concatenate(true_labs)
    pred_labs = np.concatenate(pred_labs)
    roc_score = roc_auc_score(true_labs, pred_labs)
    return test_acc, roc_score

def evaluate_filtration(dataloader, model, device, **kwargs):
    num_samples = 0
    correct = 0 
    
    model = model.eval().to(device)
    running_mae = 0
    running_mse = 0
    crit = torch.nn.MSELoss()
    testing_filt = []
    testing_gril = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if not hasattr(batch, 'node_lab'): batch.node_lab = None
            # batch.boundary_info = [e.to(device) for e in batch.boundary_info]
            
            f_hat, gril_h0, gril_h1 = model(batch)    
            
            # error = torch.abs(f_hat - batch.filt).sum().data
            squared_error = crit(gril_h0, batch.h0).item()
            # running_mae += error
            running_mse += squared_error
            num_samples += batch.y.size(0)
            testing_filt.append(f_hat.cpu().numpy())
            testing_gril.append(gril_h0.cpu().numpy())
        
        save = kwargs.get('save', False)
        epoch_i = kwargs.get('epoch_num', 1)
        ds_type = kwargs.get('ds_type', 'test')
        if save:
            if not os.path.exists("saved_filtrations_grils"):
                os.makedirs("saved_filtrations_grils", exist_ok=True)
            
            with open(f"saved_filtrations_grils/{ds_type}_filtration_epoch_{epoch_i}.pkl", 'wb') as fid:
                pkl.dump(testing_filt, fid)
            
            with open(f"saved_filtrations_grils/{ds_type}_gril_epoch_{epoch_i}.pkl", 'wb') as fid:
                pkl.dump(testing_filt, fid)
        
    # mse = torch.sqrt(running_mse/len(dataloader))
    return running_mse / num_samples


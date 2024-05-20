import torch
import torch.nn as nn
import os
import argparse
from torch_geometric.datasets import TUDataset
import torch_geometric
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from model import ClassifierHead
from data import dataset_factory, train_test_val_split
import sys
import uuid
import datetime
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from multipers.multipers import MultiPersistenceImageWrapper, MultiPersistenceLandscapeWrapper, PersistenceImageWrapper, SubsampleWrapper


from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import json

def prepare_signature(exp_cfg, X, train_index, test_index, validation_index):
    signature = exp_cfg['signature']
    params = exp_cfg['params']
    print(f'Preparing signature {signature} with params {params}')
    mls_wrap = None
    if signature == 'mls_50':
        X_train  = [[X[nf][n] for nf in range(len(X))] for n in train_index]
        X_test   = [[X[nf][n] for nf in range(len(X))] for n in test_index]
        X_val    = [[X[nf][n] for nf in range(len(X))] for n in validation_index]
        mls_wrap = MultiPersistenceLandscapeWrapper(power=params['power'], step=params['step'], k=params['k'])
        X_train = mls_wrap.transform(X_train)
        X_test = mls_wrap.transform(X_test)
        X_val = mls_wrap.transform(X_val)
    elif signature == 'mpi_50':
        X_train  = [[X[nf][n] for nf in range(len(X))] for n in train_index]
        X_test   = [[X[nf][n] for nf in range(len(X))] for n in test_index]
        X_val    = [[X[nf][n] for nf in range(len(X))] for n in validation_index]
        mls_wrap = MultiPersistenceImageWrapper(bdw=params['bdw'], power=params['power'], step=params['step'])
        X_train = mls_wrap.transform(X_train)
        X_test = mls_wrap.transform(X_test)
        X_val = mls_wrap.transform(X_val)
    elif signature == 'mk':
        # Xmk = sum([  X[nf] for nf in range(len(X))  ])
        # Xmk = np.hstack(X)
        scaler = MinMaxScaler()
        h0, h1 = X[0], X[1]
        X_train, X_test, X_val = np.hstack([h0[train_index,:][:,train_index], h1[train_index, :][:, train_index]]) , np.hstack([h0[test_index,:][:,train_index], h1[test_index, :][:, train_index]]), np.hstack([h0[validation_index,:][:,train_index], h1[validation_index, :][:, train_index]])
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)
    elif signature == 'pi_50':
        X_train  = [[X[nf][n] for nf in range(len(X))] for n in train_index]
        X_test   = [[X[nf][n] for nf in range(len(X))] for n in test_index]
        X_val    = [[X[nf][n] for nf in range(len(X))] for n in validation_index]
        mls_wrap = PersistenceImageWrapper(bdw=params['bdw'], power=params['power'], step=params['step'])
        X_train = mls_wrap.transform(X_train)
        X_test = mls_wrap.transform(X_test)
        X_val = mls_wrap.transform(X_val)
    elif signature == 'ls':
        Xls = np.hstack(X)
        mls_wrap = SubsampleWrapper(step=params['step'])
        X_train, X_test, X_val = Xls[train_index,:], Xls[test_index,:], Xls[validation_index,:]
        X_train = mls_wrap.transform(X_train)
        X_test = mls_wrap.transform(X_test)
        X_val = mls_wrap.transform(X_val)
    elif signature == 'gril_50' or signature == 'nn_gril':
        X_train  = [[X[nf][n] for nf in range(len(X))] for n in train_index]
        X_test   = [[X[nf][n] for nf in range(len(X))] for n in test_index]
        X_val    = [[X[nf][n] for nf in range(len(X))] for n in validation_index]
        mls_wrap = MultiPersistenceLandscapeWrapper(power=params['power'], step=params['step'], k=params['k'])
        X_train = mls_wrap.transform(X_train)
        X_test = mls_wrap.transform(X_test)
        X_val = mls_wrap.transform(X_val)
    else:
        raise ValueError(f'Unknown signature {signature}')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    
    return X_train, X_test, X_val



def get_balance(split):
    y = np.array([d.y.item() for d in split])
    balance = np.unique(y, return_counts=True)[1]
    inv_wt = balance[::-1] / balance.sum()
    return torch.tensor(inv_wt, dtype=torch.float32)
    



    
class LinClassifier(nn.Module):
    def __init__(self, dataset, in_features, cls_hidden_dimension, drop_out=0.1) -> None:
        super().__init__()
        self.lin_h0 = nn.Linear(in_features, cls_hidden_dimension)
        self.lin_h1 = nn.Linear(in_features, cls_hidden_dimension)
        self.act = F.relu
        cls_in_dim = 2 * cls_hidden_dimension
        self.cls = ClassifierHead(dataset, dim_in= cls_in_dim, hidden_dim=cls_hidden_dimension, drop_out=drop_out)
        

    def forward(self, h0, h1):
        lmbda_0 = self.act(self.lin_h0(h0))
        lmbda_1 = self.act(self.lin_h1(h1))
        lmbda = torch.cat([lmbda_0, lmbda_1]).reshape((1, -1))
        # lmbda = lmbda_0.reshape((1, -1))
        # lmbda = lmbda.to(gpu_dev)
        lmbda = self.act(lmbda)
        z = self.cls(lmbda)
        return z
    
class LinClassifier2(nn.Module):
    def __init__(self, dataset, in_features, cls_hidden_dimension, drop_out=0.1) -> None:
        super().__init__()
        
        self.lin_1 = nn.Linear(2 * in_features, cls_hidden_dimension)
        # self.lin_2 = nn.Linear(in_features, cls_hidden_dimension)
        self.act = F.relu
        cls_in_dim = cls_hidden_dimension
        self.cls = ClassifierHead(dataset, dim_in= cls_in_dim, hidden_dim=cls_hidden_dimension, drop_out=drop_out)
        

    def forward(self, h0, h1):
        # lmbda_0 = self.act(self.lin_h0(h0))
        # lmbda_1 = self.act(self.lin_h1(h1))
        lmbda = torch.cat([h0, h1]).reshape((1, -1))
        # lmbda = lmbda_0.reshape((1, -1))
        # lmbda = lmbda.to(gpu_dev)
        lmbda = self.act(self.lin_1(lmbda))
        z = self.cls(lmbda)
        return z
    
def evaluate_ls(dataloader, ls, model, device):
    input_dim = ls.shape[1]
    num_samples = 0
    correct = 0 
    
    model = model.eval().to(device)
    pred_labs = []
    true_labs = []
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            batch = batch.to(device)
            if not hasattr(batch, 'node_lab'): batch.node_lab = None
            # batch.boundary_info = [e.to(device) for e in batch.boundary_info]
            h0_ls_ = ls[batch_i, 0:input_dim//2].to(device)
            h1_ls_ = ls[batch_i, input_dim//2:].to(device)
            y_hat = model(h0_ls_, h1_ls_)
            
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

def experiment(exp_cfg, device, output_dir=None, verbose=True, output_cache=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training", flush=True)
    
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    training_cfg = exp_cfg['training']
    model_cfg = exp_cfg['model']

    dataset = dataset_factory(exp_cfg['dataset_name'], verbose=verbose)
    
    split_ds, split_i = train_test_val_split(
        dataset,
        validation_ratio=training_cfg['validation_ratio'],
        n_splits=args.num_folds,
        verbose=verbose)

    cv_test_acc = [[] for _ in range(len(split_ds))]
    cv_val_acc = [[] for _ in range(len(split_ds))]
    cv_test_roc = [[] for _ in range(len(split_ds))]
    cv_val_roc = [[] for _ in range(len(split_ds))]
    cv_epoch_loss = [[] for _ in range(len(split_ds))]

    uiid = str(uuid.uuid4())

    if output_dir is not None:
        output_path = os.path.join(output_dir, uiid + '.pickle')

    ret = {} if output_cache is None else output_cache

    ret['exp_cfg'] = exp_cfg
    ret['cv_test_acc'] = cv_test_acc
    ret['cv_val_acc']  = cv_val_acc
    ret['cv_test_roc'] = cv_test_roc
    ret['cv_val_roc']  = cv_val_roc
    ret['cv_indices_trn_tst_val'] = split_i
    ret['cv_epoch_loss'] = cv_epoch_loss
    ret['start_time'] = str(datetime.datetime.now())
    ret['id'] = uiid
    ret['finished_training'] = False
    # saved_model_pth = os.path.join("experiment_logs",str(datetime.datetime.now()).replace(' ', '_').split('.')[0])
    # if not os.path.exists(saved_model_pth):
    #     os.makedirs(saved_model_pth, exist_ok=True)
    
    topofeat_path = os.path.join('./graph_landscapes', exp_cfg['dataset_name'])
    if exp_cfg['signature'] == 'gril_50':
        h0_ls_pth = os.path.join(topofeat_path, f'{exp_cfg["signature"]}_HKS-RC-0.pkl')
        h1_ls_pth = os.path.join(topofeat_path, f'{exp_cfg["signature"]}_HKS-RC-1.pkl')
    elif exp_cfg['signature'] == 'nn_gril':
        h0_ls_pth = None
        h1_ls_pth = None
    else:
        h0_ls_pth = os.path.join(topofeat_path, f'{exp_cfg["signature"]}_Alpha-DTM-0.pkl')
        h1_ls_pth = os.path.join(topofeat_path, f'{exp_cfg["signature"]}_Alpha-DTM-1.pkl')
    
    if h0_ls_pth is not None and h1_ls_pth is not None:
        with open(h0_ls_pth, 'rb') as fid:
            h0 = pickle.load(fid)
        with open(h1_ls_pth, 'rb') as fid:
            h1 = pickle.load(fid)
        X = [h0, h1]

    for fold_i, (train_split, test_split, validation_split) in enumerate(split_ds):
        
        # model = model_factory(model_cfg, dataset).to(device)
        train_split_i, test_split_i, validation_split_i = split_i[fold_i]
        if exp_cfg['signature'] == 'nn_gril':
            topofeat_path = f'./saved_filtrations_grils/{exp_cfg["dataset_name"]}'
            h0_ls_pth = os.path.join(topofeat_path, f'{exp_cfg["dataset_name"]}_{fold_i}-0.pkl')
            h1_ls_pth = os.path.join(topofeat_path, f'{exp_cfg["dataset_name"]}_{fold_i}-1.pkl') 
            with open(h0_ls_pth, 'rb') as fid:
                h0 = pickle.load(fid)
            with open(h1_ls_pth, 'rb') as fid:
                h1 = pickle.load(fid)
            X = [h0, h1]
        
        ls_train, ls_test, ls_val = prepare_signature(exp_cfg, X, train_split_i, test_split_i, validation_split_i)
        input_dim = ls_train.shape[1]        
        
        model = LinClassifier2(dataset, in_features=input_dim//2, cls_hidden_dimension=model_cfg['cls_hidden_dimension']).to(device)
        if verbose and fold_i == 0:
            print(model, flush=True)

        opt = optim.Adam(
            model.parameters(),
            lr=training_cfg['lr'],
            weight_decay=training_cfg['weight_decay']
        )

        scheduler = MultiStepLR(opt,
                                milestones=list(range(0,
                                                      training_cfg['num_epochs'],
                                                      training_cfg['epoch_step'])
                                               )[1:],
                                gamma=training_cfg['lr_drop_fact'])

        dl_train = torch_geometric.loader.DataLoader(
            train_split,
            # collate_fn=my_collate,
            batch_size=training_cfg['batch_size'],
            shuffle=False,
            # if last batch would have size 1 we have to drop it ...
            drop_last=(len(train_split) % training_cfg['batch_size'] == 1)
        )

        dl_test = torch_geometric.loader.DataLoader(
            test_split ,
            # collate_fn=my_collate,
            batch_size=training_cfg['batch_size'],
            shuffle=False
        )

        dl_val = None
        if training_cfg['validation_ratio'] > 0:
            dl_val = torch_geometric.loader.DataLoader(
                validation_split,
                # collate_fn=my_collate,
                batch_size=training_cfg['batch_size'],
                shuffle=False
            )
        
        wt = get_balance(train_split).to(device)
        for epoch_i in range(1, training_cfg['num_epochs'] + 1):
            model.train()
            epoch_loss = 0
            num_correct = 0
            num_total = 0
            pred_labs, true_labs = [], []
            for batch_i, batch in enumerate(dl_train, start=1):

                batch = batch.to(device)
                if not hasattr(batch, 'node_lab'): batch.node_lab = None
                # batch.boundary_info = [e.to(device) for e in batch.boundary_info]
                h0_train_ = ls_train[batch_i-1, 0:input_dim//2].to(device)
                h1_train_ = ls_train[batch_i-1, input_dim//2:].to(device)
                y_hat = model(h0_train_, h1_train_)

                loss = torch.nn.functional.cross_entropy(y_hat, batch.y, weight=wt)
                opt.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                opt.step()
                y_pred = y_hat.max(dim=1)[1]
                pred_labs.append(y_pred.detach().cpu().numpy())
                true_labs.append(batch.y.detach().cpu().numpy())
                num_correct += (y_pred == batch.y).sum().item()
                num_total += batch.y.size(0)
                # if verbose:
                #     print("Epoch {}/{}, Batch {}/{}".format(
                #         epoch_i,
                #         training_cfg['num_epochs'],
                #         batch_i,
                #         len(dl_train)),
                #         end='\r', flush=True)

                # break # todo remove!!!
            
            # scheduler.step()
            
            if verbose: print('')
            pred_labs = np.concatenate(pred_labs)
            true_labs = np.concatenate(true_labs)
            train_roc = roc_auc_score(true_labs, pred_labs)
            test_acc, test_roc = evaluate_ls(dl_test, ls_test, model, device)
            cv_test_acc[fold_i].append(test_acc*100.0)
            cv_test_roc[fold_i].append(test_roc*100.0)
            cv_epoch_loss[fold_i].append(epoch_loss)

            val_acc = None
            if training_cfg['validation_ratio'] > 0.0:
                val_acc, val_roc = evaluate_ls(dl_val, ls_val, model, device)
                cv_val_acc[fold_i].append(val_acc * 100.0)
                cv_val_roc[fold_i].append(val_roc * 100.0)

            if verbose:
                epoch_loss = epoch_loss / num_total
                train_acc = num_correct / num_total
                print("Fold {}/{} loss {:.2f} | train_acc {:.2f} | test_acc {:.2f} | val_acc {:.2f}".format(fold_i, len(split_ds), epoch_loss, train_acc*100, test_acc*100.0, val_acc*100.0), flush=True)
                print("train_roc {:.2f} | test_roc {:.2f} | val_roc {:.2f}".format(train_roc*100, test_roc*100.0, val_roc*100.0), flush=True)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        model_tag = exp_cfg['tag']
        # if saved_model_pth is not None:
        #     model_file = os.path.join(saved_model_pth,  f'{exp_cfg["dataset_name"]}_{model_tag}_{fold_i}.pth')
        #     torch.save(model.state_dict(), model_file)

        #     with open(output_path, 'bw') as fid:
        #         pickle.dump(file=fid, obj=ret)

    ret['finished_training'] = True
    # if output_dir is not None:
    #     with open(output_path, 'bw') as fid:
    #         pickle.dump(file=fid, obj=ret)

    return ret



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    device = torch.device(device)
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SimpleNNBaseline')
    parser.add_argument('--dataset', type=str, default='EGFR')
    parser.add_argument('--num_folds', type=int, default=5)
    args = parser.parse_args()
    uuid_str = str(uuid.uuid4())

    __training_cfg = {
    'lr': 1e-2,
    'lr_drop_fact': 0.5,
    'num_epochs': 50,
    'epoch_step': 10,
    'batch_size': 1,
    'weight_decay': 1e-6,
    'validation_ratio': 0.1,
    }


    __model_cfg_meta = {
        'model_type': args.model,
        'use_super_level_set_filtration': False,
        'use_node_degree': False,
        'set_node_degree_uninformative': False,
        'pooling_strategy': 'sum',
        'use_node_label': False,
        'input_features_dim': 82,
        'use_node_features': True,
        'gin_number': 1,
        'gin_dimension': 64,
        'gin_mlp_type': 'lin_bn_lrelu_lin',
        'num_struct_elements': 100,
        'cls_hidden_dimension': 256,
        'drop_out': 0.1,
        'ls_dim': 5000,
        }
    
    with open('static_params.json', 'r') as fid:
        grid_search_params = json.load(fid)
    
    static_results = {'dataset': args.dataset, 'model': args.model}
    uiid = str(uuid.uuid4())
    output_dir = 'static_experiment_logs'

    if not os.path.exists(output_dir): 
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, uiid + '.json')
    
    for k, v in grid_search_params.items():
        signature = k
        params = v
        static_results[signature] = []
        for i in range(len(params)):
            __params_list_mls = params[i]
            result_dict = params[i].copy()
            __training_cfg['lr'] = __params_list_mls['lr']
            __exp_cfg_meta = {
                'dataset_name': args.dataset,
                'training': __training_cfg,
                'model': __model_cfg_meta,
                'tag': args.model,
                'signature': signature,
                'params': __params_list_mls,
                }

            __exp_res_meta = {
                'exp_cfg': __exp_cfg_meta,
                'cv_test_acc': list,
                'cv_val_acc': list,
                'cv_indices_trn_tst_val': list,
                'cv_epoch_loss': list,
                'start_time': list,
                'id': str
                }
            res = experiment(exp_cfg=__exp_cfg_meta, device='cuda', output_dir='experiment_logs', verbose=True)
            cv_acc_last = [x[-1] for x in res['cv_test_acc'] if len(x) > 0]
            cv_roc_last = [x[-1] for x in res['cv_test_roc'] if len(x) > 0]
            result_dict['mean_test_acc'] = np.mean(cv_acc_last)
            result_dict['std_test_acc'] = np.std(cv_acc_last)
            result_dict['mean_tes_roc'] = np.mean(cv_roc_last)
            result_dict['std_test_roc'] = np.std(cv_roc_last)
            static_results[signature].append(result_dict)
    with open(output_path, 'w') as fid:
        json.dump(static_results, fid, indent=4)        

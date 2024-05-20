import os.path as osp
import itertools
import copy
import uuid
import os
import pickle
import datetime

import torch
import torch.nn.functional as F
import torch_geometric

import numpy as np

import torch.nn as nn
import torch.optim as optim



from torch.nn import Sequential, Linear, ReLU
from torch.optim.lr_scheduler import MultiStepLR

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool

from collections import defaultdict, Counter

from model import GIN, SimpleNNBaseline, GIN_MPML, GCN_MPML, GCN, GAT_MPML, GAT, GIN_MPML_Learned
from data import dataset_factory, train_test_val_split, EGFRDatasetGeometric
from utils import my_collate, evaluate
from sklearn.metrics import roc_auc_score
import argparse
import sys
import glob

import torch.multiprocessing as mp
# try:
#      mp.set_start_method('spawn')
# except RuntimeError:
#     pass


__training_cfg = {
    'lr': float,
    'lr_drop_fact': float,
    'num_epochs': int,
    'epoch_step': int,
    'batch_size': int,
    'weight_decay': float,
    'validation_ratio': float,
}


__model_cfg_meta = {
    'model_type': str,
    'use_super_level_set_filtration': bool,
    'use_node_degree': bool,
    'set_node_degree_uninformative': bool,
    'pooling_strategy': str,
    'use_node_label': bool,
    'gin_number': int,
    'gin_dimension': int,
    'gin_mlp_type': str,
    'num_struct_elements': int,
    'cls_hidden_dimension': int,
    'drop_out': float,
}


__exp_cfg_meta = {
    'dataset_name': str,
    'training': __training_cfg,
    'model': __model_cfg_meta,
    'tag': str
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


def model_factory(model_cfg: dict, dataset):
    str_2_type = {
        'GCN_MPML': GCN_MPML,
        'GAT_MPML': GAT_MPML,
        'GIN_MPML': GIN_MPML_Learned,
        'GAT': GAT,
        'GIN': GIN,
        'GCN': GCN, 
        'SimpleNNBaseline': SimpleNNBaseline
    }

    model_type = model_cfg['model_type']
    Model= str_2_type[model_type]
    return Model(dataset, **model_cfg)



def experiment(exp_cfg, device, output_dir=None, verbose=True, output_cache=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training", flush=True)
    training_cfg = exp_cfg['training']
    model_cfg = exp_cfg['model']
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    dataset = dataset_factory(exp_cfg['dataset_name'], verbose=verbose)
    if model_cfg['use_node_features']:
        model_cfg['input_features_dim'] = dataset[0].x.shape[1]
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
        output_path = osp.join(output_dir, uiid + '.pickle')

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
    saved_model_pth = os.path.join("experiment_logs",str(datetime.datetime.now()).replace(' ', '_').split('.')[0])
    if not os.path.exists(saved_model_pth):
        os.makedirs(saved_model_pth, exist_ok=True)
    
    for fold_i, (train_split, test_split, validation_split) in enumerate(split_ds):
        
        model = model_factory(model_cfg, dataset).to(device)

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
            shuffle=True,
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

                y_hat = model(batch)

                loss = torch.nn.functional.cross_entropy(y_hat, batch.y)
                opt.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                opt.step()
                y_pred = y_hat.max(dim=1)[1]
                pred_labs.append(y_pred.detach().cpu().numpy())
                true_labs.append(batch.y.detach().cpu().numpy())
                num_correct += (y_pred == batch.y).sum().item()
                num_total += batch.y.size(0)
                if verbose and (batch_i == 1 or batch_i % 100 == 0):
                    print("Epoch {}/{}, Batch {}/{}".format(
                        epoch_i,
                        training_cfg['num_epochs'],
                        batch_i,
                        len(dl_train)),
                        end='\r', file=sys.stderr, flush=True)

                # break # todo remove!!!
            
            scheduler.step()
            
            if verbose: print('')
            pred_labs = np.concatenate(pred_labs)
            true_labs = np.concatenate(true_labs)
            train_roc = roc_auc_score(true_labs, pred_labs)
            test_acc, test_roc = evaluate(dl_test, model, device)
            cv_test_acc[fold_i].append(test_acc*100.0)
            cv_test_roc[fold_i].append(test_roc*100.0)
            cv_epoch_loss[fold_i].append(epoch_loss)

            val_acc = None
            if training_cfg['validation_ratio'] > 0.0:
                val_acc, val_roc = evaluate(dl_val, model, device)
                cv_val_acc[fold_i].append(val_acc * 100.0)
                cv_val_roc[fold_i].append(val_roc * 100.0)

            if verbose:
                epoch_loss = epoch_loss / num_total
                train_acc = num_correct / num_total
                print("loss {:.2f} | train_acc {:.2f} | test_acc {:.2f} | val_acc {:.2f}".format(epoch_loss, train_acc*100, test_acc*100.0, val_acc*100.0), flush=True)
                print("train_roc {:.2f} | test_roc {:.2f} | val_roc {:.2f}".format(train_roc*100, test_roc*100.0, val_roc*100.0), flush=True)

        # break #todo remove!!!
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        model_tag = exp_cfg['tag']
        if saved_model_pth is not None:
            model_file = osp.join(saved_model_pth,  f'{exp_cfg["dataset_name"]}_{model_tag}_{fold_i}.pth')
            torch.save(model.state_dict(), model_file)

            with open(output_path, 'bw') as fid:
                pickle.dump(file=fid, obj=ret)

    ret['finished_training'] = True
    if output_dir is not None:
        with open(output_path, 'bw') as fid:
            pickle.dump(file=fid, obj=ret)

    return ret


def experiment_task(args):
    exp_cfg, output_dir, device_counter, lock, max_process_on_device = args

    with lock:
        device = None
        for k, v in device_counter.items():
            if v < max_process_on_device:
                device_id = k
                device = 'cuda:{}'.format(device_id)

                break
        device_counter[device_id] += 1

    assert device is not None

    try:
        print(exp_cfg['dataset_name'], flush=True)
        experiment(exp_cfg, device, output_dir=output_dir, verbose=False)
        device_counter[device_id] -= 1

    except Exception as ex:
        ex.exp_cfg = exp_cfg
        device_counter[device_id] -= 1

        return ex


def experiment_multi_device(exp_cfgs, output_dir, visible_devices, max_process_on_device):
    assert isinstance(exp_cfgs, list)
    assert isinstance(visible_devices, list)
    assert osp.isdir(output_dir)
    assert all((i < torch.cuda.device_count() for i in visible_devices))

    num_device = len(visible_devices)

    manager = mp.Manager()
    device_counter = manager.dict({t: 0 for t in visible_devices})
    lock = manager.Lock()

    task_args = [(exp_cfg, output_dir, device_counter, lock, max_process_on_device) for exp_cfg in exp_cfgs]

    ret = []
    with mp.Pool(num_device*max_process_on_device, maxtasksperchild=1) as pool:

        for i, r in enumerate(pool.imap_unordered(experiment_task, task_args)):
            ret.append(r)

            if r is None:
                print("# Finished job {}/{}".format(i + 1, len(task_args)))

            else:
                print("#")
                print("# Error in job {}/{}".format(i, len(task_args)))
                print("#")
                print("# Error:")
                print(r)
                print("# experiment configuration:")
                print(r.exp_cfg)

    ret = [r for r in ret if r is not None]
    if len(ret) > 0:
        with open(osp.join(output_dir, 'errors.pickle'), 'bw') as fid:
            pickle.dump(obj=ret, file=fid)

def find_candidate_logs(exp_cfg):
    candidate_file_names = []
    exp_logs = glob.glob(f"./experiment_logs/*.pickle")
    for exp_log in exp_logs:
        with open(exp_log, "rb") as res:
            ret = pickle.load(res)
        saved_exp_cfg = ret['exp_cfg']
        if saved_exp_cfg == exp_cfg:
            candidate_file_names.append((exp_log, ret['start_time']))
    candidate_file_names = sorted(candidate_file_names, key=lambda x:x[1])
    return candidate_file_names


def restart_experiment(exp_cfg, device, output_dir=None, verbose=True, output_cache=None):
    try:
        assert output_cache is not None
    except:
        raise Exception("Restarting experiment is only available with matching output_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training", flush=True)
    training_cfg = exp_cfg['training']
    model_cfg = exp_cfg['model']
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    dataset = dataset_factory(exp_cfg['dataset_name'], verbose=verbose)
    if model_cfg['use_node_features']:
        model_cfg['input_features_dim'] = dataset[0].x.shape[1]
    
    
    split_ds, split_i = train_test_val_split(
        dataset,
        validation_ratio=training_cfg['validation_ratio'],
        n_splits=args.num_folds,
        verbose=verbose)
    
    with open(output_cache, "rb") as res_cache:
        ret = pickle.load(res_cache)

    cv_test_acc = ret['cv_test_acc']
    cv_val_acc = ret['cv_val_acc']
    cv_epoch_loss = ret['cv_epoch_loss']

    ret['cv_test_acc'] = cv_test_acc
    ret['cv_val_acc']  = cv_val_acc
    # ret['cv_indices_trn_tst_val'] = split_i
    ret['cv_epoch_loss'] = cv_epoch_loss

    # uiid = str(uuid.uuid4())

    # if output_dir is not None:
    #     output_path = osp.join(output_dir, uiid + '.pickle')

    
    
    saved_model_pth = os.path.join("experiment_logs",str(datetime.datetime.now()).replace(' ', '_').split('.')[0])
    if not os.path.exists(saved_model_pth):
        os.makedirs(saved_model_pth, exist_ok=True)
    # pickle.dump(split_i, open(os.path.join(saved_model_pth, f"{exp_cfg['dataset_name']}_fold_splits.pkl"), "wb"))
    
    cv_folds_available = sum([1 for cv in ret['cv_test_acc'] if len(cv) == ret['exp_cfg']['training']['num_epochs']])

    
    for fold_i, (train_split, test_split, validation_split) in enumerate(split_ds):
        if fold_i < cv_folds_available:
            continue
        model = model_factory(model_cfg, dataset).to(device)

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
            shuffle=True,
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

                y_hat = model(batch)

                loss = torch.nn.functional.cross_entropy(y_hat, batch.y)
                opt.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                opt.step()
                y_pred = y_hat.max(dim=1)[1]
                pred_labs.append(y_pred.detach().cpu().numpy())
                true_labs.append(batch.y.detach().cpu().numpy())
                num_correct += (y_pred == batch.y).sum().item()
                num_total += batch.y.size(0)
                if verbose and (batch_i == 1 or batch_i % 100 == 0):
                    print("Epoch {}/{}, Batch {}/{}".format(
                        epoch_i,
                        training_cfg['num_epochs'],
                        batch_i,
                        len(dl_train)),
                        end='\r', file=sys.stderr, flush=True)

                # break # todo remove!!!
            
            scheduler.step()
            
            if verbose: print('')
            pred_labs = np.concatenate(pred_labs)
            true_labs = np.concatenate(true_labs)
            train_roc = roc_auc_score(true_labs, pred_labs)
            test_acc, test_roc = evaluate(dl_test, model, device)
            cv_test_acc[fold_i].append(test_acc*100.0)
            cv_epoch_loss[fold_i].append(epoch_loss)

            val_acc = None
            if training_cfg['validation_ratio'] > 0.0:
                val_acc, val_roc = evaluate(dl_val, model, device)
                cv_val_acc[fold_i].append(val_acc * 100.0)

            if verbose:
                epoch_loss = epoch_loss / num_total
                train_acc = num_correct / num_total
                print("loss {:.2f} | train_acc {:.2f} | test_acc {:.2f} | val_acc {:.2f}".format(epoch_loss, train_acc*100, test_acc*100.0, val_acc*100.0), flush=True)
                print("train_roc {:.2f} | test_roc {:.2f} | val_roc {:.2f}".format(train_roc*100, test_roc*100.0, val_roc*100.0), flush=True)

        # break #todo remove!!!
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        model_tag = exp_cfg['tag']
        if saved_model_pth is not None:
            model_file = osp.join(saved_model_pth,  f'{exp_cfg["dataset_name"]}_{model_tag}_{fold_i}.pth')
            torch.save(model.state_dict(), model_file)

            with open(output_cache, 'bw') as fid:
                pickle.dump(file=fid, obj=ret)

    ret['finished_training'] = True
    if output_dir is not None:
        with open(output_cache, 'bw') as fid:
            pickle.dump(file=fid, obj=ret)

    return ret



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GIN_MPML', choices=['GIN', 'GCN', 'GAT', 'SimpleNNBaseline', 'GIN_MPML', 'GCN_MPML', 'GAT_MPML'])
    parser.add_argument('--dataset', type=str, default='EGFR')
    parser.add_argument('--fp', default=None, choices=['maccs', 'ecfp', 'morgan2', 'morgan3'])
    parser.add_argument('--num_folds', type=int, default=5)
    args = parser.parse_args()


    __training_cfg = {
    'lr': 1e-4,
    'lr_drop_fact': 0.5,
    'num_epochs': 25,
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
        'cls_hidden_dimension': 32,
        'drop_out': 0.1,
        'ls_dim': 5000,
        'concat_fp': args.fp,
        }


    __exp_cfg_meta = {
        'dataset_name': args.dataset,
        'training': __training_cfg,
        'model': __model_cfg_meta,
        'tag': args.model
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
    experiment(exp_cfg=__exp_cfg_meta, device='cuda', output_dir='experiment_logs', verbose=True)
    # candidate_logs = find_candidate_logs(__exp_cfg_meta)
    # print(candidate_logs)
    # cached_log_to_read_from = candidate_logs[0][0]
    # print(f'Reading cached output from {cached_log_to_read_from}')
    # restart_experiment(exp_cfg=__exp_cfg_meta, device='cuda',output_dir='experiment_logs', verbose=True, output_cache=cached_log_to_read_from)



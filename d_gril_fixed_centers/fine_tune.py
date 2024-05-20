import torch
import torch.nn as nn
from model import GIN, SimpleNNBaseline, GIN_MPML, GCN_MPML, GCN, GAT_MPML, GAT, GIN_MPML_Learned, GIN_MPML_GRILHead
from data import dataset_factory, train_test_val_split
from utils import my_collate, evaluate
import argparse
import pickle
import glob
import torch_geometric
import datetime
import os

def test_model(args, probable_candidate, verbose=True, num_folds=5):
    print(f"Loading from {probable_candidate[0]} start-time: {probable_candidate[1]}")
    with open(f"{probable_candidate[0]}", "rb") as log_file:
        res = pickle.load(log_file)
    exp_cfg = res['exp_cfg']
    model_cfg = exp_cfg['model']
    subpth = res['start_time'].replace(' ', '_').split('.')[0]
    output_dir = f'./saved_filtrations_grils/{subpth}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/{os.path.basename(probable_candidate[0])}", "wb") as metadata:
        pickle.dump(res, metadata)
    ds_name = args.dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset_factory(ds_name, verbose=verbose)
    data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
    for fold_i in range(num_folds):
        gril_h0 = []
        gril_h1 = []
        filt = []
        model_pth = f'{args.pth}/{ds_name}_{args.model}_{fold_i}.pth'
        model = GIN_MPML_GRILHead(model_pth, dataset, model_cfg)
        model.to(device)
        # model.load_state_dict(torch.load(f'./saved_models/{ds_name}/{args.model}/{fold_i}.pth'))
        model.eval()
        with torch.no_grad():
            for batch_i, batch in enumerate(data_loader, start=1):
                batch = batch.to(device)
                if not hasattr(batch, 'node_lab'): batch.node_lab = None
                # batch.boundary_info = [e.to(device) for e in batch.boundary_info]
                f_v, gril_values_h0, gril_values_h1 = model(batch)
                gril_h0.append(gril_values_h0.detach().cpu().numpy())
                gril_h1.append(gril_values_h1.detach().cpu().numpy())
                filt.append(f_v.detach().cpu().numpy())
                print(f'Batch {batch_i} / {len(data_loader)} done.', end='\r')
            print()
            h0_save_path = f'{output_dir}/{ds_name}_{fold_i}-0.pkl'
            with open(h0_save_path, 'wb') as f:
                pickle.dump(gril_h0, f)
            h1_save_path = f'{output_dir}/{ds_name}_{fold_i}-1.pkl'
            with open(h1_save_path, 'wb') as f:
                pickle.dump(gril_h1, f)
            filt_save_path = f'{output_dir}/{ds_name}_{fold_i}-filt.pkl'
            with open(filt_save_path, 'wb') as f:
                pickle.dump(filt, f)
            print(f'Fold {fold_i + 1} / {num_folds} done. GRIL values can be found in {h0_save_path} and {h1_save_path}. Filtrations can be found in {filt_save_path}')

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


            




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GIN_MPML', choices=['GIN', 'GCN', 'GAT', 'SimpleNNBaseline', 'GIN_MPML', 'GCN_MPML', 'GAT_MPML'])
    parser.add_argument('--dataset', type=str, default='EGFR')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--pth', type=str, default="experiment_logs/2023-11-07_11:07:35")
    args = parser.parse_args()
    
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
    'concat_fp': None,
    }
    __training_cfg = {
        'lr': 1e-2,
        'lr_drop_fact': 0.5,
        'num_epochs': 15,
        'epoch_step': 10,
        'batch_size': 1,
        'weight_decay': 1e-6,
        'validation_ratio': 0.1,
        }

    __exp_cfg_meta = {
            'dataset_name': args.dataset,
            'training': __training_cfg,
            'model': __model_cfg_meta,
            'tag': 'GIN_MPML'
            }
    candidate_logs = find_candidate_logs(__exp_cfg_meta)
    probable_candidate = candidate_logs[0]
    
    test_model(args, probable_candidate, verbose=True, num_folds=args.num_folds)
    
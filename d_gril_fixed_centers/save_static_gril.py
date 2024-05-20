import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import gril.gril as gril
import os
import argparse
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
from scipy.sparse.linalg import lsqr, cg, eigsh
from scipy.linalg import eigh
from data import EGFRDatasetGeometric
from tqdm import tqdm
import pickle



def pre_process_edges(edge_index):
    e = edge_index.permute(1, 0)
    e = e.sort(1)
    e = e[0].tolist()
    e = set([tuple(ee) for ee in e])
    return torch.tensor([ee for ee in e], dtype=torch.long, device=edge_index.device)

def get_hks(L, K, ts):
    """
    From https://github.com/ctralie/pyhks/blob/master/hks.py
    ----------
    L : Graph Laplacian

    K : int
        Number of eigenvalues/eigenvectors to use
    ts : ndarray (T)
        The time scales at which to compute the HKS
    
    Returns
    -------
    hks : ndarray (N, T)
        A array of the heat kernel signatures at each of N points
        at T time intervals
    """
    (eigvalues, eigvectors) = eigh(L)
    res = (eigvectors[:, :, None]**2)*np.exp(-eigvalues[None, :, None] * ts.flatten()[None, None, :])
    return np.sum(res, 1)

# def get_hks_filtration(x, edge_index, nn_k=6):
#     g = nx.Graph()
#     g.add_nodes_from(range(x.shape[0]))
#     edges = pre_process_edges(edge_index)
#     edges_list = edges.tolist()
#     g.add_edges_from((e[0], e[1]) for e in edges_list)
#     frc = FormanRicci(g)
#     frc.compute_ricci_curvature()
#     graph_laplacian = nx.normalized_laplacian_matrix(g).toarray().astype(float)
#     hks = get_hks(graph_laplacian, x.shape[0], ts=np.array([1, 10]))
#     f_v_x = hks[:, -1]
#     f = []

#     for n in range(x.shape[0]):
#         v_curv = frc.G.nodes[n]['formanCurvature']
#         f.append([f_v_x[n],v_curv])
#         # print(f"Node: {n} f_x: {f_v_x[n]} f_y: {v_curv}")
    
#     for e in edges_list:
#         e_x = max([f[e[0]][0], f[e[1]][0]])
#         e_curv = frc.G[e[0]][e[1]]["formanCurvature"]
#         e_y = max([f[e[0]][1], f[e[1]][1], e_curv])
#         f.append([e_x, e_y])
#         # print(f"Edge: ({e[0]}, {e[1]}) f_x: {e_x} f_y: {e_y} e_curv: {e_curv}")
    
#     filt = torch.tensor(f, device=x.device)
#     return filt, edges

def get_hks_rc_bifiltration(num_nodes, edge_index, nn_k=6):
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    edges = pre_process_edges(edge_index)
    edges_list = edges.tolist()
    g.add_edges_from((e[0], e[1]) for e in edges_list)
    frc = FormanRicci(g)
    frc.compute_ricci_curvature()
    graph_laplacian = nx.normalized_laplacian_matrix(g).toarray().astype(float)
    hks = get_hks(graph_laplacian, num_nodes, ts=np.array([1, 10]))
    f_v_x = hks[:, -1]
    f = []

    for n in range(num_nodes):
        v_curv = frc.G.nodes[n]['formanCurvature']
        f.append([f_v_x[n],v_curv])
        # print(f"Node: {n} f_x: {f_v_x[n]} f_y: {v_curv}")
    f = np.array(f)
    f = (f - f.min(axis=0)) / (f.max(0) - f.min(0) + 1e-4)
    f_e = []
    for e in edges_list:
        e_x = max([f[e[0], 0], f[e[1], 0]]) 
        e_curv = frc.G[e[0]][e[1]]["formanCurvature"]
        e_y = max([f[e[0], 1], f[e[1], 1], e_curv])
        f_e.append([e_x, e_y])
        # print(f"Edge: ({e[0]}, {e[1]}) f_x: {e_x} f_y: {e_y} e_curv: {e_curv}")
    f_e = np.array(f_e)
    # f_e = (f_e - f_e.min(axis=0)) / (f_e.max(0) - f_e.min(0))
    f_e = f_e + 1e-4
    try:
        f = np.row_stack((f, f_e))
    except:
        f = f 
    filt = torch.tensor(f, device=edge_index.device)
    
    return filt, edges

def get_filtration(x, edge_index, nn_k=6):
    edges = pre_process_edges(edge_index)
    d_xx = torch.cdist(x, x)
    nn_k = min([nn_k, x.shape[0]])
    d_xx = d_xx.topk(nn_k, 1, largest=False).values
    # d_xy = -(d_xx * d_xx)
    d_xx = -d_xx[:, 1:]
    d_xx = torch.exp(d_xx).sum(1) / nn_k 
    d_xx = 1 - d_xx
    # d_xx = d_xx * 3.;
    # d_xx = d_xx / d_xx.max()
    e_val = d_xx.unsqueeze(0).expand((edges.size(0), -1))
    e_val = e_val.gather(1, edges)
    e_val = e_val.max(1)[0]

    # tri_val = d_xx.unsqueeze(0).expand((tri.size(0), -1))
    # tri_val = tri_val.gather(1, tri)
    # tri_val = tri_val.max(1)[0]

    e_val_x = x[edges[:, 0]]
    e_val_y = x[edges[:, 1]]
    e_val_y = torch.norm(e_val_x - e_val_y, dim=1)
    e_val_y = 1 - torch.exp(-e_val_y)

    # tri_val_2 = e_val_y.unsqueeze(0).expand((tri_converted.size(0), -1))
    # tri_val_2 = tri_val_2.gather(1, tri_converted)
    # tri_val_2 = tri_val_2.max(1)[0]

    f_v = torch.cat([d_xx.view((-1, 1)), torch.zeros((d_xx.size(0), 1))], dim=1)
    e_val = torch.cat([e_val.view((-1, 1)), e_val_y.view((-1, 1))], dim=1)
    # tri_val = torch.cat([tri_val.view((-1, 1)), tri_val_2.view((-1, 1))], dim=1) + 0.02

    # filt = torch.cat([f_v, e_val, tri_val], dim=0)
    filt = torch.cat([f_v, e_val], dim=0)
    return filt, edges

def get_simplices(num_vertices, edges, triangles=None):
    simp = [[i] for i in range(num_vertices)]

    for e in edges:
        e_ = sorted([e[0].item(), e[1].item()])
        simp.append(e_)
    if triangles is not None:
        for f in triangles:
            f_ = sorted([f[0].item(), f[1].item(), f[2].item()])
            simp.append(f_)
    return simp


class MultiPersLandscapeValLayer(nn.Module):
    def __init__(self, res, hom_rank, step=10, l=2):
        super().__init__()
        self.res = res
        self.step = step
        self.l = l
        # self.grid_resolution = grid_resolution
        self.sample_pts = self.sample_grid()
        self.hom_rank = hom_rank
        self.filt_layer = get_filtration
        self.mpl = gril.MultiPers(hom_rank=hom_rank, l=l, res=res, ranks=list(range(1, 6)))
        self.mpl.set_max_jobs(40)

    def sample_grid(self):
        pts = []
        num_division = int(1.0 / self.res)
        for j in range(0, num_division, self.step):
            for i in range(0, num_division, self.step):
                pts.append((i, j))
        return pts

    def forward(self, x, edge_index):
        # f_ = (f - f.min(dim=0).values) / (f.max(dim=0).values - f.min(dim=0).values)
        # f_ = Snap.apply(f)
        num_vertices = x.shape[0]
        f, e = self.filt_layer(x, edge_index)
        simplices = get_simplices(num_vertices, e)
        pers_inp = [(f, simplices)]
        bars = self.mpl.compute_landscape(self.sample_pts, pers_inp)
        return bars


class MultiPersLandscapeValLayer_no_node_feats(nn.Module):
    def __init__(self, res, hom_rank, step=10, l=2):
        super().__init__()
        self.res = res
        self.step = step
        self.l = l
        # self.grid_resolution = grid_resolution
        self.sample_pts = self.sample_grid()
        self.hom_rank = hom_rank
        self.filt_layer = get_hks_rc_bifiltration
        self.mpl = gril.MultiPers(hom_rank=hom_rank, l=l, res=res, step=step, ranks=list(range(1, 4)))
        self.mpl.set_max_jobs(40)

    def sample_grid(self):
        pts = []
        num_division = int(1.0 / self.res)
        for j in range(0, num_division, self.step):
            for i in range(0, num_division, self.step):
                pts.append((i, j))
        return pts

    def forward(self, num_nodes, edge_index):
        
        # f_ = Snap.apply(f)
        f, e = self.filt_layer(num_nodes, edge_index)
        # f = (f - f.min(dim=0).values) / (f.max(dim=0).values - f.min(dim=0).values)
        simplices = get_simplices(num_nodes, e)
        pers_inp = [(f, simplices)]
        bars = self.mpl.compute_landscape(self.sample_pts, pers_inp)
        return bars
    
def save_single_landscape(data, batch_i, landscape_dir):
    batch = data[batch_i]
    out_file_1 = os.path.join(landscape_dir, f"graph_{batch_i}.pt")
    layer1_0 = MultiPersLandscapeValLayer_no_node_feats(res=0.01, hom_rank=0, step=2, l=2)
    if not os.path.exists(out_file_1):
        layer1_0.mpl.set_hom_rank(0)
        out1_0 = layer1_0(batch.num_nodes, batch.edge_index)
        lmbda1_0 = out1_0[0]
        layer1_0.mpl.set_hom_rank(1)
        out1_1 = layer1_0(batch.num_nodes, batch.edge_index)
        lmbda1_1 = out1_1[0]
        torch.save({"H0": lmbda1_0, "H1": lmbda1_1}, out_file_1)
        # print(f"Saved to {out_file_1}")

def save_landscapes(data, landscape_dir):
    gril_h0 = []
    gril_h1 = []
    layer1_0 = MultiPersLandscapeValLayer_no_node_feats(res=0.01, hom_rank=0, step=2, l=2)
    for batch_i in tqdm(range(len(data))):
        batch = data[batch_i]
        layer1_0.mpl.set_hom_rank(0)
        out1_0 = layer1_0(batch.num_nodes, batch.edge_index)
        lmbda1_0 = out1_0[0].reshape(50, 50, 3)
        layer1_0.mpl.set_hom_rank(1)
        out1_1 = layer1_0(batch.num_nodes, batch.edge_index)
        lmbda1_1 = out1_1[0].reshape(50, 50, 3)
        gril_h0.append(lmbda1_0.detach().cpu().numpy())
        gril_h1.append(lmbda1_1.detach().cpu().numpy())
    
    with open(os.path.join(landscape_dir, "gril_50_HKS-RC-0.pkl"), "wb") as f:
        pickle.dump(gril_h0, f)
    with open(os.path.join(landscape_dir, "gril_50_HKS-RC-1.pkl"), "wb") as f:
        pickle.dump(gril_h1, f)
    print(f"Saved to {landscape_dir}")

    
    


if __name__ == '__main__':
    TUDataset_names = ['NCI1', 
                       'MUTAG', 
                       'PROTEINS', 
                       'DHFR', 
                       'COX2', 
                       'DD', 
                       'IMDB-BINARY', 
                       'REDDIT-BINARY', 
                       'IMDB-MULTI', 
                       'REDDIT-MULTI-5K',
                       'EGFR']
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PROTEINS", type=str, choices=TUDataset_names)
    parser.add_argument("--sample", default=0, type=int)

    args = parser.parse_args()
    ds_name = args.dataset
    if ds_name == "EGFR":
        data = EGFRDatasetGeometric(root='./data/', name=ds_name)
    else:
        data = TUDataset(root='./data/', name=ds_name)
    # dl_train = DataLoader(data, batch_size=1, shuffle=False)
    print(len(data))
    

    landscape_dir_1 = f"graph_landscapes/{ds_name}"

    os.makedirs(landscape_dir_1, exist_ok=True)
    # batch_i = args.sample
    # for batch_i in tqdm(range(len(data))):
    #     save_single_landscape(data, batch_i, landscape_dir_1)
        # print(f"Saved {batch_i + 1} of {len(data)}")
    
    save_landscapes(data, landscape_dir_1)
   
    
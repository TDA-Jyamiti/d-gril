import os
import torch
import numpy as np
import torch_geometric
import torch_geometric.data

from torch_geometric.datasets import TUDataset
from collections import Counter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pickle as pkl
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
from scipy.linalg import eigh
import pandas as pd

from ogb.graphproppred import PygGraphPropPredDataset

from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

import datamol as dm
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
from molfeat.trans.graph import PYGGraphTransformer
from molfeat.calc.atom import AtomCalculator
from torch_geometric.data import InMemoryDataset, download_url

POWERFUL_GNN_DATASET_NAMES =  ["PTC_PGNN"]
TU_DORTMUND_DATASET_NAMES = [
    "NCI1", "PTC_MR", 'PTC_FM', 
    'PTC_FR', 'PTC_MM', "PROTEINS", 
    "REDDIT-BINARY", "REDDIT-MULTI-5K", 
    "ENZYMES", "DD", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "COLLAB", "COX2", "DHFR"
]

MOL_DATASET_NAMES = ["EGFR", 
                     "ERRB2",
                     "CHEMBL279",
                     "CHEMBL203",
                     "CHEMBL4005",
                     "CHEMBL2971",
                     "CHEMBL2835",
                     "CHEMBL2842",
                     "CHEMBL260",
                     "CHEMBL2147",
                     "CHEMBL5145",
                     "CHEMBL1163125",
                     "CHEMBL3717",
                     "CHEMBL2599",
                     "CHEMBL267",
                     "CHEMBL2148",
                     "CHEMBL4282",
                     "CHEMBL4040",
                     "CHEMBL3130",
                     "CHEMBL2815",
                     "CHEMBL4722",
                     "CHEMBL262",
                     ]

OGB_DATA_NAMES = ["ogbg-molhiv", "ogbg-molpcba", 
                  "ogbg-molbbbp", "ogbg-molclintox", 
                  "ogbg-molmuv", "ogbg-moltox21", 
                  "ogbg-moltoxcast", "ogbg-molhiv", 
                  "ogbg-molpcba", "ogbg-molbbbp", 
                  "ogbg-molclintox", "ogbg-molmuv", 
                  "ogbg-moltox21", "ogbg-moltoxcast"]


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
    f = np.row_stack((f, f_e))
    filt = torch.tensor(f, device=edge_index.device)
    
    return filt, edges

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        assert isinstance(X, list)
        self.data = X
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for i in range(len(self.data)):
            yield self.data[i]
    
    
class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        assert isinstance(indices, (list, tuple))
        self.ds = dataset
        self.indices = tuple(indices)
        
        assert len(indices) <= len(dataset)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.ds[self.indices[idx]]


def load_powerfull_gnn_dataset_PTC():
    has_node_features = False
    
    dataset_name = "PTC"
    path = "/home/pma/chofer/repositories/powerful-gnns/dataset/{}/{}.txt".format(dataset_name, dataset_name)

    with open(path, 'r') as f:
        num_graphs = int(f.readline().strip())

        data = []

        graph_label_map = {}
        node_lab_map = {}

        for i in range(num_graphs):
            row = f.readline().strip().split()
            num_nodes, graph_label = [int(w) for w in row]

            if graph_label not in graph_label_map:
                graph_label_map[graph_label] = len(graph_label_map)

            graph_label = graph_label_map[graph_label]


            nodes = []
            node_labs = []
            edges = []
            node_features = []

            for node_id in range(num_nodes):
                nodes.append(node_id)

                row = f.readline().strip().split()

                node_lab = int(row[0])

                if node_lab not in node_lab_map:
                    node_lab_map[node_lab] = len(node_lab_map)

                node_labs.append(node_lab_map[node_lab])

                num_neighbors = int(row[1])            
                neighbors = [int(i) for i in row[2:num_neighbors+2]]
                assert num_neighbors == len(neighbors)

                edges += [(node_id, neighbor_id) for neighbor_id in neighbors]

                if has_node_features:                
                    node_features = [float(i) for i in row[(2 + num_neighbors):]]
                    assert len(node_features) != 0           

            # x = torch.tensor(node_features) if has_node_features else None
            x = torch.tensor(node_labs, dtype=torch.long)

            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_index = edge_index.permute(1, 0)
            tmp = edge_index.index_select(0, torch.tensor([1, 0]))
            edge_index = torch.cat([edge_index, tmp], dim=1)

            y = torch.tensor([graph_label])
            
            d = torch_geometric.data.Data(
                    x=x, 
                    edge_index=edge_index, 
                    y = y
                )

            d.num_nodes = num_nodes
            
            data.append(d)      

    max_node_lab = max([d.x.max().item() for d in data]) + 1
    eye = torch.eye(max_node_lab, dtype=torch.long)
    for d in data:
        node_lab = eye.index_select(0, d.x)
        d.x = node_lab

    ds = SimpleDataset(data)
    ds.name = dataset_name

    return ds


def get_boundary_info(g):
    
    e = g.edge_index.permute(1, 0).sort(1)[0].tolist()
    e = set([tuple(ee) for ee in e])
    return torch.tensor([ee for ee in e], dtype=torch.long)  


def enhance_TUDataset(ds, **kwargs):
      
    X = []
    targets = []
    
    max_degree_by_graph = []
    num_nodes = []
    num_edges = []
    # ls_h0 = kwargs['ls_h0']
    # ls_h1 = kwargs['ls_h1']
    
    for i, d in enumerate(ds):
        
        targets.append(d.y.item())
        if hasattr(d, 'x') and d.x is None:
            d.x = torch.empty((d.num_nodes, 0)) 
        
        # boundary_info = get_boundary_info(d)
        # d.boundary_info = boundary_info
        
        num_nodes.append(d.num_nodes)
        num_edges.append(d.edge_index.size(1) // 2)
        
        degree = torch.zeros(d.num_nodes, dtype=torch.long)

        for k, v in Counter(d.edge_index.flatten().tolist()).items():
            degree[k] = v
        degree = torch.div(degree, 2, rounding_mode='floor')
        max_degree_by_graph.append(degree.max().item())
        # h0 = ls_h0[i].sum(-1).flatten()
        # h0 = torch.tensor(h0, dtype=torch.float)
        # h1 = ls_h1[i].sum(-1).flatten()
        # h1 = torch.tensor(h1, dtype=torch.float)
        # ls = torch.cat([h0, h1]).reshape((1, -1))
        d.node_deg = degree
        # d.ls = ls
        # d.h0 = h0
        X.append(d)
        
    max_node_deg =  max(max_degree_by_graph)
    
    num_node_lab = None
   
    if (hasattr(X[0], 'x') and X[0].x.nelement() !=0):

        all_node_lab = []
        for d in X:
            assert d.x.sum() == d.x.size(0) # really one hot encoded?
            node_lab = d.x.argmax(1).tolist()
            d.node_lab = node_lab
            all_node_lab += node_lab

        all_node_lab = set(all_node_lab)  
        num_node_lab = len(all_node_lab)    
        label_map = {k: i for i, k in enumerate(sorted(all_node_lab))}

        for d in X:
            d.node_lab = [label_map[f] for f in d.node_lab]
            d.node_lab = torch.tensor(d.node_lab, dtype=torch.long) 
    else:
        for d in X:
            d.node_lab = None       
                
    new_ds =  SimpleDataset(X)
    
    new_ds.max_node_deg = max_node_deg
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    new_ds.num_classes   = len(set(targets))
    new_ds.num_node_lab = num_node_lab
    
    return new_ds

def enhance_EGFRDataset(ds, **kwargs):
    X = []
    targets = []
    
    max_degree_by_graph = []
    num_nodes = []
    num_edges = []
    # ls_h0 = kwargs['ls_h0']
    # ls_h1 = kwargs['ls_h1']
    
    for i, d in enumerate(ds):
        
        targets.append(d.y.item())
        if hasattr(d, 'x') and d.x is None:
            d.x = torch.empty((d.num_nodes, 0)) 
        
        # boundary_info = get_boundary_info(d)
        # d.boundary_info = boundary_info
        
        num_nodes.append(d.num_nodes)
        num_edges.append(d.edge_index.size(1) // 2)
        
        degree = torch.zeros(d.num_nodes, dtype=torch.long)

        for k, v in Counter(d.edge_index.flatten().tolist()).items():
            degree[k] = v
        degree = torch.div(degree, 2, rounding_mode='floor')
        max_degree_by_graph.append(degree.max().item())
        # h0 = ls_h0[i].sum(-1).flatten()
        # h0 = torch.tensor(h0, dtype=torch.float)
        # h1 = ls_h1[i].sum(-1).flatten()
        # h1 = torch.tensor(h1, dtype=torch.float)
        # ls = torch.cat([h0, h1]).reshape((1, -1))
        d.node_deg = degree
        # d.ls = ls
        # d.h0 = h0
        X.append(d)
        
    max_node_deg =  max(max_degree_by_graph)
    
    num_node_lab = None
   
    if (hasattr(X[0], 'x') and X[0].x.nelement() !=0):

        all_node_lab = []
        for d in X:
            assert d.node_labels.sum() == d.node_labels.size(0) # really one hot encoded?
            node_lab = d.node_labels.argmax(1).tolist()
            d.node_lab = node_lab
            all_node_lab += node_lab

        all_node_lab = set(all_node_lab)  
        num_node_lab = len(all_node_lab)    
        label_map = {k: i for i, k in enumerate(sorted(all_node_lab))}

        for d in X:
            d.node_lab = [label_map[f] for f in d.node_lab]
            d.node_lab = torch.tensor(d.node_lab, dtype=torch.long) 
    else:
        for d in X:
            d.node_lab = None       
                
    new_ds =  SimpleDataset(X)
    
    new_ds.max_node_deg = max_node_deg
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    new_ds.num_classes   = len(set(targets))
    new_ds.num_node_lab = num_node_lab
    
    return new_ds


def enhance_TUDataset_with_filtration(ds, **kwargs):
      
    X = []
    targets = []
    
    max_degree_by_graph = []
    num_nodes = []
    num_edges = []
    ls_h0 = kwargs['ls_h0']
    ls_h1 = kwargs['ls_h1']
    
    for i, d in enumerate(ds):
        
        targets.append(d.y.item())
        if hasattr(d, 'x') and d.x is None:
            d.x = torch.empty((d.num_nodes, 0)) 
        
        # boundary_info = get_boundary_info(d)
        # d.boundary_info = boundary_info
        
        num_nodes.append(d.num_nodes)
        num_edges.append(d.edge_index.size(1) // 2)
        
        degree = torch.zeros(d.num_nodes, dtype=torch.long)

        for k, v in Counter(d.edge_index.flatten().tolist()).items():
            degree[k] = v
        degree = torch.div(degree, 2, rounding_mode='floor')
        max_degree_by_graph.append(degree.max().item())
        # h0 = ls_h0[i].sum(-1).flatten()
        # h0 = torch.tensor(h0, dtype=torch.float)
        # h1 = ls_h1[i].sum(-1).flatten()
        # h1 = torch.tensor(h1, dtype=torch.float)
        # ls = torch.cat([h0, h1]).reshape((1, -1))
        d.node_deg = degree
        # filt, _ = get_hks_rc_bifiltration(d.num_nodes, d.edge_index)
        # d.filt = torch.sigmoid(filt)
        # d.ls = ls
        # d.h1 = h1
        X.append(d)
        
    max_node_deg =  max(max_degree_by_graph)
    
    num_node_lab = None
   
    if (hasattr(X[0], 'x') and X[0].x.nelement() !=0):

        all_node_lab = []
        for d in X:
            assert d.x.sum() == d.x.size(0) # really one hot encoded?
            node_lab = d.x.argmax(1).tolist()
            d.node_lab = node_lab
            all_node_lab += node_lab

        all_node_lab = set(all_node_lab)  
        num_node_lab = len(all_node_lab)    
        label_map = {k: i for i, k in enumerate(sorted(all_node_lab))}

        for d in X:
            d.node_lab = [label_map[f] for f in d.node_lab]
            d.node_lab = torch.tensor(d.node_lab, dtype=torch.long) 
    else:
        for d in X:
            d.node_lab = None       
                
    new_ds =  SimpleDataset(X)
    
    new_ds.max_node_deg = max_node_deg
    new_ds.avg_num_nodes = np.mean(num_nodes)
    new_ds.avg_num_edges = np.mean(num_edges)
    new_ds.num_classes   = len(set(targets))
    new_ds.num_node_lab = num_node_lab
    
    return new_ds


def dataset_factory(dataset_name, verbose=True):
    if dataset_name in TU_DORTMUND_DATASET_NAMES:

        path = './data/{}/'.format(dataset_name)
        dataset = TUDataset(path, name=dataset_name)
        dataset = enhance_TUDataset(dataset)
        # ls_h0_pth = f"./graph_landscapes/{dataset_name}/{dataset_name}_mpml_HKS-RC-0.pkl"
        # ls_h1_pth = f"./graph_landscapes/{dataset_name}/{dataset_name}_mpml_HKS-RC-1.pkl"
        # ls_h0 = pkl.load(open(ls_h0_pth, "rb"))
        # ls_h1 = pkl.load(open(ls_h1_pth, "rb"))


    elif dataset_name in POWERFUL_GNN_DATASET_NAMES:
        if dataset_name == "PTC_PGNN":
            dataset = load_powerfull_gnn_dataset_PTC()
    elif dataset_name in MOL_DATASET_NAMES:
        
        dataset = EGFRDatasetGeometric(root="./data", name=dataset_name)
        dataset = enhance_EGFRDataset(dataset)
    
    elif dataset_name in OGB_DATA_NAMES:
        
        dataset = PygGraphPropPredDataset(name=dataset_name, root='./data')

    else:
        raise ValueError("dataset_name not in {}".format(TU_DORTMUND_DATASET_NAMES + POWERFUL_GNN_DATASET_NAMES))
    # ds_name = dataset_name
    # dataset = enhance_TUDataset(dataset, ls_h0=ls_h0, ls_h1=ls_h1)
    # dataset = enhance_TUDataset_with_filtration(dataset)
   
    

    if verbose:
        print("# Dataset: ", dataset_name, flush=True)    
        print('# num samples: ', len(dataset), flush=True)
        print('# num classes: ', dataset.num_classes, flush=True)
        print('#')
        # print('# max node degree: ', dataset.max_node_deg, flush=True)
        # print('# num node labels: ', dataset.num_node_lab, flush=True)
        print('#')
        # print('# avg number of nodes: ', dataset.avg_num_nodes, flush=True)
        # print('# avg number of edges: ', dataset.avg_num_edges, flush=True)

    
    return dataset


def train_test_val_split(
    dataset, 
    seed=42, 
    n_splits=10, 
    verbose=True, 
    validation_ratio=0.0):

    skf = StratifiedKFold(
        n_splits=n_splits, 
        shuffle = True, 
        random_state = seed, 
    )

    targets = [x.y.item() for x in dataset]
    split_idx = list(skf.split(np.zeros(len(dataset)), targets))

    if verbose:
        print('# num splits: ', len(split_idx))
        print('# validation ratio: ', validation_ratio)

    split_ds = []
    split_i = []
    for train_i, test_i in split_idx:
        not_test_i, test_i = train_i.tolist(), test_i.tolist()

        if validation_ratio == 0.0:
            validation_i = []
            train_i = not_test_i

        else:
            skf = StratifiedShuffleSplit(
                n_splits=1, 
                random_state = seed, 
                test_size=validation_ratio
            )

            targets = [dataset[i].y.item() for i in not_test_i]
            train_i, validation_i = list(skf.split(np.zeros(len(not_test_i)), targets))[0]
            train_i, validation_i = train_i.tolist(), validation_i.tolist()  

            # We need the indices w.r.t. the original dataset 
            # not w.r.t. the current train fold ... 
            train_i = [not_test_i[j] for j in train_i]
            validation_i = [not_test_i[j] for j in validation_i]
            
        assert len(set(train_i).intersection(set(validation_i))) == 0
        
        train = Subset(dataset, train_i)
        test = Subset(dataset, test_i)
        validation = Subset(dataset, validation_i)

        assert sum([len(train), len(test), len(validation)]) == len(dataset)

        split_ds.append((train, test, validation))
        split_i.append((train_i, test_i, validation_i))

    return split_ds, split_i


    
class EGFRDatasetGeometric(InMemoryDataset):
    def __init__(self, root, name, threshold=6.3, transform=None, pre_transform=None, pre_filter=None):
        self.threshold = threshold
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')
    
    @property
    def labels(self):
        return self.data.y.numpy()
    
    @property
    def raw_file_names(self):
        return [f'{self.name}.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def smiles_to_fp(self, method, smiles, n_bits=2048):
        """
        Encode a molecule from a SMILES string into a fingerprint.

        Parameters
        ----------
        smiles : str
            The SMILES string defining the molecule.

        method : str
            The type of fingerprint to use. Default is MACCS keys.

        n_bits : int
            The length of the fingerprint.

        Returns
        -------
        array
            The fingerprint array.

        """

        # convert smiles to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        if method == "maccs":
            return np.array(MACCSkeys.GenMACCSKeys(mol))
        if method == "morgan2":
            fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
            return np.array(fpg.GetFingerprint(mol))
        if method == "morgan3":
            fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
            return np.array(fpg.GetFingerprint(mol))
        if method == "ecfp":
            calc = FPCalculator("ecfp")
            fp = calc(smiles)
            return np.array(fp)

    
    def process(self):
        # Read data into huge `Data` list.
        chembl_df = pd.read_csv(os.path.join(self.raw_paths[0]))
        chembl_df = chembl_df[["molecule_chembl_id", "smiles", "pIC50"]]
        # chembl_df["active"] = np.zeros(len(chembl_df))
        chembl_df = chembl_df.assign(active = np.zeros(len(chembl_df)))
        # Mark every molecule as active with an pIC50 of > 8, 0 otherwise
        chembl_df.loc[chembl_df[chembl_df.pIC50 >= self.threshold].index, "active"] = 1
        mol_transf = PYGGraphTransformer(atom_featurizer=AtomCalculator())
        data_list = mol_transf(chembl_df["smiles"].values)
        ac = AtomCalculator(concat=False)
        for i, data in enumerate(data_list):
            smiles = chembl_df.loc[i, "smiles"]
            data.y = torch.tensor([chembl_df.loc[i, "active"]], dtype=torch.long)
            node_labels = ac(smiles)["atom_one_hot"]
            data.node_labels = torch.tensor(node_labels, dtype=torch.long)
            maccs = self.smiles_to_fp('maccs', smiles)
            ecfp = self.smiles_to_fp('ecfp', smiles)
            morgan2 = self.smiles_to_fp('morgan2', smiles)
            morgan3 = self.smiles_to_fp('morgan3', smiles)
            data.maccs = torch.from_numpy(maccs)
            data.ecfp = torch.from_numpy(ecfp)
            data.morgan2 = torch.from_numpy(morgan2)
            data.morgan3 = torch.from_numpy(morgan3)

        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    dataset = EGFRDatasetGeometric(root='./data', name='EGFR')
    data = dataset[0]
    print(data)

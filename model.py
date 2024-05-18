import torch
import numpy as np
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geonn
import functools
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool, global_sort_pool
from torch.autograd import Function
import gril.gril as gril
from data import pre_process_edges

EPS = 1e-4

def gin_mlp_factory(gin_mlp_type: str, dim_in: int, dim_out: int):
    if gin_mlp_type == 'lin':
        return nn.Linear(dim_in, dim_out)

    elif gin_mlp_type == 'lin_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in), 
            nn.LeakyReLU(), 
            nn.Linear(dim_in, dim_out)
        )

    elif gin_mlp_type == 'lin_bn_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in), 
            nn.BatchNorm1d(dim_in), 
            nn.LeakyReLU(), 
            nn.Linear(dim_in, dim_out)
        )
    else: 
        raise ValueError("Unknown gin_mlp_type!")


def gnn_mlp_factory(gnn_type: str, gin_mlp_type: str, dim_in: int, dim_out: int):
    if gnn_type == 'GCN':
        conv = GCNConv
    elif gnn_type == 'GAT':
        conv = GATConv

    if gin_mlp_type == 'lin':
        return conv(dim_in, dim_out)

    elif gin_mlp_type == 'lin_lrelu_lin':
        return nn.Sequential(
            conv(dim_in, dim_in), 
            nn.LeakyReLU(), 
            conv(dim_in, dim_out)
        )

    elif gin_mlp_type == 'lin_bn_lrelu_lin':
        return nn.Sequential(
            conv(dim_in, dim_in), 
            nn.BatchNorm1d(dim_in), 
            nn.LeakyReLU(), 
            conv(dim_in, dim_out)
        )
    else: 
        raise ValueError("Unknown gin_mlp_type!")
    
def compute_lub(f_v: torch.Tensor, edges:torch.Tensor):
    if edges.numel() == 0:
        return f_v
    f_v_x = f_v[:, 0]
    f_v_y = f_v[:, 1]
    
    e_x = f_v_x.unsqueeze(0).expand((edges.size(0), -1))
    e_x = e_x.gather(1, edges)
    e_x = e_x.max(1)[0]

    e_y = f_v_y.unsqueeze(0).expand((edges.size(0), -1))
    e_y = e_y.gather(1, edges)
    e_y = e_y.max(1)[0]
    f_e = (torch.column_stack([e_x, e_y]) + EPS)
    f = torch.row_stack([f_v, f_e])
    return f

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

def ClassifierHead(
    dataset, 
    dim_in: int=None, 
    hidden_dim: int=None, 
    drop_out: float=None):

    assert (0.0 <= drop_out) and (drop_out < 1.0)
    assert dim_in is not None
    assert drop_out is not None
    assert hidden_dim is not None

    tmp = [
        nn.Linear(dim_in, hidden_dim), 
        nn.LeakyReLU(),             
    ]

    if drop_out > 0:
        tmp += [nn.Dropout(p=drop_out)]

    tmp += [nn.Linear(hidden_dim, dataset.num_classes)]

    return nn.Sequential(*tmp)


class DegreeOnlyFiltration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch):
        tmp = []
        for i, j in zip(batch.sample_pos[:-1], batch.sample_pos[1:]):
            max_deg = batch.node_deg[i:j].max()
            
            t = torch.ones(j - i, dtype=torch.float, device=batch.node_deg.device)
            t = t * max_deg 
            tmp.append(t)

        max_deg = torch.cat(tmp, dim=0)
                    
        normalized_node_deg = batch.node_deg.float() / max_deg

        return normalized_node_deg 


class OneHotEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        eye = torch.eye(dim, dtype=torch.float)

        self.register_buffer('eye', eye)

    def forward(self, batch):
        assert batch.dtype == torch.long 

        return self.eye.index_select(0, batch)
    
    @property
    def dim(self):
        return self.eye.size(1)


class UniformativeDummyEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        b = torch.ones(1, dim, dtype=torch.float)
        self.register_buffer('ones', b)

    def forward(self, batch):
        assert batch.dtype == torch.long 
        return self.ones.expand(batch.size(0), -1)
    
    @property
    def dim(self):
        return self.ones.size(1)


class GIN(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        use_node_features: bool = None,
        pooling_strategy: str=None,
        concat_fp: str = None,
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        elif use_node_features:
            try:
                assert 'input_features_dim' in kwargs and kwargs['input_features_dim'] is not None
            except:
                raise Exception("If using input features input_feature_dim must be set")
            self.embed_deg = None

        self.use_node_features = use_node_features
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0
        dim_input += kwargs['input_features_dim'] if use_node_features else 0 
        assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)    
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))      
        
        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
            #nn.Conv1d(
            #    in_channels=gin_dimension, 
            #    out_channels=gin_dimension, 
            #    kernel_size=self.k
            #)
        else:
            raise ValueError

        # self.cls = ClassifierHead(
        #     dataset, 
        #     dim_in=gin_dimension, 
        #     hidden_dim=cls_hidden_dimension, 
        #     drop_out=drop_out
        # )
        cls_in_dim = sum(dims)
        self.concat_fp = concat_fp
        if self.concat_fp == 'maccs':
            cls_in_dim += 167
        elif self.concat_fp in ['ecfp','morgan2', 'morgan3']:
            cls_in_dim += 2048
        
        self.cls = ClassifierHead(dataset, 
                                  dim_in=cls_in_dim, 
                                  hidden_dim=cls_hidden_dimension, 
                                  drop_out=drop_out)         
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        gpu_dev = batch.edge_index.device
        fp = batch[self.concat_fp] if self.concat_fp else torch.empty([], device=gpu_dev)
        if not self.use_node_features:
            tmp = [e(x) for e, x in 
                zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
                if e is not None] 
            
            tmp = torch.cat(tmp, dim=1)
        else:
            tmp = batch.x

        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)

        x = torch.cat(z, dim=1)
        # x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            #x = x.view(x.size(0), self.gin_dimension * self.k)
            x = self.sort_pool_nn(x)
            x = x.squeeze()

        fp = batch[self.concat_fp] if self.concat_fp else torch.empty([], device=batch.edge_index.device)
        if self.concat_fp is not None:
            x = torch.column_stack([x, fp.view(1, -1)])
        
        x = self.act(x)
        
        if not self.use_as_feature_extractor:
            x = self.cls(x)
        
        return x


class SimpleNNBaseline(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        set_node_degree_uninformative: bool=None,
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        pooling_strategy: str=None,    
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        else:
            self.embed_deg = None
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0 
        assert dim_input > 0
        
        self.mlp = gin_mlp_factory(gin_mlp_type, dim_input, dim) 
        
        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
        else:
            raise ValueError

        self.cls = ClassifierHead(
            dataset, 
            dim_in=gin_dimension, 
            hidden_dim=cls_hidden_dimension, 
            drop_out=drop_out
        )         
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab

        edge_index = batch.edge_index
        
        tmp = [e(x) for e, x in 
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None] 
        
        x = torch.cat(tmp, dim=1)    
        
        x = self.mlp(x) 
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            x = self.sort_pool_nn(x)
            x = x.squeeze()

        if not self.use_as_feature_extractor:
            x = self.cls(x)
        return x


class GIN_MPML(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        pooling_strategy: str=None,    
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        else:
            self.embed_deg = None
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0 
        assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)    
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))   
        
        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
            #nn.Conv1d(
            #    in_channels=gin_dimension, 
            #    out_channels=gin_dimension, 
            #    kernel_size=self.k
            #)
        else:
            raise ValueError
        
        self.lin_h = nn.Linear(kwargs['ls_dim'], cls_hidden_dimension)
        # self.lin_h1 = nn.Linear(kwargs['ls_dim'],cls_hidden_dimension)

        self.cls = ClassifierHead(
            dataset, 
            dim_in=2*gin_dimension, 
            hidden_dim=cls_hidden_dimension, 
            drop_out=drop_out
        )         
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        
        tmp = [e(x) for e, x in 
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None] 
        
        tmp = torch.cat(tmp, dim=1)
        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        # x = torch.cat(z, dim=1)
        x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            #x = x.view(x.size(0), self.gin_dimension * self.k)
            x = self.sort_pool_nn(x)
            x = x.squeeze()
        ls = self.lin_h(batch.ls)
        x = torch.cat([x, ls], dim=-1)
        if not self.use_as_feature_extractor:
            x = self.cls(x)
        
        return x


class GCN_MPML(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        pooling_strategy: str=None,    
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        else:
            self.embed_deg = None
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0 
        assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = GCNConv(in_channels=n_1, out_channels=n_2)    
            self.convs.append(l)
            self.bns.append(nn.BatchNorm1d(n_2))   
        
        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
            #nn.Conv1d(
            #    in_channels=gin_dimension, 
            #    out_channels=gin_dimension, 
            #    kernel_size=self.k
            #)
        else:
            raise ValueError
        
        self.lin_h = nn.Linear(kwargs['ls_dim'], cls_hidden_dimension)
        # self.lin_h1 = nn.Linear(kwargs['ls_dim'],cls_hidden_dimension)

        self.cls = ClassifierHead(
            dataset, 
            dim_in=2*gin_dimension, 
            hidden_dim=cls_hidden_dimension, 
            drop_out=drop_out
        )         
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        
        tmp = [e(x) for e, x in 
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None] 
        
        tmp = torch.cat(tmp, dim=1)
        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        # x = torch.cat(z, dim=1)
        x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            #x = x.view(x.size(0), self.gin_dimension * self.k)
            x = self.sort_pool_nn(x)
            x = x.squeeze()
        ls = self.lin_h(batch.ls)
        x = torch.cat([x, ls], dim=-1)
        if not self.use_as_feature_extractor:
            x = self.cls(x)
        
        return x

class GCN_MPML_Learned(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        use_node_features: bool = None,
        pooling_strategy: str=None,
        concat_fp: str = None,
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        try:
            max_node_deg = dataset.max_node_deg
            num_node_lab = dataset.num_node_lab
        except:
            use_node_degree = False
            use_node_label = False
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        elif use_node_features:
            try:
                assert 'input_features_dim' in kwargs and kwargs['input_features_dim'] is not None
            except:
                raise Exception("If using input features input_feature_dim must be set")
            self.embed_deg = None

        self.use_node_features = use_node_features
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0
        dim_input += kwargs['input_features_dim'] if use_node_features else 0 
        # assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        # for n_1, n_2 in zip(dims[:-1], dims[1:]):            
        #     l = gin_mlp_factory(gin_mlp_type, n_1, n_2)    
        #     self.convs.append(GINConv(l, train_eps=True))
        #     self.bns.append(nn.BatchNorm1d(n_2))   
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = GCNConv(in_channels=n_1, out_channels=n_2)    
            self.convs.append(l)
            self.bns.append(nn.BatchNorm1d(n_2))
        
        self.lin_h = nn.Sequential(
            nn.Linear(sum(dims), dim),
            nn.BatchNorm1d(dim), 
            nn.LeakyReLU(),
            nn.Linear(dim, 2), 
            nn.Sigmoid()
        )
        
        self.sample_pts = nn.Parameter(torch.tensor(np.random.randint(0,100,size=(25,2)) * 0.01, dtype=torch.float, requires_grad=True))
        # self.lin_h = nn.Linear(gin_dimension, 2)
        self.mpml_0 = MultiPersDiff(res=0.01, hom_rank=0, num_center_pts=25, step=20, l=2)
        # self.mpml_0 = MultiPersDiffOld(res=0.01, hom_rank=0, step=25, l=2)
        # self.mpml_1 = MultiPersDiff(res=0.01, hom_rank=1, step=10, l=2)
        
        # num_centre_pts = int(1 / (10 * 0.01))
        # num_centre_pts = num_centre_pts * num_centre_pts
        num_centre_pts = 25

        self.lin_h0 = nn.Linear(num_centre_pts, cls_hidden_dimension)
        self.lin_h1 = nn.Linear(num_centre_pts, cls_hidden_dimension)
        cls_in_dim = 2 * cls_hidden_dimension
        self.concat_fp = concat_fp
        if self.concat_fp == 'maccs':
            cls_in_dim += 167
        elif self.concat_fp in ['ecfp','morgan2', 'morgan3']:
            cls_in_dim += 2048
        
        self.cls = ClassifierHead(dataset, dim_in= cls_in_dim, hidden_dim=cls_hidden_dimension, drop_out=drop_out)

              
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        try:
            node_deg  = batch.node_deg
        except:
            node_deg = None
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        gpu_dev = batch.edge_index.device
        fp = batch[self.concat_fp] if self.concat_fp else torch.empty([], device=gpu_dev)
        if not self.use_node_features:
            tmp = [e(x) for e, x in 
                zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
                if e is not None] 
            
            tmp = torch.cat(tmp, dim=1)
        else:
            tmp = batch.x

        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        x = torch.cat(z, dim=1)
        
        # if not self.use_as_feature_extractor:
        #     x = self.cls(x)
        f = self.lin_h(x)
        # f, edge_index = f.to(torch.device('cpu')), edge_index.to(torch.device('cpu'))
        lmbda_0, lmbda_1 = self.mpml_0(f, edge_index, self.sample_pts)
        lmbda_0, lmbda_1 = lmbda_0.to(f.device), lmbda_1.to(f.device)
        lmbda_0, lmbda_1= lmbda_0.sum(-1), lmbda_1.sum(-1)
        # self.mpml_0.mpl.set_hom_rank(1)
        # lmbda_1 = self.mpml_0(f, edge_index)
        # lmbda_1 = lmbda_1.sum(-1).to(gpu_dev)
        # self.mpml_0.mpl.refresh_rank_info()
        # print(f"{lmbda_0.shape}")
        # print(f"{lmbda_1.shape}")
        lmbda_0 = self.lin_h0(lmbda_0)
        lmbda_1 = self.lin_h1(lmbda_1)
        lmbda = torch.cat([lmbda_0, lmbda_1]).reshape((1, -1))
        # lmbda = lmbda_0.reshape((1, -1))
        # lmbda = lmbda.to(gpu_dev)
        if self.concat_fp is not None:
            lmbda = torch.column_stack([lmbda, fp.view(1, -1)])
        lmbda = self.act(lmbda)
        z = self.cls(lmbda)
        return z

class GAT_MPML(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        pooling_strategy: str=None,    
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        else:
            self.embed_deg = None
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0 
        assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        num_head = 1
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = GATConv(in_channels=num_head * n_1, out_channels=n_2, heads=8)
            self.convs.append(l)
            num_head = 8
            self.bns.append(nn.BatchNorm1d(num_head * n_2))
              
        
        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
            #nn.Conv1d(
            #    in_channels=gin_dimension, 
            #    out_channels=gin_dimension, 
            #    kernel_size=self.k
            #)
        else:
            raise ValueError
        
        self.lin_h = nn.Linear(kwargs['ls_dim'], cls_hidden_dimension)
        # self.lin_h1 = nn.Linear(kwargs['ls_dim'],cls_hidden_dimension)

        self.cls = ClassifierHead(
            dataset, 
            dim_in=(num_head + 1) * gin_dimension, 
            hidden_dim=cls_hidden_dimension, 
            drop_out=drop_out
        )         
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        
        tmp = [e(x) for e, x in 
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None] 
        
        tmp = torch.cat(tmp, dim=1)
        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        # x = torch.cat(z, dim=1)
        x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            #x = x.view(x.size(0), self.gin_dimension * self.k)
            x = self.sort_pool_nn(x)
            x = x.squeeze()
        ls = self.lin_h(batch.ls)
        x = torch.cat([x, ls], dim=-1)
        if not self.use_as_feature_extractor:
            x = self.cls(x)
        
        return x
    
class GAT_MPML_Learned(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        use_node_features: bool = None,
        pooling_strategy: str=None,
        concat_fp: str = None,
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        try:
            max_node_deg = dataset.max_node_deg
            num_node_lab = dataset.num_node_lab
        except:
            use_node_degree = False
            use_node_label = False
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        elif use_node_features:
            try:
                assert 'input_features_dim' in kwargs and kwargs['input_features_dim'] is not None
            except:
                raise Exception("If using input features input_feature_dim must be set")
            self.embed_deg = None

        self.use_node_features = use_node_features
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0
        dim_input += kwargs['input_features_dim'] if use_node_features else 0 
        # assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        # for n_1, n_2 in zip(dims[:-1], dims[1:]):            
        #     l = gin_mlp_factory(gin_mlp_type, n_1, n_2)    
        #     self.convs.append(GINConv(l, train_eps=True))
        #     self.bns.append(nn.BatchNorm1d(n_2))   
        
        num_head = 1
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = GATConv(in_channels=num_head * n_1, out_channels=n_2, heads=8)
            self.convs.append(l)
            num_head = 8
            self.bns.append(nn.BatchNorm1d(num_head * n_2))
        
        lin_in_dim = (num_head * n_2) + dims[0]
        self.lin_h = nn.Sequential(
            nn.Linear(lin_in_dim, dim),
            nn.BatchNorm1d(dim), 
            nn.LeakyReLU(),
            nn.Linear(dim, 2), 
            nn.Sigmoid()
        )
        
        self.sample_pts = nn.Parameter(torch.tensor(np.random.randint(0,100,size=(25,2)) * 0.01, dtype=torch.float, requires_grad=True))
        # self.lin_h = nn.Linear(gin_dimension, 2)
        self.mpml_0 = MultiPersDiff(res=0.01, hom_rank=0, num_center_pts=25, step=20, l=2)
        # self.mpml_0 = MultiPersDiffOld(res=0.01, hom_rank=0, step=25, l=2)
        # self.mpml_1 = MultiPersDiff(res=0.01, hom_rank=1, step=10, l=2)
        
        # num_centre_pts = int(1 / (10 * 0.01))
        # num_centre_pts = num_centre_pts * num_centre_pts
        num_centre_pts = 25

        self.lin_h0 = nn.Linear(num_centre_pts, cls_hidden_dimension)
        self.lin_h1 = nn.Linear(num_centre_pts, cls_hidden_dimension)
        cls_in_dim = 2 * cls_hidden_dimension
        self.concat_fp = concat_fp
        if self.concat_fp == 'maccs':
            cls_in_dim += 167
        elif self.concat_fp in ['ecfp','morgan2', 'morgan3']:
            cls_in_dim += 2048
        
        self.cls = ClassifierHead(dataset, dim_in= cls_in_dim, hidden_dim=cls_hidden_dimension, drop_out=drop_out)

              
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        try:
            node_deg  = batch.node_deg
        except:
            node_deg = None
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        gpu_dev = batch.edge_index.device
        fp = batch[self.concat_fp] if self.concat_fp else torch.empty([], device=gpu_dev)
        if not self.use_node_features:
            tmp = [e(x) for e, x in 
                zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
                if e is not None] 
            
            tmp = torch.cat(tmp, dim=1)
        else:
            tmp = batch.x

        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        x = torch.cat(z, dim=1)
        
        # if not self.use_as_feature_extractor:
        #     x = self.cls(x)
        f = self.lin_h(x)
        # f, edge_index = f.to(torch.device('cpu')), edge_index.to(torch.device('cpu'))
        lmbda_0, lmbda_1 = self.mpml_0(f, edge_index, self.sample_pts)
        lmbda_0, lmbda_1 = lmbda_0.to(f.device), lmbda_1.to(f.device)
        lmbda_0, lmbda_1= lmbda_0.sum(-1), lmbda_1.sum(-1)
        # self.mpml_0.mpl.set_hom_rank(1)
        # lmbda_1 = self.mpml_0(f, edge_index)
        # lmbda_1 = lmbda_1.sum(-1).to(gpu_dev)
        # self.mpml_0.mpl.refresh_rank_info()
        # print(f"{lmbda_0.shape}")
        # print(f"{lmbda_1.shape}")
        lmbda_0 = self.lin_h0(lmbda_0)
        lmbda_1 = self.lin_h1(lmbda_1)
        lmbda = torch.cat([lmbda_0, lmbda_1]).reshape((1, -1))
        # lmbda = lmbda_0.reshape((1, -1))
        # lmbda = lmbda.to(gpu_dev)
        if self.concat_fp is not None:
            lmbda = torch.column_stack([lmbda, fp.view(1, -1)])
        lmbda = self.act(lmbda)
        z = self.cls(lmbda)
        return z

class GCN(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        pooling_strategy: str=None,    
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        else:
            self.embed_deg = None
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0 
        assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = GCNConv(in_channels=n_1, out_channels=n_2)    
            self.convs.append(l)
            self.bns.append(nn.BatchNorm1d(n_2))   
        
        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
            #nn.Conv1d(
            #    in_channels=gin_dimension, 
            #    out_channels=gin_dimension, 
            #    kernel_size=self.k
            #)
        else:
            raise ValueError
        
        # self.lin_h = nn.Linear(kwargs['ls_dim'], cls_hidden_dimension)
        # self.lin_h1 = nn.Linear(kwargs['ls_dim'],cls_hidden_dimension)

        self.cls = ClassifierHead(
            dataset, 
            dim_in= gin_dimension, 
            hidden_dim=cls_hidden_dimension, 
            drop_out=drop_out
        )         
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        
        tmp = [e(x) for e, x in 
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None] 
        
        tmp = torch.cat(tmp, dim=1)
        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        # x = torch.cat(z, dim=1)
        x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            #x = x.view(x.size(0), self.gin_dimension * self.k)
            x = self.sort_pool_nn(x)
            x = x.squeeze()
        # ls = self.lin_h(batch.ls)
        # x = torch.cat([x, ls], dim=-1)
        if not self.use_as_feature_extractor:
            x = self.cls(x)
        
        return x

class GAT(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        pooling_strategy: str=None,    
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        else:
            self.embed_deg = None
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0 
        assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        num_head = 1
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = GATConv(in_channels=num_head * n_1, out_channels=n_2, heads=8)
            self.convs.append(l)
            num_head = 8
            self.bns.append(nn.BatchNorm1d(num_head * n_2))
              
        
        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
            #nn.Conv1d(
            #    in_channels=gin_dimension, 
            #    out_channels=gin_dimension, 
            #    kernel_size=self.k
            #)
        else:
            raise ValueError
        
        # self.lin_h = nn.Linear(kwargs['ls_dim'], cls_hidden_dimension)
        # self.lin_h1 = nn.Linear(kwargs['ls_dim'],cls_hidden_dimension)

        self.cls = ClassifierHead(
            dataset, 
            dim_in= num_head * gin_dimension, 
            hidden_dim=cls_hidden_dimension, 
            drop_out=drop_out
        )         
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        
        tmp = [e(x) for e, x in 
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None] 
        
        tmp = torch.cat(tmp, dim=1)
        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        # x = torch.cat(z, dim=1)
        x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            #x = x.view(x.size(0), self.gin_dimension * self.k)
            x = self.sort_pool_nn(x)
            x = x.squeeze()
        # ls = self.lin_h(batch.ls)
        # x = torch.cat([x, ls], dim=-1)
        if not self.use_as_feature_extractor:
            x = self.cls(x)
        
        return x

class GIN_MPML_Learned(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        use_node_features: bool = None,
        pooling_strategy: str=None,
        concat_fp: str = None,
        num_center_pts: int=None,
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        try:
            max_node_deg = dataset.max_node_deg
            num_node_lab = dataset.num_node_lab
        except:
            use_node_degree = False
            use_node_label = False
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        elif use_node_features:
            try:
                assert 'input_features_dim' in kwargs and kwargs['input_features_dim'] is not None
            except:
                raise Exception("If using input features input_feature_dim must be set")
            self.embed_deg = None

        self.use_node_features = use_node_features
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0
        dim_input += kwargs['input_features_dim'] if use_node_features else 0 
        # assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)    
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))   
        
        
        self.lin_h = nn.Sequential(
            nn.Linear(sum(dims), dim),
            nn.BatchNorm1d(dim), 
            nn.LeakyReLU(),
            nn.Linear(dim, 2), 
            nn.Sigmoid()
        )
        
        self.num_center_pts = num_center_pts
        self.sample_pts = nn.Parameter(torch.tensor(np.random.randint(0,100,size=(self.num_center_pts,2)) * 0.01, dtype=torch.float, requires_grad=True))
        # self.lin_h = nn.Linear(gin_dimension, 2)
        self.mpml_0 = MultiPersDiff(res=0.01, hom_rank=0, num_center_pts=self.num_center_pts, step=20, l=2)
        # self.mpml_0 = MultiPersDiffOld(res=0.01, hom_rank=0, step=25, l=2)
        # self.mpml_1 = MultiPersDiff(res=0.01, hom_rank=1, step=10, l=2)
        
        # num_centre_pts = int(1 / (10 * 0.01))
        # num_centre_pts = num_centre_pts * num_centre_pts
        # num_centre_pts = 25
        

        self.lin_h0 = nn.Linear(self.num_center_pts, cls_hidden_dimension)
        self.lin_h1 = nn.Linear(self.num_center_pts, cls_hidden_dimension)
        cls_in_dim = 2 * cls_hidden_dimension
        self.concat_fp = concat_fp
        if self.concat_fp == 'maccs':
            cls_in_dim += 167
        elif self.concat_fp in ['ecfp','morgan2', 'morgan3']:
            cls_in_dim += 2048
        
        self.cls = ClassifierHead(dataset, dim_in= cls_in_dim, hidden_dim=cls_hidden_dimension, drop_out=drop_out)

              
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        try:
            node_deg  = batch.node_deg
        except:
            node_deg = None
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        gpu_dev = batch.edge_index.device
        fp = batch[self.concat_fp] if self.concat_fp else torch.empty([], device=gpu_dev)
        if not self.use_node_features:
            tmp = [e(x) for e, x in 
                zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
                if e is not None] 
            
            tmp = torch.cat(tmp, dim=1)
        else:
            tmp = batch.x

        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        x = torch.cat(z, dim=1)
        
        # if not self.use_as_feature_extractor:
        #     x = self.cls(x)
        f = self.lin_h(x)
        # f, edge_index = f.to(torch.device('cpu')), edge_index.to(torch.device('cpu'))
        lmbda_0, lmbda_1 = self.mpml_0(f, edge_index, self.sample_pts)
        lmbda_0, lmbda_1 = lmbda_0.to(f.device), lmbda_1.to(f.device)
        lmbda_0, lmbda_1= lmbda_0.sum(-1), lmbda_1.sum(-1)
        # self.mpml_0.mpl.set_hom_rank(1)
        # lmbda_1 = self.mpml_0(f, edge_index)
        # lmbda_1 = lmbda_1.sum(-1).to(gpu_dev)
        # self.mpml_0.mpl.refresh_rank_info()
        # print(f"{lmbda_0.shape}")
        # print(f"{lmbda_1.shape}")
        lmbda_0 = self.lin_h0(lmbda_0)
        lmbda_1 = self.lin_h1(lmbda_1)
        lmbda = torch.cat([lmbda_0, lmbda_1]).reshape((1, -1))
        # lmbda = lmbda_0.reshape((1, -1))
        # lmbda = lmbda.to(gpu_dev)
        if self.concat_fp is not None:
            lmbda = torch.column_stack([lmbda, fp.view(1, -1)])
        lmbda = self.act(lmbda)
        z = self.cls(lmbda)
        return z

def get_row_col(num_pts, num_simplices, num_lines, id_tensor: torch.Tensor):
    pt_row = torch.div(id_tensor, num_lines * num_simplices, rounding_mode='floor')
    simplex_id_col = id_tensor % (num_lines * num_simplices)
    simplex_id_col = torch.div(simplex_id_col, num_lines, rounding_mode='floor')
    
    return pt_row, simplex_id_col

class moveSimplices(Function):
    """
    @staticmethod
    THIS ONE IS SLOWER. I KNOW RIGHT?
    
    def forward(ctx, filtration, bars, sample_pts):
        # Vectorized version of movesimplices
        l = 2
        grid_res = 0.01
        num_pts = sample_pts.shape[0]
        num_simplices = filtration.shape[0]
        f_x = filtration[:, 0]
        f_y = filtration[:, 1]

        shift = torch.arange(-l, l+1, 1, dtype=filtration.dtype, device=filtration.device)
        f_x = f_x.repeat_interleave(shift.shape[0]).repeat(sample_pts.shape[0])
        f_y = f_y.repeat_interleave(shift.shape[0]).repeat(sample_pts.shape[0])
        grad_matrices = []
        # bars must have shape of (num_pts * num_pts, num_ranks). Ideally like (2500, 5).
        # Must flatten before passing.
        grad_matrix_in_this_rank = []
        # print(f"Bars: {bars}")
        
        pt_x = (grid_res * sample_pts[:, 0]).repeat_interleave(shift.shape[0] * num_simplices)
        pt_y = (grid_res * sample_pts[:, 1]).repeat_interleave(shift.shape[0] * num_simplices)
        for rk in range(bars.shape[-1]):
            grad_matrix = torch.zeros((num_pts, 2 * filtration.shape[0]))
            line_x = shift.repeat(num_pts * num_simplices) * bars[:, rk].repeat_interleave(shift.shape[0] * num_simplices)  + pt_x
            line_y = shift.repeat(num_pts * num_simplices) * bars[:, rk].repeat_interleave(shift.shape[0] * num_simplices) + pt_y
            last_y = l * bars[:, rk].repeat_interleave(shift.shape[0] * num_simplices) + pt_y
            last_x = l * bars[:, rk].repeat_interleave(shift.shape[0] * num_simplices) + pt_x

            lower_x_constraining = torch.isclose(line_x, f_x, atol=0.01) & (f_y <= last_y) & ((f_x + f_y) < (pt_x + pt_y))
            upper_x_constraining = torch.isclose(line_x, f_x, atol=0.01) & (f_y <= last_y) & ((f_x + f_y) > (pt_x + pt_y))

            lower_y_constraining = torch.isclose(line_y, f_y, atol=0.01) & (f_x <= last_x) & ((f_x + f_y) < (pt_x + pt_y))
            upper_y_constraining = torch.isclose(line_y, f_y, atol=0.01) & (f_x <= last_x) & ((f_x + f_y) > (pt_x + pt_y))

            lower_x_constraining_idx = torch.nonzero(lower_x_constraining)
            upper_x_constraining_idx = torch.nonzero(upper_x_constraining)
            
            lower_y_constraining_idx = torch.nonzero(lower_y_constraining)
            upper_y_constraining_idx = torch.nonzero(upper_y_constraining)

            lower_x_constraining_row_idx, lower_x_constraining_simplex_idx = get_row_col(num_pts, num_simplices, shift.shape[0], lower_x_constraining_idx)
            upper_x_constraining_row_idx, upper_x_constraining_simplex_idx = get_row_col(num_pts, num_simplices, shift.shape[0], upper_x_constraining_idx)
            
            lower_y_constraining_row_idx, lower_y_constraining_simplex_idx = get_row_col(num_pts, num_simplices, shift.shape[0], lower_y_constraining_idx)
            upper_y_constraining_row_idx, upper_y_constraining_simplex_idx = get_row_col(num_pts, num_simplices, shift.shape[0], upper_y_constraining_idx)

            grad_matrix[lower_x_constraining_row_idx, 2 * lower_x_constraining_simplex_idx] = -1
            grad_matrix[lower_y_constraining_row_idx, 2 * lower_y_constraining_simplex_idx + 1] = -1

            grad_matrix[upper_x_constraining_row_idx, 2 * upper_x_constraining_simplex_idx] = +1
            grad_matrix[upper_y_constraining_row_idx, 2 * upper_y_constraining_simplex_idx + 1] = +1
            grad_matrix_in_this_rank.append(grad_matrix)
        
        grad_matrices = torch.stack(grad_matrix_in_this_rank, dim=0)
        ctx.grad_matrices = grad_matrices
        return bars
    """
    
    
    @staticmethod
    def forward(ctx, filtration, bars, sample_pts_t):
        sample_pts = sample_pts_t.clone().detach()
        # sample_pts = torch.tensor(sample_pts, dtype=torch.int)
        if filtration.grad_fn is None:
            return bars[0], bars[1]
        l = 2
        grid_res = 0.01
        shift = torch.arange(-l, l+1, 1, dtype=torch.float32, device=filtration.device)
        f_x = filtration[:, 0]
        f_y = filtration[:, 1]
        f_x = f_x.view(-1, 1).expand(-1, shift.shape[0])
        f_y = f_y.view(-1, 1).expand(-1, shift.shape[0])
        grad_matrices = []
        # bars must have shape of (num_pts * num_pts, num_ranks). Ideally like (2500, 5).
        # Must flatten before passing.
        # print(f"Bars: {bars}")
        grad_matrices_list = []
        grad_matrices_list_center_pts = []
        for bar in bars:
            grad_matrix_in_this_rank = []
            grad_matrix_in_this_rank_center_pts = []
            for rk in range(bar.shape[-1]):
                grad_matrix = torch.zeros((len(sample_pts), 2 * filtration.shape[0]))
                grad_matrix_center_pts = torch.zeros((len(sample_pts), 2))
                for i, pt in enumerate(sample_pts):
                    # if torch.all(bars[i, rk] == 0):
                    #     continue
                    # pt_x = pt[0] * grid_res
                    # pt_y = pt[1] * grid_res
                    pt_x = pt[0]
                    pt_y = pt[1]
                    # line_x = shift * (bars[i, rk] + grid_res) + pt_x
                    # line_y = shift * (bars[i, rk] + grid_res) + pt_y

                    line_x = shift * (bar[i, rk] + grid_res)+ pt_x
                    line_y = shift * (bar[i, rk] + grid_res)+ pt_y

                    # print(f"pt: {pt_x}, {pt_y} bar: {bars[i, rk]} rk: {rk} lines: {line_x}")
                    
                    lower_x_constraining = torch.isclose(line_x, f_x, atol=0.01) & (f_y <= line_y[-1]) & ((f_x + f_y) < (pt_x + pt_y))
                    upper_x_constraining = torch.isclose(line_x, f_x, atol=0.01) & (f_y <= line_y[-1]) & ((f_x + f_y) > (pt_x + pt_y))

                    lower_y_constraining = torch.isclose(line_y, f_y, atol=0.01) & (f_x <= line_x[-1]) & ((f_x + f_y) < (pt_x + pt_y))
                    upper_y_constraining = torch.isclose(line_y, f_y, atol=0.01) & (f_x <= line_x[-1]) & ((f_x + f_y) > (pt_x + pt_y))

                    lower_x_constraining_idx = torch.nonzero(lower_x_constraining)[:, 0]
                    upper_x_constraining_idx = torch.nonzero(upper_x_constraining)[:, 0]
                    
                    lower_y_constraining_idx = torch.nonzero(lower_y_constraining)[:, 0]
                    upper_y_constraining_idx = torch.nonzero(upper_y_constraining)[:, 0]

                    grad_matrix[i, 2 * lower_x_constraining_idx] = -1
                    grad_matrix[i, 2 * lower_y_constraining_idx + 1] = -1

                    grad_matrix[i, 2 * upper_x_constraining_idx] = +1
                    grad_matrix[i, 2 * upper_y_constraining_idx + 1] = +1
                    
                    if len(lower_x_constraining_idx) > 0:
                        grad_matrix_center_pts[i, 0] = +1
                    if len(upper_x_constraining_idx) > 0:
                        grad_matrix_center_pts[i, 0] = -1
                    if len(lower_y_constraining_idx) > 0:
                        grad_matrix_center_pts[i, 1] = +1
                    if len(upper_y_constraining_idx) > 0:
                        grad_matrix_center_pts[i, 1] = -1
                grad_matrix_in_this_rank.append(grad_matrix)
                grad_matrix_in_this_rank_center_pts.append(grad_matrix_center_pts)
            grad_matrices = torch.stack(grad_matrix_in_this_rank, dim=0)
            grad_matrices_center_pts = torch.stack(grad_matrix_in_this_rank_center_pts, dim=0)
            grad_matrices_list.append(grad_matrices)
            grad_matrices_list_center_pts.append(grad_matrices_center_pts)
        ctx.grad_matrices_h0 = grad_matrices_list[0]
        ctx.grad_matrices_h1 = grad_matrices_list[1]
        ctx.grad_matrices_center_pts_h0 = grad_matrices_list_center_pts[0]
        ctx.grad_matrices_center_pts_h1 = grad_matrices_list_center_pts[1]
        return bars[0], bars[1]
    

    @staticmethod
    def backward(ctx, grad_output_1, grad_output_2):
        """
        I couldn't understand einsum. Then I followed 
        https://rockt.github.io/2018/04/30/einsum . This somehow made sense to me.
        To get the full gradient matrix, uncomment the following line"
        """
        # grad = torch.einsum('ji,ijk->ijk',[grad_output, ctx.grad_matrices]) 
        
        
        """
        To sum across ranks and centre points: torch.einsum('ji,ijk-> k',[grad_output, ctx.grad_matrices])
        I am using this for now. Check with team.
        """
        grad_mat_h0 = ctx.grad_matrices_h0.to(grad_output_1.device)
        grad_mat_h1 = ctx.grad_matrices_h1.to(grad_output_2.device)
        grad_mat_h0_center_pts = ctx.grad_matrices_center_pts_h0.to(grad_output_1.device)
        grad_mat_h1_center_pts = ctx.grad_matrices_center_pts_h1.to(grad_output_2.device)
        
        grad = torch.einsum('ji,ijk->k',[grad_output_1, grad_mat_h0]) + torch.einsum('ji,ijk->k',[grad_output_2, grad_mat_h1]) 
        grad_center_pts = torch.einsum('ji,ijk->jk',[grad_output_1, grad_mat_h0_center_pts]) + torch.einsum('ji,ijk->jk',[grad_output_2, grad_mat_h1_center_pts])
        # Make the gradient as shape of (m X 2), with m being number of simplices. That's what is needed.
        # Pytorch autodiff will take care of the vertices automatically.

        grad = grad.reshape((-1, 2))
        grad = grad / grad.shape[0]
        
        # grad_center_pts = grad_center_pts / 0.01
        # print(f"Grad matrices: {ctx.grad_matrices}")

        return grad, None, grad_center_pts

class MultiPersDiffFixecCenters(nn.Module):
    def __init__(self, res, hom_rank, num_center_pts=None, step=10, l=2, adaptive=False):
        super().__init__()
        self.res = res
        self.step = step
        self.l = l
        # self.grid_resolution = grid_resolution
        self.sample_pts = self.sample_grid()
        self.hom_rank = hom_rank
        self.filt_layer = compute_lub
        self.mpl = gril.MultiPers(hom_rank=hom_rank, l=l, res=res, step=step, num_centers= num_center_pts, adaptive=adaptive, ranks=list(range(1, 4)))
        self.mpl.set_max_jobs(40)
        self.sigmoid = nn.Sigmoid()

    def sample_grid(self):
        pts = []
        num_division = int(1.0 / self.res)
        for j in range(0, num_division, self.step):
            for i in range(0, num_division, self.step):
                pts.append((i, j))
        return pts

    def forward(self, f, edge_index):
        # f_ = Snap.apply(f)
        # f = self.filt_layer(x, edge_index)
        # f_v = (f - f.min(dim=0).values) / (f.max(dim=0).values - f.min(dim=0).values)
        # simplices = get_simplices(x.shape[0], edge_index)
        f_v = self.sigmoid(f)
        f_v = f_v.to('cpu')
        edge_index = edge_index.to('cpu')
        edges = pre_process_edges(edge_index)
        filt = self.filt_layer(f_v, edges)
        simplices = get_simplices(f_v.shape[0], edges)
        pers_inp = [(filt, simplices)]
        # np.save("filtration.npy", f.detach().numpy())
        # with open("simplices.txt", "w") as srf:
        #     for s in simplices:
        #         line = " ".join([str(sv) for sv in s])
        #         srf.write(f"{line}\n")

        self.mpl.set_hom_rank(0)
        bars_h0 = self.mpl.compute_landscape(self.sample_pts, pers_inp)
        self.mpl.set_hom_rank(1)
        bars_h1 = self.mpl.compute_landscape(self.sample_pts, pers_inp)
        # bars = bars[0].reshape(-1, bars.shape[1])
        # bars = bars[0].to(f.device)
        # filt = filt.to(f.device)
        self.mpl.refresh_rank_info()
        # sample_pts_t = torch.tensor(self.sample_pts, device=f_v.device)
        lambdas_h0, lambdas_h1 = moveSimplices.apply(filt, [bars_h0[0], bars_h1[0]], self.sample_pts)
        return lambdas_h0, lambdas_h1
    
class MultiPersDiff(nn.Module):
    def __init__(self, res, hom_rank, num_center_pts=16, step=20, l=2, adaptive=True):
        super().__init__()
        self.res = res
        self.num_center_pts = num_center_pts
        self.l = l
        # self.grid_resolution = grid_resolution
        self.sample_pts = self.sample_grid()
        self.hom_rank = hom_rank
        self.filt_layer = compute_lub
        self.mpl = gril.MultiPers(hom_rank=hom_rank, l=l, res=res, step=step, adaptive=adaptive, num_centers=num_center_pts, ranks=list(range(1, 4)))
        self.mpl.set_max_jobs(40)
        self.sigmoid = nn.Sigmoid()

    # def sample_grid(self):
    #     pts = []
    #     num_division = int(1.0 / self.res)
    #     for j in range(0, num_division, self.step):
    #         for i in range(0, num_division, self.step):
    #             pts.append((i, j))
    #     return pts
    def sample_grid(self):
        pts = []
        num_division = int(1.0/ self.res)
        for i in range(self.num_center_pts):
            rand_arr = np.random.randint(0, num_division, size=2)
            pts.append((rand_arr[0], rand_arr[1]))
        return pts
        

    def forward(self, f, edge_index, sample_pts):
        # f_ = Snap.apply(f)
        # f = self.filt_layer(x, edge_index)
        # f_v = (f - f.min(dim=0).values) / (f.max(dim=0).values - f.min(dim=0).values)
        # simplices = get_simplices(x.shape[0], edge_index)
        f_v = self.sigmoid(f)
        f_v = f_v.to('cpu')
        edge_index = edge_index.to('cpu')
        edges = pre_process_edges(edge_index)
        filt = self.filt_layer(f_v, edges)
        simplices = get_simplices(f_v.shape[0], edges)
        pers_inp = [(filt, simplices)]
        sample_pts = sample_pts.to('cpu')
        # np.save("filtration.npy", f.detach().numpy())
        # with open("simplices.txt", "w") as srf:
        #     for s in simplices:
        #         line = " ".join([str(sv) for sv in s])
        #         srf.write(f"{line}\n")
        sample_pts_copy = sample_pts.clone()
        sample_pts_np = sample_pts_copy.detach()/0.01
        sample_pts_np = np.array(sample_pts_np, dtype=int)
        # sample_pts = sample_pts * self.res
        self.mpl.set_hom_rank(0)
        bars_h0 = self.mpl.compute_landscape(sample_pts_np, pers_inp)
        self.mpl.set_hom_rank(1)
        bars_h1 = self.mpl.compute_landscape(sample_pts_np, pers_inp)
        # bars = bars[0].reshape(-1, bars.shape[1])
        # bars = bars[0].to(f.device)
        # filt = filt.to(f.device)
        self.mpl.refresh_rank_info()
        # sample_pts_t = torch.tensor(sample_pts, device=f_v.device, dtype=torch.float, requires_grad=True)
        lambdas_h0, lambdas_h1 = moveSimplices.apply(filt, [bars_h0[0], bars_h1[0]], sample_pts)
        return lambdas_h0, lambdas_h1


class GIN_FiltHead(nn.Module):
    def __init__(self, 
        dataset, 
        use_node_degree: bool=None, 
        use_node_label: bool=None, 
        gin_number: int=None, 
        gin_dimension: int=None,
        gin_mlp_type: str=None, 
        cls_hidden_dimension: int=None, 
        drop_out: float=None,
        set_node_degree_uninformative: bool=None,
        pooling_strategy: str=None,    
        **kwargs,  
    ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension
        
        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab
        
        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg+1)
        else:
            self.embed_deg = None
        
        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None
        
        dim_input = 0 
        dim_input += self.embed_deg.dim if use_node_degree else 0 
        dim_input += self.embed_lab.dim if use_node_label else 0 
        assert dim_input > 0 
        
        dims = [dim_input] + (gin_number)*[dim]
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.act   = torch.nn.functional.leaky_relu
        
        for n_1, n_2 in zip(dims[:-1], dims[1:]):            
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)    
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))   
        
        
        
        self.lin_h = nn.Linear(gin_dimension, 2)
        self.sigmoid = nn.Sigmoid()

        self.mpml_0 = MultiPersDiff(res=0.01, hom_rank=0, step=10, l=2)
        # self.mpml_1 = MultiPersDiff(res=0.01, hom_rank=1, step=10, l=2)
        
        

              
   
    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features  

    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        gpu_dev = batch.edge_index.device
        
        tmp = [e(x) for e, x in 
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None] 
        
        tmp = torch.cat(tmp, dim=1)
        
        z = [tmp]        
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)
  
        f_v = self.lin_h(z[-1])

        edges = pre_process_edges(edge_index)
        f = compute_lub(f_v, edges)
        lmbda_h0, lmbda_h1 = self.mpml_0(f, edge_index)
        # lmbda_0 = lmbda_0[1]
        lmbda_h0 = lmbda_h0.to(f.device).sum(-1)
        lmbda_h1 = lmbda_h1.to(f.device).sum(-1)
        
        return f, lmbda_h0, lmbda_h1


class GIN_MPML_GRILHead(nn.Module):
    def __init__(self,
        pth, 
        dataset, 
        model_cfg,  
    ):
        super().__init__()
        self.gin_mpml = GIN_MPML_Learned(dataset, **model_cfg)
        self.gin_mpml.load_state_dict(torch.load(pth))
        self.gin_mpml.mpml_0 = MultiPersDiff(res=0.01, hom_rank=0, step=10, l=2)
        # self.mpml_1 = MultiPersDiff(res=0.01, hom_rank=1, step=10, l=2)
        
        self.num_centre_pts = int(1 / (10 * 0.01))
        # self.num_centre_pts = self.num_centre_pts * self.num_centre_pts
    
    def forward(self, batch):
        
        node_deg  = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index
        gpu_dev = batch.edge_index.device
        fp = batch[self.gin_mpml.concat_fp] if self.gin_mpml.concat_fp else torch.empty([], device=gpu_dev)
        if not self.gin_mpml.use_node_features:
            tmp = [e(x) for e, x in 
                zip([self.gin_mpml.embed_deg, self.gin_mpml.embed_lab], [node_deg, node_lab])
                if e is not None] 
            
            tmp = torch.cat(tmp, dim=1)
        else:
            tmp = batch.x

        
        z = [tmp]        
        
        for conv, bn in zip(self.gin_mpml.convs, self.gin_mpml.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.gin_mpml.act(x)
            z.append(x)
  
        x = torch.cat(z, dim=1)
        
        # if not self.use_as_feature_extractor:
        #     x = self.cls(x)
        f = self.gin_mpml.lin_h(x)
        f_copy = torch.sigmoid(f)
        # f, edge_index = f.to(torch.device('cpu')), edge_index.to(torch.device('cpu'))
        lmbda_0, lmbda_1 = self.gin_mpml.mpml_0(f, edge_index)
        # lmbda_0, lmbda_1 = lmbda_0.to(f.device), lmbda_1.to(f.device)
        # lmbda_0, lmbda_1= lmbda_0.sum(-1), lmbda_1.sum(-1)
        return f_copy, lmbda_0.reshape((self.num_centre_pts, self.num_centre_pts, -1)), lmbda_1.reshape((self.num_centre_pts, self.num_centre_pts, -1))


        



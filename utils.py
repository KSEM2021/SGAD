import random
import torch
import os
from torch_geometric.data import Data
import json
#from sklearn.decomposition import PCA
#from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (
    remove_self_loops,
    to_undirected,
    coalesce,
    is_undirected,
    contains_self_loops,
)
import torch.nn.functional as F
from preprocess import get_base
from numpy import inf

def weighted_binary_cross_entropy(
    score,
    labels,
    pos_weight,
    eps=1e-7,
):
    """
    BCE on the fused probability score.

    score is already in [0, 1], so BCEWithLogitsLoss must not be used here.
    """
    score = score.clamp(min=eps, max=1.0 - eps)
    labels = labels.float().view(-1)

    per_sample_loss = F.binary_cross_entropy(
        score,
        labels,
        reduction="none",
    )

    sample_weight = torch.where(
        labels >= 0.5,
        torch.as_tensor(
            pos_weight,
            dtype=score.dtype,
            device=score.device,
        ),
        torch.ones_like(score),
    )
    return (per_sample_loss * sample_weight).mean()

        
def compute_pos_weight(labels):
    labels = labels.view(-1)
    num_anomaly = (labels >= 0.5).sum().float()
    num_normal = (labels < 0.5).sum().float()

    if num_anomaly.item() == 0:
        return labels.new_tensor(1.0)

    return num_normal / num_anomaly

def test_eval(labels, probs):
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        score['AUROC'] = roc_auc_score(labels, probs)
        score['AUPRC'] = average_precision_score(labels, probs)
    return score


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adj_sharp(adj):
    """
    Sparse implementation of

        D_hat^{-1/2} A_hat D_hat^{-1/2}

    where
        A_hat = 2I - A
        D_hat = 2I + D

    All operations are sparse.
    """

    adj = sp.coo_matrix(
        adj,
        dtype=np.float32
    )

    n = adj.shape[0]

    # Degree of original adjacency A
    degree = np.asarray(
        adj.sum(axis=1)
    ).reshape(-1)

    # D_hat^{-1/2}
    d_inv_sqrt = np.power(
        degree + 2.0,
        -0.5
    ).astype(np.float32)

    # A_hat = 2I - A
    adj_hat = (
        2.0 * sp.eye(
            n,
            dtype=np.float32,
            format="coo"
        )
        - adj
    ).tocoo()

    # Sparse normalization:
    # (D_hat^{-1/2} A_hat D_hat^{-1/2})_ij
    # = d_i^{-1/2} * A_hat_ij * d_j^{-1/2}
    adj_hat.data *= (
        d_inv_sqrt[adj_hat.row]
        * d_inv_sqrt[adj_hat.col]
    )

    return adj_hat

def get_sp_adj(adj):
    
    # 对称化邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 添加自环
    #adj = adj + sp.eye(adj.shape[0])

    # 计算度向量
    degrees = np.array(adj.sum(axis=1)).flatten()

    # 构造 D^(-0.5) 稀疏对角矩阵
    with np.errstate(divide='ignore'):
        deg_inv_sqrt = np.power(degrees, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    # 计算规范化邻接矩阵：I-D^{-0.5} * A * D^{-0.5}
    Lap = sp.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt

    # 转为 COO 格式并保存
    Lap = Lap.tocoo()
    Lap = sparse_mx_to_torch_sparse_tensor(Lap)

    return Lap


def compute_node_homophily(
    edge_index,
    labels,
):
    """
    edge_index: [2, E]
    labels:     [N]

    return:
        homophily: [N]
    """

    num_nodes = labels.size(0)

    src = edge_index[0]
    dst = edge_index[1]

    # 去掉 self-loop
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    # 每条边是否同标签
    same = (
        labels[src] == labels[dst]
    ).float()  # [E]

    # 每个节点的同标签邻居数量
    same_count = torch.zeros(
        num_nodes,
        device=labels.device,
    )

    same_count.index_add_(
        0,
        src,
        same,
    )

    # 每个节点的邻居数量
    degree = torch.zeros(
        num_nodes,
        device=labels.device,
    )

    degree.index_add_(
        0,
        src,
        torch.ones_like(same),
    )

    # 每个节点的同配度
    homophily = (
        same_count
        / degree.clamp_min(1.0)
    )

    # 孤立节点
    homophily[degree == 0] = -1

    return homophily

class Dataset:
    def __init__(self, name='cora', basename='all', prefix='/home/aiguoguo/guoguo/datasets'):

        self.train_idx = None
        self.val_idx = None
        self.train_mask = None
        self.val_mask = None
        self.graph = None
        self.x_list = None
        self.name = name
        self.base_name = basename
        prefix2 = '/home/aiguoguo/guoguo/datasets-plot/data/degree_anomaly_feature_1/5'

        pt_path = os.path.join(prefix2, f'{name}.pt')
        mat_path = os.path.join(prefix, f'{name}.mat')

        # =====================================================
        # 1. 优先读取 .pt
        # =====================================================
        if os.path.exists(pt_path):
            print(f'Load PT dataset: {pt_path}')

            try:
                loaded_data = torch.load(pt_path, map_location='cpu', weights_only=False)
            except TypeError:
                loaded_data = torch.load(pt_path, map_location='cpu')

            # 新生成的异常特征
            #if self.name in ['BlogCatalog','Flickr','ACM']:
            #    feat = loaded_data.x_pca
            #else:
            feat = loaded_data.x

            if not torch.is_tensor(feat):
                feat = torch.tensor(feat, dtype=torch.float)
            feat = feat.float()

            num_nodes = feat.size(0)

            # edge_index
            edge_index = loaded_data.edge_index.long().cpu()
            edge_index, _ = remove_self_loops(edge_index)

            if not is_undirected(edge_index):
                edge_index = to_undirected(edge_index, num_nodes=num_nodes)

            edge_index = coalesce(edge_index, num_nodes=num_nodes)

            # 异常标签
            if hasattr(loaded_data, 'y'):
                label = loaded_data.y
            elif hasattr(loaded_data, 'ano_labels'):
                label = loaded_data.ano_labels
            else:
                raise ValueError(f'{pt_path} does not contain y or ano_labels.')

            if not torch.is_tensor(label):
                label = torch.tensor(label)

            label = label.view(-1).float()

            # edge_index -> scipy adjacency
            row = edge_index[0].numpy()
            col = edge_index[1].numpy()
            values = np.ones(edge_index.size(1), dtype=np.float32)

            adj = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_nodes)).tocsr()

        # =====================================================
        # 2. .pt 不存在则读取原始 .mat
        # =====================================================
        else:
            print(f'PT file does not exist. Load MAT dataset: {mat_path}')

            if not os.path.exists(mat_path):
                raise FileNotFoundError(f'Neither {pt_path} nor {mat_path} exists.')

            data = sio.loadmat(mat_path)

            adj = data['Network'] if 'Network' in data else data['A']
            feat = data['Attributes']
            num_nodes = feat.shape[0]

            # adjacency -> edge_index
            adj_sp = sp.csr_matrix(adj)
            row, col = adj_sp.nonzero()
            edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)

            edge_index, _ = remove_self_loops(edge_index)

            if not is_undirected(edge_index):
                edge_index = to_undirected(edge_index, num_nodes=num_nodes)

            edge_index = coalesce(edge_index, num_nodes=num_nodes)

            # feature preprocessing
            if name in ['Amazon', 'Yelpchi', 'Tolokers', 'Tfinance']:
                feat = sp.lil_matrix(feat)
                feat = preprocess_features(feat)
            else:
                feat = sp.lil_matrix(feat).toarray()

            feat = torch.FloatTensor(feat)

            # label
            label = data['Label'] if 'Label' in data else data['gnd']
            label = torch.tensor(np.squeeze(np.asarray(label)), dtype=torch.float)

        # =====================================================
        # 3. 统一处理 .pt 和 .mat
        # =====================================================
        self.label = label
        self.feat = feat
        self.edge_index = edge_index

        # =====================================================
        # 4. Normalized adjacency
        # =====================================================
        if name in ['Yelpchi', 'Facebook']:
            adj_norm = normalize_adj(adj)
        else:
            adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        self.adj_norm = adj_norm

        # =====================================================
        # 5. Laplacian
        # =====================================================
        Lap = get_sp_adj(adj)
        self.Lap = Lap

        # =====================================================
        # 6. Sharpen adjacency
        # =====================================================
        adj_hat = normalize_adj_sharp(adj)
        adj_hat = sparse_mx_to_torch_sparse_tensor(adj_hat)
        self.adj_hat = adj_hat

        # =====================================================
        # 7. Binary anomaly labels
        # =====================================================
        ano_labels = label.view(-1).float()

        # =====================================================
        # 8. Node homophily
        # =====================================================
        homophily = compute_node_homophily(edge_index, ano_labels)
        self.homophily = homophily

        # =====================================================
        # 9. PyG Data
        # =====================================================
        data = Data(
            x=self.feat,
            x_list=self.x_list,
            adj_norm=self.adj_norm,
            adj_hat=self.adj_hat,
            ano_labels=ano_labels,
            Lap=self.Lap,
            edge_index=self.edge_index,
            homophily=self.homophily,
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            train_mask=self.train_mask,
            val_mask=self.val_mask
        )

        self.graph = data
    
    def split_vail(self, ratio=0.6):
        y = self.graph.ano_labels
        num_nodes = y.shape[0]
        normal_idx = torch.where(y == 0)[0].tolist()
        abnormal_idx = torch.where(y == 1)[0].tolist()
        # =====================================================
        # 2. Shuffle
        # =====================================================
        random.shuffle(normal_idx)
        random.shuffle(abnormal_idx)
        # =====================================================
        # 3. 60% train / 40% validation
        # =====================================================
        normal_train_num = int(len(normal_idx) * ratio)
        abnormal_train_num = int(len(abnormal_idx) * ratio)

        # Normal
        normal_train_idx = normal_idx[:normal_train_num]
        normal_val_idx = normal_idx[normal_train_num:]

        # Abnormal
        abnormal_train_idx = abnormal_idx[:abnormal_train_num]
        abnormal_val_idx = abnormal_idx[abnormal_train_num:]

        # =====================================================
        # 4. Merge normal + abnormal
        # =====================================================

        train_idx = (normal_train_idx + abnormal_train_idx)

        val_idx = (normal_val_idx+ abnormal_val_idx)

        # 再随机打乱一次，避免前面全是 normal
        random.shuffle(train_idx)
        random.shuffle(val_idx)

        # =====================================================
        # 5. Convert to Tensor
        # =====================================================

        train_idx = torch.tensor(train_idx,dtype=torch.long)

        # =====================================================
        # 6. Create boolean masks
        # =====================================================

        train_mask = torch.zeros(
            num_nodes,
            dtype=torch.bool
        )

        val_mask = torch.zeros(
            num_nodes,
            dtype=torch.bool
        )

        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # =====================================================
        # 7. Save to graph
        # =====================================================

        self.graph.train_idx = train_idx
        self.graph.val_idx = val_idx

        self.graph.train_mask = train_mask
        self.graph.val_mask = val_mask
    


    def load_base(self, K, edge_attr=None):
        file_path = '/home/niuchaoxi/guoguo/spectral_base/' + self.name + '_' + self.base_name + '_' + str(K) + '.pkl'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                list_mat = pickle.load(f)
                self.graph.x_list = list_mat
        else:
            list_mat = get_base(self.base_name, K, self.feat, self.edge_index, edge_attr)
            self.graph.x_list = list_mat
            #with open(file_path, 'wb') as f:
            #    pickle.dump(list_mat, f)
        #return list_mat


def read_json(model, K, shot, json_dir):
    # Construct the filename based on the dataset name and shot
    filename = f"{json_dir}/{model}_{K}_{shot}.json"

    # Check if the file exists
    if os.path.exists(filename):
        # Read the JSON file and return the dictionary
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {filename}: {e}")
                return None
    else:
        print(f"JSON file {filename} not found.")
        return None

def reconstruction_loss(
    x,
    x_hat,
):
    return 0.5 * torch.sum(
        (x - x_hat) ** 2
    )
    

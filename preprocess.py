import torch
import math
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from scipy.special import comb
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from torch_geometric.utils import is_undirected, to_undirected, add_self_loops, remove_self_loops, contains_isolated_nodes
import os
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def init_temp(base_name, K):  
    if base_name == 'mono':
        bound = np.sqrt(3/(K))
        TEMP = np.random.uniform(-bound, bound, K)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        temp = torch.tensor(TEMP).float()
    elif base_name == 'cheb':
        temp = torch.zeros(K).float()   
        # temp.data[0]=1.0
        temp.data.fill_(1.0)

    
    else:
        assert False, 'base_name error'
    return temp


def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   #adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sys_normalized_adjacency_i(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
'''
def load_base(dataname, base_name, K, x, edge_index, edge_attr=None):
    file_path = '/home/guoguoai/GFM-AD/spectral_base/' + dataname + '_' + base_name + '_' + str(K) + '.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            list_mat = pickle.load(f)
    else:
        list_mat = get_base(base_name, K, x, edge_index, edge_attr)
        with open(file_path, 'wb') as f:
            pickle.dump(list_mat, f)
    return list_mat
'''
def get_base(base_name, K, x, edge_index, edge_attr=None):
    if base_name == 'mono':
        list_mat = mono_base(K, x, edge_index, edge_attr)
    elif base_name == 'cheb':
        list_mat = cheb_base(K, x, edge_index, edge_attr)
    elif base_name == 'cheb2':
        list_mat = cheb2_base(K, x, edge_index, edge_attr)
    elif base_name == 'bern':
        list_mat = bern_base(K, x, edge_index, edge_attr)
    elif base_name == 'opt':
        list_mat = opt_base(K, x, edge_index, edge_attr)
    elif base_name == 'jac':
        list_mat = jacobi_base(K, x, edge_index, edge_attr)
    elif base_name == 'leg':
        list_mat = legendre_base(K, x, edge_index, edge_attr)
    elif base_name == 'ppr':
        list_mat = ppr_base(K, x, edge_index, edge_attr)
    elif base_name == 'heat':
        list_mat = heat_base(K, x, edge_index, edge_attr)
    elif base_name =='dif':
        list_mat = diffusion_wavelet_base(K, x, edge_index, edge_attr)
    elif base_name =='lag':
        list_mat = laguerre_base(K, x, edge_index, edge_attr)


    elif base_name == 'all':

        #list_mono = mono_base(K, x, edge_index, edge_attr)

        #list_cheb = cheb_base(K, x, edge_index, edge_attr)
        
        #list_bern = bern_base(K, x, edge_index, edge_attr)

        list_opt = opt_base(K, x, edge_index, edge_attr)
        
        list_jac = jacobi_base(K, x, edge_index, edge_attr)
        '''
        list_leg = legendre_base(K, x, edge_index, edge_attr)
        list_ppr = ppr_base(K, x, edge_index, edge_attr)
        list_heat = heat_base(K, x, edge_index, edge_attr)
        list_dif = diffusion_wavelet_base(K, x, edge_index, edge_attr)
        list_lag = laguerre_base(K, x, edge_index, edge_attr)
        '''
        # Python list concatenation
        list_mat = (
            #list_mono
            #list_cheb
            #+ list_bern
            list_opt
            +list_jac
           # +list_leg
            #+list_ppr
           # +list_heat
           # +list_dif
           # +list_lag
        
        )

    return list_mat
def cheb2_base(
    K,
    x,
    edge_index,
    edge_attr,
):
    """
    Chebyshev polynomial of the second kind.

    H_k = U_k(L_tilde) X
    """

    node_dim = 0

    edge_index_L, norm_L = get_laplacian(
        edge_index,
        edge_attr,
        normalization='sym',
        dtype=x.dtype,
        num_nodes=x.size(node_dim),
    )

    edge_index_tilde, norm_tilde = add_self_loops(
        edge_index_L,
        norm_L,
        fill_value=-1.0,
        num_nodes=x.size(node_dim),
    )

    L_tilde = to_scipy_sparse_matrix(
        edge_index_tilde,
        norm_tilde,
        x.size(node_dim),
    )

    L_tilde = sparse_mx_to_torch_sparse_tensor(
        L_tilde
    ).to(x.device)

    list_mat = []

    U0 = x
    list_mat.append(U0)

    if K == 0:
        return list_mat

    U1 = 2.0 * torch.spmm(
        L_tilde,
        x,
    )

    list_mat.append(U1)

    Ukm2 = U0
    Ukm1 = U1

    for k in range(2, K + 1):

        Uk = (
            2.0
            * torch.spmm(
                L_tilde,
                Ukm1,
            )
            -
            Ukm2
        )

        list_mat.append(Uk)

        Ukm2 = Ukm1
        Ukm1 = Uk

    assert len(list_mat) == K + 1

    return list_mat
def mono_base(K, x, edge_index, edge_attr):
    edge_index, norm = gcn_norm(edge_index, edge_attr, num_nodes=x.size(0), dtype=x.dtype)
    adj = to_scipy_sparse_matrix(edge_index, norm, x.size(0))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    device = x.device
    adj = adj.to(device)

    list_mat = []
    list_mat.append(x)
    tmp_mat = x
    for _ in range(K):
        tmp_mat = torch.spmm(adj, tmp_mat)
        list_mat.append(tmp_mat)
    return list_mat

def cheb_base(K, x, edge_index, edge_attr):
    # self.temp.data.fill_(0.0)
    # self.temp.data[0]=1.0
    # coe[i]/i**self.q #The positive constant
    node_dim = 0
    #L=I-D^(-0.5)AD^(-0.5)
    edge_index1, norm1 = get_laplacian(edge_index, edge_attr,normalization='sym', dtype=x.dtype, num_nodes=x.size(node_dim))

    #L_tilde=L-I
    edge_index_tilde, norm_tilde = add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(node_dim))
    L_tilde = to_scipy_sparse_matrix(edge_index_tilde, norm_tilde, x.size(node_dim))
    L_tilde = sparse_mx_to_torch_sparse_tensor(L_tilde)

    device = x.device
    L_tilde = L_tilde.to(device)
    
    list_mat = []
    Tx_0=x
    list_mat.append(Tx_0)
    Tx_1 = torch.spmm(L_tilde, x)
    list_mat.append(Tx_1)

    for i in range(2, K+1):
        Tx_2 = 2 * torch.spmm(L_tilde, Tx_1) - Tx_0
        list_mat.append(Tx_2)
        Tx_0, Tx_1 = Tx_1, Tx_2

    return list_mat


def bern_base(K, x, edge_index, edge_attr):
    edge_weight = edge_attr
    node_dim = 0
    #L=I-D^(-0.5)AD^(-0.5)
    edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(node_dim))
    Matrix_L = to_scipy_sparse_matrix(edge_index1, norm1, x.size(node_dim))
    Matrix_L = sparse_mx_to_torch_sparse_tensor(Matrix_L)
    #2I-L
    edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(node_dim))
    Matrix_2I_L = to_scipy_sparse_matrix(edge_index2, norm2, x.size(node_dim))
    Matrix_2I_L = sparse_mx_to_torch_sparse_tensor(Matrix_2I_L)
    list_mat = []
    tmp=[]
    tmp.append(x)
    for i in range(K):
        x = torch.spmm(Matrix_2I_L, x)
        tmp.append(x)
    tmp[K] = (comb(K,0)/(2**K))*tmp[K]
    list_mat.append(tmp[K])
    for i in range(K):
        x = tmp[K-i-1]
        x = torch.spmm(Matrix_L, x)
        for _ in range(i):
            x = torch.spmm(Matrix_L, x)
        list_mat.append((comb(K,i+1)/(2**K))*x)
    assert len(list_mat)==K+1
    return list_mat

def opt_base(K, x, edge_index, edge_attr):
    edge_weight = edge_attr
    node_dim = 0
    list_mat = []
    _, norm_A = gcn_norm(edge_index, add_self_loops=False) # norm_A = D^-1/2 A D^-1/2
    Matrix_norm_A = to_scipy_sparse_matrix(edge_index, norm_A, x.size(node_dim))
    Matrix_norm_A = sparse_mx_to_torch_sparse_tensor(Matrix_norm_A)
    blank_noise = torch.randn_like(x)*1e-7
    x = x + blank_noise
    last_h = x / torch.clamp((torch.norm(x,dim=0)), 1e-8)
    list_mat.append(last_h)
    second_last_h = torch.zeros_like(last_h)
    for _ in range(1, K+1):
        h_i = torch.spmm(Matrix_norm_A, last_h)
        _t = torch.einsum('nh,nh->h',h_i,last_h)
        h_i = h_i - torch.einsum('h,nh->nh', _t, last_h)
        _t = torch.einsum('nh,nh->h',h_i,second_last_h)
        h_i = h_i - torch.einsum('h,nh->nh', _t, second_last_h)
        h_i = h_i / torch.clamp((torch.norm(h_i,dim=0)),1e-8)
        list_mat.append(h_i)
        second_last_h = last_h
        last_h = h_i
    return list_mat

def jacobi_base(
    K,
    x,
    edge_index,
    edge_attr,
    alpha=0.5,
    beta=0.5,
):
    """
    Jacobi polynomial basis:
        H_k = P_k^{(alpha, beta)}(L_tilde) X

    L_tilde = L - I
    spectrum(L_tilde) in [-1, 1]

    No eigendecomposition.
    Only sparse matrix multiplication.
    """

    node_dim = 0

    # --------------------------------------------------
    # L = I - D^{-1/2} A D^{-1/2}
    # --------------------------------------------------
    edge_index_L, norm_L = get_laplacian(
        edge_index,
        edge_attr,
        normalization='sym',
        dtype=x.dtype,
        num_nodes=x.size(node_dim),
    )

    # --------------------------------------------------
    # L_tilde = L - I
    # --------------------------------------------------
    edge_index_tilde, norm_tilde = add_self_loops(
        edge_index_L,
        norm_L,
        fill_value=-1.0,
        num_nodes=x.size(node_dim),
    )

    L_tilde = to_scipy_sparse_matrix(
        edge_index_tilde,
        norm_tilde,
        x.size(node_dim),
    )

    L_tilde = sparse_mx_to_torch_sparse_tensor(
        L_tilde
    ).to(x.device)

    # --------------------------------------------------
    # P_0
    # --------------------------------------------------
    list_mat = []

    P0 = x
    list_mat.append(P0)

    if K == 0:
        return list_mat

    # --------------------------------------------------
    # P_1
    #
    # P1(x) =
    # 1/2 [(alpha-beta) + (alpha+beta+2)x]
    # --------------------------------------------------
    Lx = torch.spmm(
        L_tilde,
        x,
    )

    P1 = 0.5 * (
        (alpha - beta) * x
        +
        (alpha + beta + 2.0) * Lx
    )

    list_mat.append(P1)

    # --------------------------------------------------
    # Jacobi recurrence
    # --------------------------------------------------
    Pkm1 = P0
    Pk = P1

    for k in range(1, K):

        kf = float(k)

        L_Pk = torch.spmm(
            L_tilde,
            Pk,
        )

        a1 = (
            2.0
            * (kf + 1.0)
            * (kf + alpha + beta + 1.0)
            * (2.0 * kf + alpha + beta)
        )

        a2 = (
            2.0 * kf
            + alpha
            + beta
            + 1.0
        )

        a3 = (
            2.0 * kf
            + alpha
            + beta
        ) * (
            2.0 * kf
            + alpha
            + beta
            + 2.0
        )

        a4 = (
            alpha ** 2
            - beta ** 2
        )

        a5 = (
            2.0
            * (kf + alpha)
            * (kf + beta)
            * (
                2.0 * kf
                + alpha
                + beta
                + 2.0
            )
        )

        Pkp1 = (
            a2 * (
                a3 * L_Pk
                +
                a4 * Pk
            )
            -
            a5 * Pkm1
        ) / a1

        list_mat.append(Pkp1)

        Pkm1 = Pk
        Pk = Pkp1

    assert len(list_mat) == K + 1

    return list_mat


def legendre_base(
    K,
    x,
    edge_index,
    edge_attr,
):
    """
    Legendre polynomial basis.

    H_k = P_k(L_tilde) X
    """

    node_dim = 0

    edge_index_L, norm_L = get_laplacian(
        edge_index,
        edge_attr,
        normalization='sym',
        dtype=x.dtype,
        num_nodes=x.size(node_dim),
    )

    # L_tilde = L - I
    edge_index_tilde, norm_tilde = add_self_loops(
        edge_index_L,
        norm_L,
        fill_value=-1.0,
        num_nodes=x.size(node_dim),
    )

    L_tilde = to_scipy_sparse_matrix(
        edge_index_tilde,
        norm_tilde,
        x.size(node_dim),
    )

    L_tilde = sparse_mx_to_torch_sparse_tensor(
        L_tilde
    ).to(x.device)

    list_mat = []

    # P0 = 1
    P0 = x
    list_mat.append(P0)

    if K == 0:
        return list_mat

    # P1 = x
    P1 = torch.spmm(
        L_tilde,
        x,
    )

    list_mat.append(P1)

    # --------------------------------------------------
    # P_{k+1}
    # =
    # (2k+1)/(k+1) x P_k
    # -
    # k/(k+1) P_{k-1}
    # --------------------------------------------------
    Pkm1 = P0
    Pk = P1

    for k in range(1, K):

        L_Pk = torch.spmm(
            L_tilde,
            Pk,
        )

        Pkp1 = (
            ((2.0 * k + 1.0) / (k + 1.0))
            * L_Pk
            -
            (k / (k + 1.0))
            * Pkm1
        )

        list_mat.append(Pkp1)

        Pkm1 = Pk
        Pk = Pkp1

    assert len(list_mat) == K + 1

    return list_mat

def ppr_base(
    K,
    x,
    edge_index,
    edge_attr,
    alpha=0.1,
):
    """
    PPR / APPNP-style spectral propagation.

    No eigendecomposition.
    """

    edge_index_norm, norm = gcn_norm(
        edge_index,
        edge_attr,
        num_nodes=x.size(0),
        dtype=x.dtype,
    )

    adj = to_scipy_sparse_matrix(
        edge_index_norm,
        norm,
        x.size(0),
    )

    adj = sparse_mx_to_torch_sparse_tensor(
        adj
    ).to(x.device)

    list_mat = []

    x0 = x
    h = x

    list_mat.append(h)

    for _ in range(K):

        h = (
            (1.0 - alpha)
            * torch.spmm(
                adj,
                h,
            )
            +
            alpha * x0
        )

        list_mat.append(h)

    assert len(list_mat) == K + 1

    return list_mat

def heat_base(
    K,
    x,
    edge_index,
    edge_attr,
    tau=0.5,
):
    """
    Discrete heat diffusion basis.

    H_k = (I - tau L)^k X

    For normalized Laplacian:
        lambda in [0, 2]

    tau <= 0.5 gives a non-negative low-pass response.
    """

    node_dim = 0

    edge_index_L, norm_L = get_laplacian(
        edge_index,
        edge_attr,
        normalization='sym',
        dtype=x.dtype,
        num_nodes=x.size(node_dim),
    )

    L = to_scipy_sparse_matrix(
        edge_index_L,
        norm_L,
        x.size(node_dim),
    )

    L = sparse_mx_to_torch_sparse_tensor(
        L
    ).to(x.device)

    list_mat = []

    h = x
    list_mat.append(h)

    for _ in range(K):

        Lh = torch.spmm(
            L,
            h,
        )

        h = h - tau * Lh

        list_mat.append(h)

    assert len(list_mat) == K + 1

    return list_mat

def diffusion_wavelet_base(
    K,
    x,
    edge_index,
    edge_attr,
    tau=0.5,
):
    """
    Diffusion-wavelet / diffusion-residual basis.

    H_0 = X

    H_k =
        tau L (I - tau L)^(k-1) X

    for k >= 1.

    No eigendecomposition.
    """

    node_dim = 0

    edge_index_L, norm_L = get_laplacian(
        edge_index,
        edge_attr,
        normalization='sym',
        dtype=x.dtype,
        num_nodes=x.size(node_dim),
    )

    L = to_scipy_sparse_matrix(
        edge_index_L,
        norm_L,
        x.size(node_dim),
    )

    L = sparse_mx_to_torch_sparse_tensor(
        L
    ).to(x.device)

    list_mat = []

    # Keep original signal as H_0
    list_mat.append(x)

    smooth = x

    for _ in range(K):

        L_smooth = torch.spmm(
            L,
            smooth,
        )

        next_smooth = (
            smooth
            -
            tau * L_smooth
        )

        # Difference between two diffusion scales
        wavelet = (
            smooth
            -
            next_smooth
        )

        list_mat.append(
            wavelet
        )

        smooth = next_smooth

    assert len(list_mat) == K + 1

    return list_mat

def laguerre_base(
    K,
    x,
    edge_index,
    edge_attr,
):
    """
    Laguerre polynomial basis.

    H_k = L_k(L) X
    """

    node_dim = 0

    edge_index_L, norm_L = get_laplacian(
        edge_index,
        edge_attr,
        normalization='sym',
        dtype=x.dtype,
        num_nodes=x.size(node_dim),
    )

    L = to_scipy_sparse_matrix(
        edge_index_L,
        norm_L,
        x.size(node_dim),
    )

    L = sparse_mx_to_torch_sparse_tensor(
        L
    ).to(x.device)

    list_mat = []

    # L_0(lambda) = 1
    H0 = x
    list_mat.append(H0)

    if K == 0:
        return list_mat

    # L_1(lambda) = 1 - lambda
    H1 = x - torch.spmm(
        L,
        x,
    )

    list_mat.append(H1)

    Hkm2 = H0
    Hkm1 = H1

    for k in range(2, K + 1):

        L_Hkm1 = torch.spmm(
            L,
            Hkm1,
        )

        Hk = (
            (2.0 * k - 1.0)
            * Hkm1
            -
            L_Hkm1
            -
            (k - 1.0)
            * Hkm2
        ) / k

        list_mat.append(Hk)

        Hkm2 = Hkm1
        Hkm1 = Hk

    assert len(list_mat) == K + 1

    return list_mat

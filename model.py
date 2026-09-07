from torch import nn
import torch.nn.functional as F
import random
import torch
from torch.nn import Parameter, Linear, ModuleList, LeakyReLU
import math
from torch_geometric.nn import MessagePassing
import numpy as np
from torch import Tensor
#from sim import neighborhood_similarity

def init_params(module, n_layers):
    """Initialize Linear/Embedding layers."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(max(n_layers, 1)))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, bias=True):
        super().__init__()
        if n_layers == 1:
            self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=bias)])
        else:
            dims = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
            self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(n_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
            nn.init.orthogonal_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(self, x: Tensor) -> Tensor:
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.gelu(x)
        x = self.lins[-1](x)
        return x

class GraphViewTransformation(nn.Module):
    def __init__(self, in_C: int = 32, hid_C: int = 32, vt_depth: int = 1):
        super().__init__()      
        self.in_C = in_C
        self.hid_C = hid_C
        self.vt_depth = vt_depth

        self._init_parameters()
        self.reset_parameters()

    def _init_parameters(self):
        self.mlp = MLP(self.in_C, self.hid_C, 1, self.vt_depth)

    def reset_parameters(self):
        with torch.no_grad():
            self.mlp.reset_parameters()
    def forward(self, X):
        N, Fdim, C = X.shape  # [N, Q, C]
        X = X.reshape(-1, C)  # [N*Q, C]
        x_input = self.mlp(X)  # [N*Q, 1]
        x_input = x_input.view(N, Fdim)  # [N, Q]
        return x_input



class SGAD(nn.Module):
    def __init__(self, 
        input_dim,
        h_feats,
        n_layers,
        alpha, 
        beta,         
        hidden_dim,
        num_queries,   
        dropout,     
        high_homophily_threshold,
        low_homophily_threshold,
        activation='ReLU',  **kwargs):
        super(SGAD, self).__init__()

        self.input_dim = input_dim
        self.h_feats = h_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_queries = num_queries
        self.alpha = alpha
        self.beta = beta
        self.Init ='PPR'

        # Homophily-guided prototype contrastive learning
        self.high_homophily_threshold = high_homophily_threshold
        self.low_homophily_threshold = low_homophily_threshold
        self.contrastive_tau = 0.2
        self.score_tau = 1.0
        
        
        
        self.feature_attention = nn.Sequential(
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
                bias=False,
            ),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(
                self.hidden_dim,
                self.num_queries,
                bias=False,
            ),
        )  

        self.share_encoder = nn.Sequential(
            Linear(
                self.num_queries,
                self.h_feats,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Linear(
                self.h_feats,
                self.num_queries,
                bias=False,
            ),
            nn.LayerNorm(num_queries),
        )  #[K+1 to H]
        self.adaptive_fuse = Adaptive_fusion(self.input_dim, self.alpha, self.Init)

        #self.trans = GraphViewTransformation(self.input_dim, self.hidden_dim)
        self.MPNN = TwoStageGCN(self.num_queries, self.n_layers, self.dropout)

        # Two graph-agnostic class prototypes in the shared [R]-dimensional space.
        # R == self.num_queries, because both views have shape [N, R].
        self.normal_proto = Parameter(torch.empty(self.num_queries))
        self.anomaly_proto = Parameter(torch.empty(self.num_queries))

        self.apply(lambda module: init_params(module, n_layers=self.n_layers))
        self.reset_parameters()
        self.adaptive_fuse.reset_parameters()
        


    def reset_parameters(self) -> None:
        """
        初始化可学习Query。
        """
#        nn.init.xavier_uniform_(self.queries)
        nn.init.normal_(self.normal_proto, mean=0.0, std=0.02)
        nn.init.normal_(self.anomaly_proto, mean=0.0, std=0.02)

        self.prototype_initialized = False


    def forward(self, graph):
        #print(input_mat.shape)
        self.graph = graph
        x_list = graph.x_list
        input_mat = torch.stack(x_list, dim=2)  # [N, d, k]
        adj_norm = graph.adj_norm
        adj_hat = graph.adj_hat
        
        #input_mat = self.share_encoder(input_mat) # just for common dataset # [N,d,H]

        #input_mat = F.dropout(input_mat, self.dropout, training=self.training)
        
        attention_weights = self.feature_attention(input_mat)

        query_embeddings = torch.einsum(
            "ndf,ndh->nfh",
            attention_weights,
            input_mat,
        )
        
        #x_input = self.trans(query_embeddings) # [N, R]
        x_input = self.adaptive_fuse(query_embeddings)# [N, Q]
        
        x_input = self.share_encoder(x_input)# [N, H]
        encoder_output, x_output = self.MPNN(x_input, adj_norm, adj_hat)  # [N, H]
        #score_rec = self.get_rec_score(x_input, x_output)
        score,score_proto=  self.get_test_score(
            graph,
            x_input,
            encoder_output,
            x_output,
            self.beta,
        )
        #score = score_trs + score_rec
        return {
            'raw':x_input,
            "latent": encoder_output,      # view 1: [N, R]
            "output": x_output,            # view 2: [N, R]
            "score": score,                # [N], larger means more anomalous
            #"score_unsim": score_unsim,               
            #"sim": loc_sim, 
            'score_pro':score_proto
        }

    def _neighbor_mean(self, embeddings, edge_index):
        """
        Compute each node's mean 1-hop neighborhood representation directly
        from edge_index, without constructing a dense N x N adjacency matrix.

        embeddings: [N, R]
        edge_index: [2, E]
        """
        edge_index = edge_index.to(embeddings.device)
        src = edge_index[0]
        dst = edge_index[1]

        # Self-loops should not be used as neighborhood targets.
        non_self = src != dst
        src = src[non_self]
        dst = dst[non_self]

        num_nodes = embeddings.size(0)
        neighbor_sum = torch.zeros_like(embeddings)

        # Treat src as the center and dst as its neighbor. For an undirected
        # edge_index containing both directions this gives the standard 1-hop mean.
        neighbor_sum.index_add_(0, src, embeddings[dst])

        degree = torch.zeros(
            num_nodes,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        degree.index_add_(
            0,
            src,
            torch.ones(
                src.numel(),
                device=embeddings.device,
                dtype=embeddings.dtype,
            ),
        )

        neighbor_embeddings = (
            neighbor_sum / degree.clamp_min(1.0).unsqueeze(-1)
        )
        return neighbor_embeddings, degree
    
    @torch.no_grad()
    def initialize_prototypes(
        self,
        view1_embeddings,
        view2_embeddings,
        labels,
    ):

        labels = labels.view(-1).long()

        embeddings = 0.5 * (
        view1_embeddings.detach()
        + 
        view2_embeddings.detach()
        )

        embeddings = F.normalize(
        embeddings,
        dim=-1
        )

        normal_center = embeddings[
        labels == 0
        ].mean(dim=0)

        anomaly_center = embeddings[
        labels == 1
        ].mean(dim=0)

        normal_center = F.normalize(
            normal_center,
            dim=0
        )

        anomaly_center = F.normalize(
            anomaly_center,
            dim=0
        )

        self.normal_proto.copy_(
            normal_center
        )

        self.anomaly_proto.copy_(
            anomaly_center
        )

        self.prototype_initialized = True

    
    def _pair_contrastive_loss(
        self,
        anchor,
        positive,
        negative,
        tau=None,
    ):
        """
        Binary InfoNCE with one positive and one negative:

            -log exp(sim_pos/tau) /
                 [exp(sim_pos/tau) + exp(sim_neg/tau)]

        The numerically stable equivalent is
        softplus((sim_neg - sim_pos) / tau).
        """
        if tau is None:
            tau = self.contrastive_tau

        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        sim_pos = torch.sum(anchor * positive, dim=-1)
        sim_neg = torch.sum(anchor * negative, dim=-1)

        return F.softplus((sim_neg - sim_pos) / tau).mean()

    def get_train_loss(
        self,
        graph,
        #fuseFeat,
        view1_embeddings,
        view2_embeddings,
        labels,
        tau=None,
    ):
        """
        Homophily-guided prototype-neighborhood contrastive loss.

        view1_embeddings, view2_embeddings: [N, R]

        1) normal  + high homophily:
           positive = neighborhood, negative = anomaly prototype
        2) normal  + low homophily:
           positive = normal prototype, negative = neighborhood
        3) anomaly + high homophily:
           positive = neighborhood, negative = normal prototype
        4) anomaly + low homophily:
           positive = anomaly prototype, negative = neighborhood

        Both current-view and cross-view neighborhoods are used. Nodes with
        homophily in [low_threshold, high_threshold] are ignored.
        """
        if tau is None:
            tau = self.contrastive_tau

        z = view1_embeddings
        h = view2_embeddings

        if z.dim() != 2 or h.dim() != 2:
            raise ValueError(
                "view1_embeddings and view2_embeddings must both be [N, R]."
            )
        if z.shape != h.shape:
            raise ValueError(
                f"The two views must have the same shape, got {z.shape} and {h.shape}."
            )
        if z.size(1) != self.num_queries:
            raise ValueError(
                f"Expected representation dimension R={self.num_queries}, "
                f"but got {z.size(1)}."
            )

        edge_index = graph.edge_index.to(z.device)
        labels = graph.ano_labels.to(z.device).view(-1).long()
        homophily = graph.homophily.to(z.device).view(-1).float()

        neighbor_z, degree = self._neighbor_mean(z, edge_index)
        neighbor_h, _ = self._neighbor_mean(h, edge_index)
        
        if not self.prototype_initialized:
            self.initialize_prototypes(
              #  fuseFeat,
                view1_embeddings,
                view2_embeddings,
                labels,
            )

        p_n = F.normalize(self.normal_proto, p=2, dim=0)
        p_a = F.normalize(self.anomaly_proto, p=2, dim=0)
        p_n_all = p_n.unsqueeze(0).expand(z.size(0), -1)
        p_a_all = p_a.unsqueeze(0).expand(z.size(0), -1)

        valid = (degree > 0) 
        high = homophily >= self.high_homophily_threshold
        low = (homophily >= 0.0) & (homophily < self.low_homophily_threshold)
        
        
        normal = (labels == 0)
        anomaly = (labels == 1) 

        high_normal = valid & high & normal
        low_normal = valid & low & normal
        high_anomaly = valid & high & anomaly
        low_anomaly = valid & low & anomaly

        num_high_normal = high_normal.sum().item()
        num_low_normal = low_normal.sum().item()
        num_high_anomaly = high_anomaly.sum().item()
        num_low_anomaly = low_anomaly.sum().item()

        num_nomal_total = num_high_normal + num_low_normal #+ num_high_anomaly + num_low_anomaly
        #+ num_low_normal + num_high_anomaly + num_low_anomaly
        num_abnomal_total = num_high_anomaly + num_low_anomaly

        group_losses = []
        group_counts = []
        '''
        # High-homophily normal: neighborhood (+), anomaly prototype (-).
        if high_normal.any():
            m = high_normal
            loss_hn = (num_high_normal/num_nomal_total) * (
                self._pair_contrastive_loss(z[m], neighbor_z[m], p_a_all[m], tau)
                + self._pair_contrastive_loss(z[m], neighbor_h[m], p_a_all[m], tau)
                + self._pair_contrastive_loss(h[m], neighbor_h[m], p_a_all[m], tau)
                + self._pair_contrastive_loss(h[m], neighbor_z[m], p_a_all[m], tau)
            )
            group_losses.append(loss_hn)
        
        
        # Low-homophily normal: normal prototype (+), neighborhood (-).
        if low_normal.any():
            m = low_normal
            loss_ln = (num_low_normal/num_nomal_total) * (
                self._pair_contrastive_loss(z[m], p_n_all[m], neighbor_z[m], tau)
                + self._pair_contrastive_loss(z[m], p_n_all[m], neighbor_h[m], tau)
                + self._pair_contrastive_loss(h[m], p_n_all[m], neighbor_h[m], tau)
                + self._pair_contrastive_loss(h[m], p_n_all[m], neighbor_z[m], tau)
            )
            group_losses.append(loss_ln)
            #group_counts.append(m.sum().float())
        '''
        # High-homophily anomaly: neighborhood (+), normal prototype (-).
        if high_anomaly.any():
            m = high_anomaly
            loss_ha = (num_high_anomaly/num_abnomal_total) * (
                self._pair_contrastive_loss(z[m], neighbor_z[m], p_n_all[m], tau)
                + self._pair_contrastive_loss(z[m], neighbor_h[m], p_n_all[m], tau)
                + self._pair_contrastive_loss(h[m], neighbor_h[m], p_n_all[m], tau)
                + self._pair_contrastive_loss(h[m], neighbor_z[m], p_n_all[m], tau)
            )
            group_losses.append(loss_ha)
            #group_counts.append(m.sum().float())
        
        # Low-homophily anomaly: anomaly prototype (+), neighborhood (-).
        if low_anomaly.any():
            m = low_anomaly
            loss_la = (num_low_anomaly/num_abnomal_total) * (
                self._pair_contrastive_loss(z[m], p_a_all[m], neighbor_z[m], tau)
                +self._pair_contrastive_loss(z[m], p_a_all[m], neighbor_h[m], tau)
                + self._pair_contrastive_loss(h[m], p_a_all[m], neighbor_h[m], tau)
                + self._pair_contrastive_loss(h[m], p_a_all[m], neighbor_z[m], tau)
            )
            group_losses.append(loss_la)
            #group_counts.append(m.sum().float())
        
        if len(group_losses) == 0:
            # Keep the returned zero connected to autograd.
            return (z.sum() + h.sum()) * 0.0

        group_losses = torch.stack(group_losses)# [G]
        '''
        nz = F.normalize(
            z,
            p=2,
            dim=-1
        )

        nh = F.normalize(
            h,
            p=2,
            dim=-1
        )

        sim_zz = F.cosine_similarity(
            nz,
            neighbor_z,
            dim=-1
        )

        sim_zh = F.cosine_similarity(
            nz,
            neighbor_h,
            dim=-1
        )

        sim_hh = F.cosine_similarity(
            nh,
            neighbor_h,
            dim=-1
        )

        sim_hz = F.cosine_similarity(
            nh,
            neighbor_z,
            dim=-1
        )

        sim = (sim_zz + sim_hh + sim_zh + sim_hz)/4
        tau = 0.2

        local_similarity = torch.sigmoid(sim/tau)
        '''
        loss_sep = F.relu(torch.sum(p_n * p_a))
        lambda_sim = 1.0
        
        
        #loss_a1 = F.binary_cross_entropy(local_similarity[low_anomaly], torch.zeros_like(local_similarity[low_anomaly]))
        #loss_a2 = F.binary_cross_entropy(local_similarity[low_normal], torch.zeros_like(local_similarity[low_normal]))

        #loss_b1 = F.binary_cross_entropy(local_similarity[high_anomaly], torch.ones_like(local_similarity[high_anomaly]))
        #loss_b2 = F.binary_cross_entropy(local_similarity[high_normal], torch.ones_like(local_similarity[high_normal]))

        #sim_loss = loss_a1  + loss_b1 
        
        lambda_sep = 2.0

        
        loss = (
            group_losses.mean()
            +  lambda_sep *loss_sep 
           # + lambda_sim * sim_loss
        )

        return loss
        

    def get_test_score(
        self,
        graph,
        raw,
        view1_embeddings,
        view2_embeddings,
        beta,
        tau=None,
        lambda_local=0.9,
        ):
        """
        Prototype + neighborhood-pattern anomaly score.

        Test time does NOT use labels or homophily.

        High-hom anomaly:
        node anomaly-like
        + neighborhood anomaly-like
        + node-neighborhood similar

        Low-hom anomaly:
        node anomaly-like
        + neighborhood normal-like
        + node-neighborhood dissimilar
        """

        if tau is None:
            tau = self.score_tau

        z = view1_embeddings
        h = view2_embeddings

        if z.dim() != 2 or h.dim() != 2:
            raise ValueError(
            "view1_embeddings and view2_embeddings "
            "must both be [N, R]."
            )

        if z.shape != h.shape:
            raise ValueError(
            f"The two views must have the same shape, "
            f"got {z.shape} and {h.shape}."
            )

        # =====================================================
        # 1. Normalize representations
        # =====================================================

        z = F.normalize(
            z,
            p=2,
            dim=-1
        )

        h = F.normalize(
            h,
            p=2,
            dim=-1
        )

        p_n = F.normalize(
            self.normal_proto,
            p=2,
            dim=0
        )

        p_a = F.normalize(
            self.anomaly_proto,
            p=2,
            dim=0
        )

        # =====================================================
        # 2. Node prototype anomaly probability
        # =====================================================

        sim_zn = torch.matmul(z,p_n)
        #sim_zn = F.cosine_similarity(z,p_n,dim=-1)
        sim_za = torch.matmul(z,p_a)

        sim_hn = torch.matmul(h,p_n)
        #sim_hn = F.cosine_similarity(h,p_n,dim=-1)
        sim_ha = torch.matmul(h,p_a)
        
        abnormal_score_z = (1.0 - sim_zn) / 2.0

        abnormal_score_h = (1.0 - sim_hn) / 2.0


        score_proto =  (abnormal_score_z + abnormal_score_h)/2
        
        #normal_proto = 0.5*(normal_score_z+normal_score_h)

        #score_proto = score_proto/(score_proto+normal_proto)
        # =====================================================
        # 3. Neighborhood representations
        # =====================================================

        edge_index = graph.edge_index.to(
            z.device
        )

        neighbor_z, degree = self._neighbor_mean(
            z,
            edge_index
        )

        neighbor_h, _ = self._neighbor_mean(
            h,
            edge_index
        )

        neighbor_z = F.normalize(
            neighbor_z,
            p=2,
            dim=-1
        )

        neighbor_h = F.normalize(
            neighbor_h,
            p=2,
            dim=-1
        )

        # =====================================================
        # 4. Node-neighborhood similarity
        #
        # Same four relations as training loss
        # =====================================================

        sim_zz = F.cosine_similarity(
            z,
            neighbor_z,
            dim=-1
        )

        sim_zh = F.cosine_similarity(
            z,
            neighbor_h,
            dim=-1
        )

        sim_hh = F.cosine_similarity(
            h,
            neighbor_h,
            dim=-1
        )

        sim_hz = F.cosine_similarity(
            h,
            neighbor_z,
            dim=-1
        )

        sim = (sim_zz + sim_hh + sim_zh + sim_hz)/4

        tau = 0.2
        local_similarity = torch.sigmoid(sim/tau)


        nbr_sim_zn = torch.matmul(
            neighbor_z,
            p_n
        )

        nbr_sim_za = torch.matmul(
            neighbor_z,
            p_a
        )

        nbr_sim_hn = torch.matmul(
            neighbor_h,
            p_n
        )

        nbr_sim_ha = torch.matmul(
            neighbor_h,
            p_a
        )
        nbr_score_z = (1.0 - nbr_sim_zn) / 2.0
        nbr_score_h = (1.0 - nbr_sim_hn) / 2.0
        
        #nbr_score_z = torch.exp(nbr_sim_za / 0.5) + torch.exp(-nbr_sim_zn / 0.5)
        #nbr_score_h = torch.exp(nbr_sim_ha / 0.5) + torch.exp(-nbr_sim_hn / 0.5) 
                
        neighbor_anomaly_prob = 0.5*(nbr_score_z + nbr_score_h)
        
        high_anomaly_evidence = (#效果如下：reddit：0.5346，tfinance：0.5898，tolokers：0.5588
            neighbor_anomaly_prob
            * local_similarity
        )

        # Low-hom anomaly:
        # node anomaly + neighbor normal + low similarity
        low_anomaly_evidence = (# 效果如下：Facebook:0.9115; Elliptic:0.6299;Disney:0.7386;Book:0.5406;Yelp:0.7310
            (1.0 - neighbor_anomaly_prob)
            * (1.0 - local_similarity)
        )

        abnormal_evidence = (
            score_proto
            * (high_anomaly_evidence + 0.1 * low_anomaly_evidence)
        )

        #score_a = abnormal_evidence 
        
        score = torch.where(
            degree > 0,
            abnormal_evidence,
            score_proto,
        )  

        return score, 1.0 - local_similarity




class Adaptive_fusion(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(Adaptive_fusion, self).__init__(aggr='add', **kwargs)
        self.K = K

        
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K
        #self.temp.data.fill_(1)

    def forward(self, x):
        #TEMP=torch.tanh(self.temp)
        matrices = torch.unbind(x, dim=-1)
        hidden = torch.zeros_like(matrices[0])
        #hidden = matrices[0]*(self.temp[0])
        for h, matrix in enumerate(matrices):
            #x = self.propagate(edge_index, x=x, norm=norm)
            gamma = (self.temp[h])
            hidden = hidden + gamma * matrix
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class TwoStageGCN(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_layers=4,
        dropout=0.0,
    ):
        super().__init__()

        assert num_layers % 2 == 0

        self.half_layers = num_layers // 2
        self.dropout = dropout

        self.encoder_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(self.half_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(self.half_layers)
        ])

        
    def propagate(self, h, adj):
        if adj.is_sparse:
            return torch.sparse.mm(adj, h)
        return adj @ h

    def encode(self, x, adj1):
        h = x
        #h0 = x

        for layer_id, linear in enumerate(
            self.encoder_layers
        ):
            h = self.propagate(
                h,
                adj1,
            )
            h = linear(h)
            h = F.relu(h)
        
        return h

    def decode(self, z,x, adj2):
        h = z
        #h0 = x

        for layer_id, linear in enumerate(
            self.decoder_layers
        ):
            
            h = self.propagate(
                h,
                adj2,
            )
            h = linear(h)
            if layer_id != self.half_layers - 1:
                h = F.relu(h)

        return h

    def forward(self, x, adj1, adj2):
        adj1 =adj1.to(x.device)
        adj2 =adj2.to(x.device)
        
        z = self.encode(
            x,
            adj1,
        )  # [N, D]

        x_hat = self.decode(
            z,
            x,
            adj2,
        )  # [N, D]


        return z, x_hat

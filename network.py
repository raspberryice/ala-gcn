import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCNLayer,GATLayer, GatedLayer, GatedAttnLayer, GraphTopoAttention,GatedAPPNPConv
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv, APPNPConv
from dgl.nn.pytorch import GraphConv
from math import sqrt
import scipy.sparse as sp
import numpy as np


class AdaptiveGAT(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes,
        n_layers, activation, dropout, heads, attn_drop, negative_slope, shared_tau=True):
        super(AdaptiveGAT, self).__init__()
        self.g = g
        self.n_classes = n_classes
        # self.weight_y = nn.Linear(n_hidden*heads[0], n_classes*heads[0])
        self.weight_y = nn.Linear(n_hidden*heads[0], n_classes)

        self.global_tau_1 = nn.Parameter(torch.zeros((1,)))
        self.global_tau_2 = nn.Parameter(torch.zeros((1,)))
        self.shared_tau = shared_tau

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, heads[0], n_hidden)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, heads[0], n_hidden)))


        self.layers = nn.ModuleList()
        self.heads = heads
        self.n_hidden = n_hidden
        self.layers.append(GATLayer(in_feats, n_hidden, heads[0], dropout,  attn_drop,
                negative_slope,
                 residual=False,
                 activation=None))

        for i in range(n_layers-1):
            self.layers.append(GatedAttnLayer(g, n_hidden, n_hidden, None, dropout, heads[i],
                 attn_drop,
                 negative_slope,
                 i+1))


    def init_weight(self, feats, labels):
        # initial weight_y is obtained by linear regression
        A = torch.mm(feats.t(), feats) + 1e-05 * torch.eye(feats.size(1))
        # (feats, feats)

        labels_one_hot = torch.zeros((feats.size(0), self.n_classes))
        for i in range(labels.size(0)):
            l = labels[i]
            labels_one_hot[i,l] = 1

        self.init_weight_y = nn.Parameter(torch.mm(torch.mm(torch.cholesky_inverse(A),feats.t()),labels_one_hot),requires_grad=False)
        nn.init.constant_(self.global_tau_1, 1/2)
        nn.init.constant_(self.global_tau_2, 1/2)

        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        return

    def forward(self, features):
        z = torch.FloatTensor([1.0,]).cuda()
        h = features
        list_z = []
        for lidx, layer in enumerate(self.layers):
            if lidx == 0:
                # first layer use initial weight_y
                logits = F.softmax(torch.mm(h,self.init_weight_y), dim=1)
                h = layer(self.g, h )
            else:
                logits = F.softmax(self.weight_y(h.reshape(h.size(0), -1)), dim=1)
                h, z = layer(self.g, h, logits, old_z=z,
                    shared_tau=self.shared_tau, tau_1=self.global_tau_1, tau_2=self.global_tau_2,
                     attn_l = self.attn_l, attn_r = self.attn_r )
                list_z.append(z.flatten())

        output = self.weight_y(h.reshape(h.size(0), -1))
        all_z = torch.stack(list_z, dim=1) # (n_nodes, n_layers)
        return output, all_z


class AdaptiveGNN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes,
        n_layers, activation, dropout, shared_tau=True):
        super(AdaptiveGNN, self).__init__()
        self.g = g
        self.n_classes = n_classes
        self.weight_y = nn.Linear(n_hidden, n_classes)

        self.global_tau_1 = nn.Parameter(torch.zeros((1,)))
        self.global_tau_2 = nn.Parameter(torch.zeros((1,)))
        self.shared_tau = shared_tau

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_feats, n_hidden, None, 0.))
        for i in range(n_layers-1):
            self.layers.append(GatedLayer(g, n_hidden, n_hidden, None, dropout,i+1))

    def init_weight(self, feats, labels):
        # initial weight_y is obtained by linear regression
        A = torch.mm(feats.t(), feats) + 1e-05 * torch.eye(feats.size(1))
        # (feats, feats)

        labels_one_hot = torch.zeros((feats.size(0), self.n_classes))
        for i in range(labels.size(0)):
            l = labels[i]
            labels_one_hot[i,l] = 1

        self.init_weight_y = nn.Parameter(torch.mm(torch.mm(torch.cholesky_inverse(A),feats.t()),labels_one_hot),requires_grad=False)
        nn.init.constant_(self.global_tau_1, 1/2)
        nn.init.constant_(self.global_tau_2, 1/2)
        return

    def forward(self, features):
        z = torch.FloatTensor([1.0,]).cuda()
        h = features
        list_z = []
        for lidx, layer in enumerate(self.layers):
            if lidx == 0:
                # first layer use initial weight_y
                logits = F.softmax(torch.mm(h,self.init_weight_y), dim=1)
                h = layer(self.g, h)
            else:
                logits = F.softmax(self.weight_y(h), dim=1)
                h, z = layer(self.g, h, logits, old_z=z,
                    shared_tau=self.shared_tau, tau_1=self.global_tau_1, tau_2=self.global_tau_2)
                list_z.append(z)

        output = self.weight_y(h)
        all_z = torch.stack(list_z, dim=1) # (n_nodes, n_layers)
        return output, all_z


class AdaAPPNP(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes,
        n_layers, activation, dropout, edge_drop):
        super().__init__()
        self.g = g
        self.n_classes = n_classes
        self.feat_drop = nn.Dropout(p=dropout)
        self.propagate = GatedAPPNPConv(g, n_layers, n_hidden, n_classes,edge_drop)
        self.layers = nn.ModuleList()
        self.activation = activation
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
    def init_weight(self, feats, labels):
        # initial weight_y is obtained by linear regression
        A = torch.mm(feats.t(), feats) + 1e-05 * torch.eye(feats.size(1))
        # (feats, feats)

        labels_one_hot = torch.zeros((feats.size(0), self.n_classes))
        for i in range(labels.size(0)):
            l = labels[i]
            labels_one_hot[i,l] = 1

        self.init_weight_y = nn.Parameter(torch.mm(torch.mm(torch.cholesky_inverse(A),feats.t()),labels_one_hot),requires_grad=False)
        for layer in self.layers:
            gain = nn.init.calculate_gain('relu')
            torch.nn.init.xavier_uniform_(layer.weight, gain=gain)

        return

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.layers[0](h) # no activation, so only one FFN layer
        if self.activation:
            h = self.activation(h)
        # propagation step
        logits = F.softmax(torch.mm(features,self.init_weight_y), dim=1)
        h = self.propagate(self.g, h, logits)
        return h




class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        # self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, features):
        h = features
        for i,layer in enumerate(self.layers):

            if i!=0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATLayer(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATLayer(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATLayer(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class SGC(nn.Module):
    r"""Simplifying Graph Convolution layer from paper `Simplifying Graph
    Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`__.

    .. math::
        H^{l+1} = (\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2})^K H^{l} \Theta^{l}

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    k : int
        Number of hops :math:`K`. Defaults:``1``.
    cached : bool
        If True, the module would cache

        .. math::
            (\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}})^K X\Theta

        at the first forward call. This parameter should only be set to
        ``True`` in Transductive Learning setting.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    """
    def __init__(self,graph,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None):
        super(SGC, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.graph = graph
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, feat):
        r"""Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Notes
        -----
        If ``cache`` is se to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        if self._cached_h is not None:
            feat = self._cached_h
        else:
            # compute normalization
            degs = self.graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)
            # compute (D^-1 A^k D)^k X
            for _ in range(self._k):
                feat = feat * norm
                self.graph.ndata['h'] = feat
                self.graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feat = self.graph.ndata.pop('h')
                feat = feat * norm

            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat
        return self.fc(feat)




class GTN(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 feats_d,
                 feats_t_d,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 residual,
                 concat):
        super(GTN, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gtn_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gtn_layers.append(GraphTopoAttention(g, feats_d, feats_t_d, num_hidden, heads[0],
                                                feat_drop, attn_drop, False, concat))
        # hidden layers
        fix_d = concat*(feats_d)
        for l in range(1, num_layers+1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gtn_layers.append(GraphTopoAttention(g, num_hidden*(heads[l-1]+1*concat), feats_t_d,
                            num_hidden, heads[l], feat_drop, attn_drop, residual, concat))
        # output projection
        self.gtn_layers.append(GraphTopoAttention(g, num_hidden*(heads[l-1]+1*concat), feats_t_d,
                num_classes, heads[-1], feat_drop, attn_drop, residual, concat, True))

    def forward(self, inputs, topo=None):
        h = inputs
        if topo:
            t = F.normalize(topo)
        else:
            t = None

        for l in range(self.num_layers+1):
            h = self.gtn_layers[l](h, t)
            h = self.activation(h)
        # output projection
        logits = self.gtn_layers[-1](h, t)
        return logits



class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
        return h


class GResNet(nn.Module):
    def __init__(self, g, nfeat, nhid, nclass, dropout, depth, residual_type):
        super(GResNet, self).__init__()
        self.graph = g
        self.depth = depth
        self.gc_list = nn.ModuleList()
        self.residual_weight_list = nn.ParameterList()
        self.residual_type = residual_type
        self.adj = nn.Parameter(g.adjacency_matrix(), requires_grad=False)


        if self.depth == 1:
            self.gc_list.append(GraphConv(nfeat, nclass) )
            self.residual_weight_list.append(nn.Parameter(torch.FloatTensor(nfeat, nclass)))
        else:
            for i in range(self.depth):
                if i == 0:
                    self.gc_list.append(GraphConv(nfeat, nhid))
                    self.residual_weight_list.append(nn.Parameter(torch.FloatTensor(nfeat, nhid)))
                elif i == self.depth - 1:
                    self.gc_list.append(GraphConv(nhid, nclass) )
                    self.residual_weight_list.append(nn.Parameter(torch.FloatTensor(nhid, nclass)))
                else:
                    self.gc_list.append(GraphConv(nhid, nhid))
                    self.residual_weight_list.append(nn.Parameter(torch.FloatTensor(nhid, nhid)))
        for i in range(self.depth):
            stdv = 1. / sqrt(self.residual_weight_list[i].size(1))
            self.residual_weight_list[i].data.uniform_(-stdv, stdv)
        self.dropout = dropout

    def forward(self, raw_x):
        if self.residual_type == 'naive':
            return self.forward_naive(raw_x)
        elif self.residual_type == 'raw':
            return self.forward_raw(raw_x)
        elif self.residual_type == 'graph_naive':
            return self.forward_graph_naive(raw_x)
        elif self.residual_type == 'graph_raw':
            return self.forward_graph_raw(raw_x)

    #---- raw residual ----
    def forward_raw(self, raw_x):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](self.graph, x) + torch.mm(raw_x, self.residual_weight_list[0]))
            x = F.dropout(x, self.dropout, training=self.training)
        if self.depth == 1:
            y = self.gc_list[self.depth - 1](self.graph, x) + torch.mm(raw_x, self.residual_weight_list[0])
        else:
            y = self.gc_list[self.depth-1](self.graph, x)+ torch.mm(torch.mm(raw_x, self.residual_weight_list[0]), self.residual_weight_list[self.depth-1])

        return y

    #---- naive residual ----
    def forward_naive(self, raw_x):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](self.graph, x) + torch.mm(x, self.residual_weight_list[i]))
            x = F.dropout(x, self.dropout, training=self.training)
        y = self.gc_list[self.depth-1](self.graph, x) + torch.mm(x, self.residual_weight_list[self.depth-1])

        return y

    #---- graph raw residual ----
    def forward_graph_raw(self, raw_x):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](self.graph, x) + torch.spmm(self.adj, torch.mm(raw_x, self.residual_weight_list[0])))
            x = F.dropout(x, self.dropout, training=self.training)
        if self.depth == 1:
            y = self.gc_list[self.depth - 1](self.graph, x)  + torch.spmm(self.adj, torch.mm(raw_x, self.residual_weight_list[0]))
        else:
            y = self.gc_list[self.depth-1](self.graph, x) + torch.spmm(self.adj, torch.mm(torch.mm(raw_x, self.residual_weight_list[0]), self.residual_weight_list[self.depth-1]))

        return y

    #---- graph naive residual ----
    def forward_graph_naive(self, raw_x):
        x = raw_x
        for i in range(self.depth-1):
            x = F.relu(self.gc_list[i](self.graph, x) + torch.spmm(self.adj, torch.mm(x, self.residual_weight_list[i])))
            x = F.dropout(x, self.dropout, training=self.training)
        y = self.gc_list[self.depth-1](self.graph, x) + torch.spmm(self.adj, torch.mm(x, self.residual_weight_list[self.depth-1]))

        return y


class GATResNet(nn.Module):
    def __init__(self, g, nfeat, nhid, nclass, dropout, alpha, nheads, depth, residual_type):
        """Sparse version of GAT."""
        nn.Module.__init__(self)
        self.g = g
        self.dropout = dropout
        self.depth = depth
        self.gat_list = nn.ModuleList()
        self.residual_weight_list = nn.ParameterList()
        self.residual_type = residual_type
        self.adj = nn.Parameter(g.adjacency_matrix(), requires_grad=False)
        self.norm_adj = nn.Parameter(self.adj_normalize(g.adjacency_matrix_scipy() + sp.eye(self.adj.shape[0])), requires_grad=False)

        if self.depth == 1:
            self.out_att = GATLayer(nfeat, nclass, 1, feat_drop=dropout, attn_drop=dropout, activation=F.leaky_relu)
            # self.out_att = GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=False)
            self.residual_weight_list.append(nn.Parameter(torch.FloatTensor(nfeat, nclass)))
        else:
            for depth_index in range(self.depth - 1):
                if depth_index == 0:
                    self.gat_list.append(GATLayer(nfeat, nhid, nheads, dropout, dropout, residual=True, activation=F.leaky_relu))
                    # self.gat_list[depth_index] = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
                    self.residual_weight_list.append(nn.Parameter(torch.FloatTensor(nfeat, nhid * nheads)))
                else:
                    # self.gat_list[depth_index] = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
                    self.gat_list.append(GATLayer(nhid* nheads, nhid, nheads, dropout, dropout, residual=True, activation=F.leaky_relu))
                    self.residual_weight_list.append(nn.Parameter(torch.FloatTensor(nhid * nheads, nhid * nheads)))
                # for i, attention in enumerate(self.gat_list[depth_index]):
                #     self.add_module('attention_{}_{}'.format(depth_index, i), attention)
            # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
            self.out_att = GATLayer(nhid*nheads, nclass, 1, dropout, dropout, activation=F.leaky_relu)
            self.residual_weight_list.append(nn.Parameter(torch.FloatTensor(nhid * nheads, nclass)))
            self.dropout = nn.Dropout(p=dropout)
        for i in range(self.depth):
            stdv = 1. / sqrt(self.residual_weight_list[i].size(1))
            self.residual_weight_list[i].data.uniform_(-stdv, stdv)

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        t = self.csr_to_sparse(mx)
        return t

    def csr_to_sparse(self, mx):
        '''
        convert scipy scr matrix to sparse tensor
        '''
        data = torch.FloatTensor(mx.data)
        idx = torch.LongTensor(mx.nonzero())
        t = torch.sparse.FloatTensor(idx, data, torch.Size(mx.shape))
        return t

    #---- non residual ----
    def forward(self, raw_x):
        if self.residual_type == 'naive':
            return self.forward_naive(raw_x)
        elif self.residual_type == 'raw':
            return self.forward_raw(raw_x)
        elif self.residual_type == 'graph_naive':
            return self.forward_graph_naive(raw_x)
        elif self.residual_type == 'graph_raw':
            return self.forward_graph_raw(raw_x)

    def forward_raw(self, raw_x):
        x = raw_x
        for i in range(self.depth-1):
            x = self.dropout(x)
            # x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.mm(raw_x, self.residual_weight_list[0])
            x = self.gat_list[i](self.g, x).reshape(x.size(0),-1)   + torch.mm(raw_x, self.residual_weight_list[0])
        x = self.dropout(x)
        if self.depth == 1:
            x = F.elu(self.out_att(self.g,x)).squeeze(1) + torch.mm(raw_x, self.residual_weight_list[self.depth - 1])
        else:
            x = F.elu(self.out_att(self.g, x)).squeeze(1) + torch.mm(torch.mm(raw_x, self.residual_weight_list[0]), self.residual_weight_list[self.depth-1])
        return x

    def forward_graph_raw(self, raw_x):
        x = raw_x
        for i in range(self.depth-1):
            x = self.dropout(x)
            # x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.spmm(self.norm_adj, torch.mm(raw_x, self.residual_weight_list[0]))
            x = self.gat_list[i](self.g, x).reshape(x.size(0), -1) + torch.spmm(self.adj, torch.mm(raw_x, self.residual_weight_list[0]))
        x=self.dropout(x)
        if self.depth == 1:
            x = F.elu(self.out_att(self.g, x)).squeeze(1) + torch.spmm(self.norm_adj, torch.mm(raw_x, self.residual_weight_list[self.depth - 1]))
        else:
            x = F.elu(self.out_att(self.g, x)).squeeze(1) + torch.spmm(self.norm_adj, torch.mm(torch.mm(raw_x, self.residual_weight_list[0]), self.residual_weight_list[self.depth-1]))
        return x

    def forward_naive(self, raw_x):
        x = raw_x
        for i in range(self.depth-1):
            x = self.dropout(x)
            x = self.gat_list[i](self.g, x).reshape(x.size(0), -1) + torch.mm(x, self.residual_weight_list[i])
            # x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.mm(x, self.residual_weight_list[i])
        x = self.dropout(x)
        x = F.elu(self.out_att(self.g, x)).squeeze(1) + torch.mm(x, self.residual_weight_list[self.depth-1])
        return x

    def forward_graph_naive(self, raw_x):
        x = raw_x
        for i in range(self.depth-1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gat_list[i](self.g, x).reshape(x.size(0), -1) + torch.spmm(self.norm_adj,torch.mm(x, self.residual_weight_list[i]) )
            # x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.spmm(self.norm_adj, torch.mm(x, self.residual_weight_list[i]))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(self.g, x)).squeeze(1) + torch.spmm(self.norm_adj, torch.mm(x, self.residual_weight_list[self.depth-1]))
        return x
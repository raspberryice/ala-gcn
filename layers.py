import dgl 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math 
import dgl.function as fn 
from dgl.nn.pytorch import edge_softmax
class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        '''uniform init.
        '''

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GATLayer(nn.Module):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATLayer, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = lambda x:x
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


def adaptive_message_func(edges):
    '''
    send data for computing metrics and update.
    '''
    return {'feat':edges.src['h'],'logits': edges.src['logits']}

def adaptive_attn_message_func(edges):
    return {'feat': edges.src['ft']* edges.data['a'], 
        'logits': edges.src['logits'], 
        'a': edges.data['a']}

def adaptive_attn_reduce_func(nodes):
    # (n_nodes, n_edges, n_classes)
    _, pred = torch.max(nodes.mailbox['logits'], dim=2)
    _, center_pred = torch.max(nodes.data['logits'], dim=1)
    n_degree = nodes.data['degree']
    # case 1 
    # ratio of common predictions 
    a = nodes.mailbox['a'].squeeze(3) #(n_node, n_neighbor, n_head, 1)
    n_head = a.size(2)
    idxs = torch.eq(pred, center_pred.unsqueeze(1)).unsqueeze(2).expand_as(a)
    f1 = torch.div(torch.sum(a*idxs, dim=1), n_degree.unsqueeze(1)) # (n_node, n_head)
    f1 = f1.detach()
    # case 2 
    # entropy of neighborhood predictions
    uniq = torch.unique(pred)
    # (n_unique)
    cnts_p = torch.zeros((pred.size(0), n_head, uniq.size(0),)).cuda()
    for i,val in enumerate(uniq):
        idxs = torch.eq(pred, val).unsqueeze(2).expand_as(a)
        tmp = torch.div(torch.sum(a*idxs, dim=1),n_degree.unsqueeze(1)) # (n_nodes, n_head)
        cnts_p[:,:,  i] = tmp 
    cnts_p = torch.clamp(cnts_p, min=1e-5)
    f2 = (-1)* torch.sum(cnts_p * torch.log(cnts_p),dim=2)
    f2 = f2.detach()
    neighbor_agg = torch.sum(nodes.mailbox['feat'], dim=1) #(n_node, n_head, n_feat)
    return {
        'f1': f1,
        'f2':f2,
        'agg': neighbor_agg,
    }


def adaptive_reduce_func(nodes):
    '''
    compute metrics and determine if we need to do neighborhood aggregation.
    '''
    # (n_nodes, n_edges, n_classes)
    _, pred = torch.max(nodes.mailbox['logits'], dim=2)
    _, center_pred = torch.max(nodes.data['logits'], dim=1)
    n_degree = nodes.data['degree']
    # case 1 
    # ratio of common predictions 
    f1 = torch.sum(torch.eq(pred,center_pred.unsqueeze(1)), dim=1)/n_degree
    f1 = f1.detach()
    # case 2 
    # entropy of neighborhood predictions
    uniq = torch.unique(pred)
    # (n_unique)
    cnts_p = torch.zeros((pred.size(0), uniq.size(0),)).cuda()
    for i,val in enumerate(uniq):
        tmp = torch.sum(torch.eq(pred, val), dim=1)/n_degree 
        cnts_p[:, i] = tmp 
    cnts_p = torch.clamp(cnts_p, min=1e-5)

    f2 = (-1)* torch.sum(cnts_p * torch.log(cnts_p),dim=1)
    f2 = f2.detach()
    return {
        'f1': f1,
        'f2':f2,
    }

class GatedAttnLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout, num_heads,
                 attn_drop=0.,
                 negative_slope=0.2,lidx=1):
        super(GatedAttnLayer, self).__init__() 
        
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        if in_feats != out_feats:
            self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False) # for first layer 
        
        
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation 
        self.tau_1 = nn.Parameter(torch.zeros((1,)))
        self.tau_2 = nn.Parameter(torch.zeros((1,)))
        
        self.ln_1 = nn.LayerNorm((g.number_of_nodes(), num_heads),elementwise_affine=False)
        self.ln_2 = nn.LayerNorm((g.number_of_nodes(),num_heads), elementwise_affine=False)

        self.reset_parameters(lidx)

    def reset_parameters(self, lidx, how='layerwise'):
        gain = nn.init.calculate_gain('relu')
        if how == 'normal':
            nn.init.normal_(self.tau_1)
            nn.init.normal_(self.tau_2)
        else:
            nn.init.constant_(self.tau_1, 1/(lidx+1))
            nn.init.constant_(self.tau_2, 1/(lidx+1))
        
        return 

    def forward(self, g, h, logits, old_z, attn_l, attn_r, shared_tau=True, tau_1=None, tau_2=None):
        g = g.local_var()
        if self.feat_drop:
            h = self.feat_drop(h)

        if hasattr(self, 'fc'):
            feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        else:
            feat = h 
        g.ndata['h'] = feat # (n_node, n_feat)
        g.ndata['logits'] = logits 
        
        

        el = (feat * attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * attn_r).sum(dim=-1).unsqueeze(-1)
        g.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(g.edata.pop('e'))
        # compute softmax
        g.edata['a'] = self.attn_drop(edge_softmax(g, e))

        g.update_all(message_func=adaptive_attn_message_func, reduce_func=adaptive_attn_reduce_func)
        f1 = g.ndata.pop('f1')
        f2 = g.ndata.pop('f2')
        norm_f1 = self.ln_1(f1) 
        norm_f2 = self.ln_2(f2)
        if shared_tau:
            z = F.sigmoid((-1)*(norm_f1-tau_1)) * F.sigmoid((-1)*(norm_f2-tau_2)) 
        else:
            # tau for each layer 
            z = F.sigmoid((-1)*(norm_f1-self.tau_1)) * F.sigmoid((-1)*(norm_f2-self.tau_2)) 
        
        gate = torch.min(old_z, z)

       
        agg = g.ndata.pop('agg')
        normagg = agg * g.ndata['norm'].unsqueeze(1)   # normalization by tgt degree
    
        if self.activation:
            normagg = self.activation(normagg)
        new_h = feat + gate.unsqueeze(2)*normagg
        return new_h,z

        





class GatedLayer(nn.Module):
    def __init__(self,g,in_feats, out_feats, activation, dropout, lidx=1):
        super(GatedLayer, self).__init__()
        self.weight_neighbors= nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.tau_1 = nn.Parameter(torch.zeros((1,)))
        self.tau_2 = nn.Parameter(torch.zeros((1,)))
        
        self.ln_1 = nn.LayerNorm((g.number_of_nodes()),elementwise_affine=False)
        self.ln_2 = nn.LayerNorm((g.number_of_nodes()), elementwise_affine=False)

        self.reset_parameters(lidx)
    
    def reset_parameters(self,lidx, how='layerwise'):
        # initialize params 
        if how == 'normal':
            nn.init.normal_(self.tau_1)
            nn.init.normal_(self.tau_2)
        else:
            nn.init.constant_(self.tau_1, 1/(lidx+1))
            nn.init.constant_(self.tau_2, 1/(lidx+1))
        return 
        
    
    def forward(self, g, h, logits, old_z, shared_tau=True, tau_1=None, tau_2=None):
        # operates on a node
        g = g.local_var()
        if self.dropout:
            h = self.dropout(h)
        g.ndata['h'] = h 
        g.ndata['logits'] = logits 
        
        g.update_all(message_func=fn.copy_u('logits','logits'), reduce_func=adaptive_reduce_func)
        f1 = g.ndata.pop('f1')
        f2 = g.ndata.pop('f2')
        norm_f1 = self.ln_1(f1)
        norm_f2 = self.ln_2(f2)
        if shared_tau:
            z = F.sigmoid((-1)*(norm_f1-tau_1)) * F.sigmoid((-1)*(norm_f2-tau_2)) 
        else:
            # tau for each layer 
            z = F.sigmoid((-1)*(norm_f1-self.tau_1)) * F.sigmoid((-1)*(norm_f2-self.tau_2)) 
        
        gate = torch.min(old_z, z)
        g.update_all(message_func=fn.copy_u('h','feat'), reduce_func=fn.sum(msg='feat', out='agg'))

        agg = g.ndata.pop('agg')
    
        normagg = agg * g.ndata['norm']   # normalization by tgt degree
    
        if self.activation:
            normagg = self.activation(normagg)
        new_h = h + gate.unsqueeze(1)*normagg
        return new_h,z 



class GatedAPPNPConv(nn.Module):
    r"""Approximate Personalized Propagation of Neural Predictions
    layer from paper `Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank <https://arxiv.org/pdf/1810.05997.pdf>`__.
    .. math::
        H^{0} & = X
        H^{t+1} & = (1-\alpha)\left(\hat{D}^{-1/2}
        \hat{A} \hat{D}^{-1/2} H^{t}\right) + \alpha H^{0}
    Parameters
    ----------
    k : int
        Number of iterations :math:`K`.
    alpha : float
        The teleport probability :math:`\alpha`.
    edge_drop : float, optional
        Dropout rate on edges that controls the
        messages received by each node. Default: ``0``.
    """
    def __init__(self,
                 g, k,
                 n_hidden, n_classes,
                 edge_drop=0., lidx=1):
        super(GatedAPPNPConv, self).__init__()
        self._k = k
        self.edge_drop = nn.Dropout(edge_drop)
        self.tau_1 = nn.Parameter(torch.zeros((1,)))
        self.tau_2 = nn.Parameter(torch.zeros((1,)))
        
        self.ln_1 = nn.LayerNorm((g.number_of_nodes()),elementwise_affine=False)
        self.ln_2 = nn.LayerNorm((g.number_of_nodes()), elementwise_affine=False)
        self.weight_y = nn.Linear(n_hidden, n_classes) 
        self.reset_parameters(lidx)
    
    def reset_parameters(self,lidx, how='layerwise'):
        # initialize params 
        if how == 'normal':
            nn.init.normal_(self.tau_1)
            nn.init.normal_(self.tau_2)
        else:
            nn.init.constant_(self.tau_1, 1/(lidx+1))
            nn.init.constant_(self.tau_2, 1/(lidx+1))
        return 


    def forward(self, graph, feat, logits):
        r"""Compute APPNP layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        graph = graph.local_var()
        norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)
        feat_0 = feat
        z = torch.FloatTensor([1.0,]).cuda()
        for lidx in range(self._k):
            # normalization by src node
            old_z = z 
            feat = feat * norm
            graph.ndata['h'] = feat
            old_feat = feat 
            if lidx != 0:
                logits = self.weight_y(feat)
            graph.ndata['logits'] = logits 
        
            graph.update_all(message_func=fn.copy_u('logits','logits'), reduce_func=adaptive_reduce_func)
            f1 = graph.ndata.pop('f1')
            f2 = graph.ndata.pop('f2')
            norm_f1 = self.ln_1(f1)
            norm_f2 = self.ln_2(f2)
            z = F.sigmoid((-1)*(norm_f1-self.tau_1)) * F.sigmoid((-1)*(norm_f2-self.tau_2)) 
        
            gate = torch.min(old_z, z)
            graph.edata['w'] = self.edge_drop(
                torch.ones(graph.number_of_edges(), 1).to(feat.device))
            graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                             fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            # normalization by dst node
            feat = feat * norm
            feat = z.unsqueeze(1)* feat + old_feat # raw features

        return feat

class GraphTopoAttention(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 topo_dim,
                 out_dim,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 residual=False,
                 concat=True,
                 last_layer=False):
        super(GraphTopoAttention, self).__init__()
        self.g = g
        self.num_heads = num_heads
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x
        # weight matrix Wl for leverage property
        if last_layer:
            self.fl = nn.Linear(in_dim+topo_dim, out_dim, bias=False)
        else:
            self.fl = nn.Linear(in_dim, num_heads*out_dim, bias=False)
        # weight matrix Wc for aggregation context
        self.fc = nn.Parameter(torch.Tensor(size=(in_dim+topo_dim, num_heads*out_dim)))
        # weight matrix Wq for neighbors' querying
        self.fq = nn.Parameter(torch.Tensor(size=(in_dim, num_heads*out_dim)))
        nn.init.xavier_normal_(self.fl.weight.data)
        nn.init.constant_(self.fc.data, 10e-3)
        nn.init.constant_(self.fq.data, 10e-3)
        self.attn_activation = nn.ELU()
        self.softmax = edge_softmax
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fl = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fl.weight.data)
            else:
                self.res_fl = None
        self.concat = concat
        self.last_layer = last_layer
    

    def forward(self, inputs, topo=None):
        # prepare
        h = self.feat_drop(inputs)  # NxD
        if topo:
            t   = self.feat_drop(topo) #N*T
       
        if not self.last_layer:
            ft = self.fl(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            if topo: 
                ft_c = torch.matmul(torch.cat((h, t), 1), self.fc).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                ft_c = torch.matmul(h, self.fc).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            ft_q = torch.matmul(h, self.fq).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            self.g.ndata.update({'ft' : ft, 'ft_c' : ft_c, 'ft_q' : ft_q})
            self.g.apply_edges(self.edge_attention)
            self.edge_softmax()

            l_s = int(0.713*self.g.edata['a_drop'].shape[0])
            topk, _ = torch.topk(self.g.edata['a_drop'], l_s, largest=False, dim=0)
            thd = torch.squeeze(topk[-1])
            self.g.edata['a_drop'] = self.g.edata['a_drop'].squeeze()
            self.g.edata['a_drop'] = torch.where(self.g.edata['a_drop']-thd<0, self.g.edata['a_drop'].new([0.0]), self.g.edata['a_drop'])
            attn_ratio = torch.div((self.g.edata['a_drop'].sum(0).squeeze()+topk.sum(0).squeeze()), self.g.edata['a_drop'].sum(0).squeeze())
            self.g.edata['a_drop'] = self.g.edata['a_drop'] * attn_ratio
            self.g.edata['a_drop'] = self.g.edata['a_drop'].unsqueeze(-1)
            
            self.g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
            ret = self.g.ndata['ft']
            if self.residual:
                if self.res_fl is not None:
                    resval = self.res_fl(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
                else:
                    resval = torch.unsqueeze(h, 1)  # Nx1xD'
                ret = resval + ret
            ret = torch.cat((ret.flatten(1), ft.mean(1).squeeze()), 1) if self.concat else ret.flatten(1)
        else:
            if topo:
                ret = self.fl(torch.cat((h, t), 1))
            else:
                ret = self.fl(h)

        return ret

    def edge_attention(self, edges):
        c = edges.dst['ft_c']
        q = edges.src['ft_q'] - c
        a = (q * c).sum(-1).unsqueeze(-1)
        return {'a': self.attn_activation(a)}
        
    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop('a'))
        self.g.edata['a_drop'] = self.attn_drop(attention)
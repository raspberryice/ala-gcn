from dgl.data import citation_graph as citegrh
import dgl
import networkx as nx
import torch
from torch.utils.data import Dataset
from dgl.data.utils import split_dataset
from dgl.data.reddit import RedditDataset
from dgl.data.gnn_benckmark import AmazonCoBuy
from dgl.data import KarateClub

import random
from math import ceil
import numpy as np

class SmallGraphDataset(Dataset):
    '''
    Small datasets do not need batching.
    '''
    def __init__(self, name, seed, self_loop=False, split=None):
        super(SmallGraphDataset, self).__init__()
        if name == 'cora':
            data = citegrh.load_cora()
            graph = data.graph
            if self_loop:
                graph = self.add_selfloop(graph)
            graph = dgl.DGLGraph(graph)
            features = data.features
            labels = data.labels

        elif name == 'citeseer':
            data = citegrh.load_citeseer()
            graph = data.graph
            if self_loop:
                graph = self.add_selfloop(graph)
            graph = dgl.DGLGraph(graph)
            features = data.features
            labels = data.labels

        elif name == 'pubmed':
            data = citegrh.load_pubmed()
            graph = data.graph
            if self_loop:
                graph = self.add_selfloop(graph)
            graph = dgl.DGLGraph(graph)
            features = data.features
            labels = data.labels

        elif name == 'amazon':
            assert(split!=None)
            data = AmazonCoBuy(name='computers')
            graph = data.data[0]
            if self_loop:
                graph.remove_edges(graph.edge_ids(graph.nodes(), graph.nodes()))
                graph.add_edges(graph.nodes(), graph.nodes())
            # must create split
            features = graph.ndata['feat']
            labels = graph.ndata['label']
        elif name =='karate':
            kG = nx.karate_club_graph()
            labels = np.array(
            [kG.nodes[i]['club'] != 'Mr. Hi' for i in kG.nodes]).astype(np.int64)
            graph = dgl.DGLGraph(kG)
            if self_loop:
                graph.remove_edges(graph.edge_ids(graph.nodes(), graph.nodes()))
                graph.add_edges(graph.nodes(), graph.nodes())
            features = torch.eye(n=graph.number_of_nodes())
            # graph.ndata['feat'] = features

            # Mr.Hi's club:1, John A's club:0
            self.train_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool)
            self.train_mask[0] = True #Mr.Hi
            self.train_mask[33] = True # John A
            self.test_mask = ~self.train_mask



        graph = self.compute_norm(graph)

        self.graph = graph
        self.features = torch.FloatTensor(features)
        self.n_features = self.features.size(1)
        self.labels = torch.LongTensor(labels)
        self.n_label = torch.unique(self.labels).size(0)
        self.n_nodes = graph.number_of_nodes()
        if hasattr(self, 'train_mask'):
            return

        if split:
            print('using {} for training data.'.format(split))
            assert(split > 0.0)
            assert(split < 1.0)
            sample_size = ceil(self.n_nodes*split)
            train_np = np.zeros(self.n_nodes, dtype=np.bool)
            test_np = np.zeros(self.n_nodes, dtype=np.bool)
            test_np[range(500,1500)] = 1

            if seed ==0:
                # use first few data points as seed 
                train_idx = range(sample_size)
                train_np[train_idx] = 1
            else:
                random.seed(seed)
                train_idx = random.sample(range(self.n_nodes-1000), sample_size)
                mapped_train_idx = [idx if idx<500 else idx+1000 for idx in train_idx]
                train_np[mapped_train_idx] =1 
            

            self.train_mask = torch.tensor(train_np, dtype=torch.bool)
            self.test_mask = torch.tensor(test_np, dtype=torch.bool)
        else: # use original split
            self.train_mask = torch.BoolTensor(data.train_mask)
            self.test_mask = torch.BoolTensor(data.test_mask)


    def add_selfloop(self, g):
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
        return g

    def compute_norm(self,g):
        n_edges = g.number_of_edges()
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        g.ndata['degree'] = degs
        return g

    
    def __len__(self):
        return self.features.size(0)

    def __getitem__(self,idx):
        '''
        return whole graph features regardless of idx.
        '''
        return {
            'features': self.features[idx,:],
            'labels': self.labels[idx],
            'train_mask': self.train_mask[idx],
            'test_mask': self.test_mask[idx],
        }

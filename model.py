import pytorch_lightning as pl
import numpy as np
import torch 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import dgl 

from network import GCN, SGC, AdaptiveGNN, GraphSAGE, GAT, AdaptiveGAT, APPNP, GTN, GResNet, GATResNet, AdaAPPNP
from data import SmallGraphDataset

class GNNModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args 
        self.dataset = SmallGraphDataset(name=args.dataset, seed=args.seed, self_loop=args.self_loop, split=args.percentage)        
        self.graph = self.dataset.graph
        for gpu_id in args.gpus.split(','):
            self.graph.to(torch.device('cuda:{}'.format(int(gpu_id.strip())))) 
        if args.model == 'gcn':
            self.net = GCN(g=self.graph, 
                in_feats=self.dataset.n_features, 
                n_hidden=args.n_hidden,  
                n_classes=self.dataset.n_label, 
                n_layers=args.n_layers,
                activation=F.relu,
                dropout=args.dropout)
        elif args.model =='sgc':
            if self.dataset == 'reddit':
                cache = True
            else:
                cache = False 
            self.net = SGC(graph=self.graph, in_feats=self.dataset.n_features,
                out_feats=self.dataset.n_label,
                k=args.n_layers+1, cached=cache,bias=args.bias, norm=None)
        elif args.model == 'adagnn':
            self.net = AdaptiveGNN(g=self.graph, in_feats=self.dataset.n_features,n_hidden=args.n_hidden,
            n_classes = self.dataset.n_label, n_layers=args.n_layers,activation=F.relu, dropout=args.dropout)
            training_feats =self.dataset.features[self.dataset.train_mask]
            self.net.init_weight(training_feats, self.dataset.labels[self.dataset.train_mask])
        elif args.model == 'graphsage':
            self.net = GraphSAGE(g=self.graph, in_feats=self.dataset.n_features, n_hidden=args.n_hidden,
            n_classes = self.dataset.n_label, n_layers=args.n_layers, activation=F.relu, 
            dropout=args.dropout, aggregator_type=args.aggregator)
        elif args.model == 'gat':
            heads = ([args.num_heads] * args.n_layers) + [args.num_out_heads]
            self.net = GAT(g=self.graph, num_layers=args.n_layers,in_dim=self.dataset.n_features, 
            num_hidden=args.n_hidden, 
            num_classes=self.dataset.n_label,
            negative_slope=args.negative_slope,
            heads=heads,
            feat_drop=args.dropout,
            attn_drop=args.attn_drop,
            residual=args.residual,
            activation=F.leaky_relu)
        elif args.model =='adagat':
            heads = ([args.num_heads] * args.n_layers) + [args.num_out_heads]
            self.net = AdaptiveGAT(g=self.graph, 
                in_feats=self.dataset.n_features, 
                n_hidden=args.n_hidden,  
                n_classes=self.dataset.n_label, 
                n_layers=args.n_layers,
                activation=F.leaky_relu,
                dropout=args.dropout,
                negative_slope=args.negative_slope,
                heads=heads,
                attn_drop=args.attn_drop)
            training_feats =self.dataset.features[self.dataset.train_mask]
            self.net.init_weight(training_feats, self.dataset.labels[self.dataset.train_mask])
        elif args.model == 'appnp':
            self.net = APPNP(g=self.graph, in_feats=self.dataset.n_features, hiddens=[args.n_hidden,]*args.n_layers, 
                n_classes=self.dataset.n_label,
                activation=F.relu,
                feat_drop=args.dropout,
                edge_drop=args.edge_drop, 
                alpha=args.alpha,
                k=args.k) 
        elif args.model == 'gtn':
            heads = ([args.num_heads] *(args.n_layers +1)) 
            self.net=GTN(g=self.graph, num_layers=args.n_layers, feats_d=self.dataset.n_features,
             feats_t_d=0,# no topo features 
             num_hidden=args.n_hidden,
             num_classes=self.dataset.n_label,
             heads= heads,
             activation=F.elu,
             feat_drop=args.dropout,
             attn_drop=args.attn_drop,
             residual=args.residual,
             concat=args.concat)
        elif args.model == 'gresnet-gcn':
            self.net = GResNet(g=self.graph, nfeat=self.dataset.n_features, nhid=args.n_hidden, 
            nclass=self.dataset.n_label, dropout=args.dropout, depth=args.n_layers,
            residual_type=args.residual_type)
        elif args.model == 'gresnet-gat': 
            self.net = GATResNet(g=self.graph, nfeat=self.dataset.n_features, nhid=args.n_hidden,
            nclass=self.dataset.n_label, dropout=args.dropout, depth=args.n_layers,
            nheads = args.num_heads, residual_type=args.residual_type, alpha=args.negative_slope)
        elif args.model == 'adaappnp':
            self.net = AdaAPPNP(g=self.graph, in_feats=self.dataset.n_features,
             n_hidden=args.n_hidden, 
            n_classes=self.dataset.n_label, n_layers=args.n_layers, 
            dropout=args.dropout, edge_drop=args.edge_drop, activation=F.relu)
            training_feats =self.dataset.features[self.dataset.train_mask]
            self.net.init_weight(training_feats, self.dataset.labels[self.dataset.train_mask])


    def training_step(self, data_batch, batch_nb):
        features = data_batch['features']
        labels = data_batch['labels']
        train_mask = data_batch['train_mask']
        test_mask = data_batch['test_mask']

        if self.hparams.model in ['adagnn','adagat']:
            logits,z  = self.net(features)
        else:
            logits = self.net(features)

        if self.hparams.model == 'gtn':
            # convert labels to one hot 
            pred = logits[train_mask]
            labels_one_hot = torch.zeros(pred.size()).to(pred.device)
            idxs = labels[train_mask].unsqueeze(1)
            labels_one_hot.scatter_(1, idxs, 1)

            loss = F.binary_cross_entropy_with_logits(pred, labels_one_hot)
        else:
            logp = F.log_softmax(logits, 1)
            nll = F.nll_loss(logp[train_mask], labels[train_mask])
            loss = nll 


        
        if self.hparams.model in ['adagnn','adagat']:
            reg = torch.norm(z * (torch.ones_like(z) - z), p=1) 
            loss = nll  + self.hparams.binary_reg * reg  
        logger = {'tng_loss': loss} 
        if self.hparams.model in ['adagnn', 'adagat']:
            logger['reg'] = reg 
            logger['z_norm'] = torch.norm(z, p=1)
        return {
            'loss': loss,
            'log': logger,
        }

    
    
    def validation_step(self, data_batch, batch_nb):
        features = data_batch['features']
        labels = data_batch['labels']
        train_mask = data_batch['train_mask']
        test_mask = data_batch['test_mask']
        if self.hparams.model in ['adagnn', 'adagat']:
            logits, z = self.net(features)
        else:
            logits = self.net(features)
        
        _, predictions = torch.max(logits[test_mask], dim=1)
        correct = torch.sum(predictions == labels[test_mask])
        return {
            'predictions': predictions,
            'correct': correct, 
        }
    def validation_end(self, outputs):
        '''
        aggregate outputs for metrics.
        '''
        predictions_all = torch.stack([x['predictions'] for x in outputs])
        correct_all = torch.stack([x['correct'] for x in outputs])
        acc = torch.sum(correct_all).item() * 1.0 / predictions_all.numel()
        logger = {'val_acc': round(acc,4)}
        return {
            'log': logger,
        }
    
    def configure_optimizers(self):
        if self.hparams.opt == 'lbfgs':
            optimizer = torch.optim.LBFGS(self.net.parameters(), lr=self.hparams.lr) # only for sgc / reddit 
        else:
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay) 
        return [optimizer,]


    @pl.data_loader
    def tng_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=len(self.dataset),shuffle=False,
            num_workers=0,pin_memory=True)
        return dataloader 
    
    @pl.data_loader
    def val_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=len(self.dataset),shuffle=False,
            num_workers=0,pin_memory=True)
        return dataloader 



    
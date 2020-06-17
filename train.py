import argparse 
import pytorch_lightning as pl
import numpy as np
import torch 
from model import GNNModel
from test_tube import Experiment
from test_tube import HyperOptArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import TestTubeLogger
import uuid 
from datetime import datetime 
import random 




if __name__ == '__main__':
    parser = HyperOptArgumentParser()
    # runtime params 
    parser.add_argument('--dataset', type=str, choices=['cora','citeseer','pubmed','amazon'], default='cora') 
    parser.add_argument('--percentage', type=float) 
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--ckpt-name', type=str)
    parser.add_argument('--seed', type=int, default=2020) 
    parser.add_argument('--opt', default='adam', choices=['adam','lbfgs'])
    parser.add_argument("--gpus", type=str, default='0',
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-6,
            help="Weight for L2 loss")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    # model params 
    parser.add_argument('--model', type=str, default='adaappnp', choices=['gcn','graphsage','sgc','adagnn','gat','adagat', 'appnp','gtn','gresnet-gcn','gresnet-gat','adaappnp'] )
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=8,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    # AdaGNN params 
    parser.add_argument('--binary-reg', type=float, default=0.0)
    # GAT params 
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    # GraphSAGE params
    parser.add_argument('--aggregator', type=str, choices=['mean', 'pool','gcn', 'lstm'], default='gcn')
    # SGC params 
    parser.add_argument('--bias', default=False, action='store_true')
    # APPNP params 
    parser.add_argument('--edge-drop', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.1, type=float, help='teleporting probability')
    parser.add_argument('--k', type=int,default=10, help='number of propagation steps.') 
    
    #GTN params 
    parser.add_argument('--sample-number', type=int, default=32, help='characteristic function sample number. Will generate feats_t.npy.')
    
    parser.add_argument('--concat', action='store_true', default=True, help='concat neighbors with itself.')
    
    # GResNet params 
    parser.add_argument('--residual-type', type=str, default='graph_raw', choices=['naive','raw','graph_naive','graph_raw'])
    
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.ckpt_name:
        d = datetime.now() 
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}_l{}_h{}_{}'.format(args.model,
                args.dataset, args.n_layers, args.n_hidden, time_str)

    checkpoint_callback = ModelCheckpoint(
        filepath=f'./checkpoints/{args.ckpt_name}',
        save_best_only=True,
        monitor='val_acc',
        mode='max',
    )
    tt_logger = TestTubeLogger(
        save_dir=args.log_dir,
        name=args.ckpt_name,
    )

    model = GNNModel(args)
    trainer = pl.Trainer(logger=tt_logger,
                      min_nb_epochs=args.n_epochs,
                      max_nb_epochs=args.n_epochs, 
                      gpus=None if args.gpus == '-1' else [int(x) for x in args.gpus.split(',')],
                      val_percent_check=1.0,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=None, 
                      )
    torch.cuda.set_device(int(args.gpus))
    trainer.fit(model)

    
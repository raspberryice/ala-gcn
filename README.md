# Ala-GCN

## Dependencies

- pytorch=1.4 (1.4 to save sparse tensors)
- dgl
- pytorch-lightning=0.5.3.2

(pytorch-lightning has gone through some breaking changes since we wrote this code. If you try using the latest version of pytorch-lightning, you will run into errors like `train_dataloader` not defined.) 


## Datasets

- Cora
- Citeseer
- Pubmed
- Amazon (co-purchasing network) 


## Models

- GCN
- GAT
- GraphSAGE
- SGC
- CS-GNN (ICLR 2020)
- GResNet (Arxiv)
- APPNP
- AdaGNN (AdaGCN)
- AdaGAT

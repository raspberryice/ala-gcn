# Ala-GCN

## Dependencies

- pytorch=1.4 (1.4 to save sparse tensors)
- dgl=0.4.1
- pytorch-lightning=0.5.3.2

(pytorch-lightning has gone through some breaking changes since we wrote this code. If you try using the latest version of pytorch-lightning, you will run into errors like `train_dataloader` not defined.) 


## Datasets

- Cora
- Citeseer
- Pubmed
- Amazon (co-purchasing network) 


## Hyperparameters 

### Cora 
```bash
python train.py --model=adagnn --dataset=cora --lr=0.01 --percentage=0.05 --n-layers=5 --dropout=0.5 --weight-decay=5e-6 --n-hidden=16 --n-epochs=200 --self-loop

```
Use 5 layers for 5% and 3%, 9 layers for 1%.

```bash
python train.py --percentage=0.05 --n-layers=3  --dataset=cora --model=adagat --lr=0.005 --dropout=0.6 --weight-decay=5.00E-04 --n-hidden=8 --n-epochs=300 --self-loop
```
Use 3 layers for 5%, 3% and 1%.
Notice that due to the size of the dataset, the variance of performance with 1% seed labels is relatively large. 

### Citeseer 

```bash
python train.py  --percentage=0.05 --n-layers=3 --dataset=citeseer --model=adagnn --lr=0.01 --dropout=0.5 --weight-decay=5.00E-06 --n-hidden=16 --n-epochs=200 --self-loop
```
Use 3 layers for 5%, 4 layers for 3% and 7 layers for 1%.

```bash
python train.py --percentage=0.05 --n-layers=3 --dataset=citeseer --model=adagat --lr=0.005 --dropout=0.8 --weight-decay=5.00E-04 --n-hidden=8 --n-epochs=300 --self-loop
```
Use 3 layers for 5%, 3 layers for 3% and 7 layers for 1%.

### Pubmed 
```bash
python train.py --percentage=0.003 --n-layers=9 --dataset=pubmed --model=adagnn --lr=0.01 --dropout=0.5 --weight-decay=5.00E-06 --n-hidden=16 --n-epochs=500 --self-loop
```
Use 9 layers for 0.3%, 0.15% and 0.05%.

```bash
python train.py --percentage=0.003 --n-layers=5 --dataset=pubmed --model=adagat --lr=0.005 --dropout=0.5 --weight-decay=5.00E-04 --n-hidden=16 --n-epochs=500' --self-loop
```
Use 5 layers for 0.3%, 5 layers for 0.15% and 7 layers for 0.05%.

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

# Self-Supervised Brain MRI Tumor Segmentation


## Reference Article
We compare ourselves with the winning article of BraTS 2018:

[3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/pdf/1810.11654v3.pdf)

Preprocessing:
- normalize to 0 mean and unit std
- random per-channel intensity shift (-0.1, 0.1)
- random scale (0.9, 1.1)
- random axis mirror flip (all 3 axes) - probability 0.5
- random crop-size: 160x192x128 
- concat 4 modalities as input channels

Optimization:
- Adam optimizer. lr=1e-4 + custom decay function in article
- epochs: 300
- batch-size: 1
- L2 reg 1e-5
output 3 nested tumor sub-regions
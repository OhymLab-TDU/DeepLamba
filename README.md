# DeepLamba: Efficient Mamba‑Based Model for Medical Image Segmentation
Official paper repository for: *[DeepLamba: Efficient Mamba‑Based Model for Medical Image Segmentation](https://link.springer.com/article/10.1007/s42979-025-04267-9)*

Deeplamba Architecture:
![network](https://github.com/OhymLab-TDU/DeepLamba/blob/main/image/DeepLamba_architecture.png)

## Experiment results
- AbdomenMRI
<div style="display:flex; justify-content:center; align-items:center;">
  <img src="https://github.com/OhymLab-TDU/DeepLamba/blob/main/image/AdomenMRI_result.png" width="45%" />
  <img src="https://github.com/OhymLab-TDU/DeepLamba/blob/main/image/AdomenMRI.png" width="45%" />
</div>

- Microscopy
<div style="display:flex; justify-content:center; align-items:center;">
  <img src="https://github.com/OhymLab-TDU/DeepLamba/blob/main/image/NuerlPSCell_result.png" width="45%" />
  <img src="https://github.com/OhymLab-TDU/DeepLamba/blob/main/image/Microscopy.png" width="45%" />
</div>

## Installation

**Step-1:** Create a new conda environment & install requirements

```shell
conda create -n deeplamba python=3.10
conda activate deeplamba

pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm
pip install torchinfo timm numba
```

**Step-2:** Install DeepLamba

```shell
git clone https://github.com/OhymLab-TDU/DeepLamba.git
cd DeepLamba/deeplamba
pip install -e .
```




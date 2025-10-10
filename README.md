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

## Dataset Prepare
We follow the [U-mamba](https://github.com/bowang-lab/U-Mamba?tab=readme-ov-file) dataset. 
First, download datasets from [U-Mamba](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN?usp=sharing).
Second, put datasets into data folder and do preprocess by 
```shell
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Training 
We use the [Swin-Umamba](https://github.com/JiarunLiu/Swin-UMamba/tree/main) scripts. We can train DeepLamba by executing:

```shell
# AbdomenMR dataset
bash scripts/train_AbdomenMR.sh nnUNetTrainerDeepLamba
# Microscopy dataset 
bash scripts/train_Microscopy.sh nnUNetTrainerDeepLamba
```

About Flops and Params calculation, we follow the [issues110](https://github.com/state-spaces/mamba/issues/110) method.

## Acknowledgements

We thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), [UMamba](https://github.com/bowang-lab/U-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-Umamba](https://github.com/JiarunLiu/Swin-UMamba/tree/main) for making their valuable code & data publicly available.

## Citation

```
@article{sun2025deeplamba,
  title={DeepLamba: Efficient Mamba-Based Model for Medical Image Segmentation},
  author={Sun, Shizhe and Ohyama, Wataru},
  journal={SN Computer Science},
  volume={6},
  number={6},
  pages={734},
  year={2025},
  publisher={Springer}
}
```





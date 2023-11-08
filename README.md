# HierNet
Code for the paper "Are 'Hierarchical' Visual Representations Hierarchical?" in NeurIPS Symmetry and Geometry Workshop in Neural Representations.

## Setup Steps
Clone and setup the repository using:
```
git clone git@github.com:ethanlshen/HierNet.git
cd HierNet
conda create -n HierNet python=3.9 --yes
conda activate HierNet
python -m pip install -r requirements.txt
```
The source code is compatible with MERU (https://github.com/facebookresearch/meru) and MRL (https://github.com/RAIVNLab/MRL). 
Clone and setup both repositories in the HierNet directory.

Next, create a checkpoint director to store model checkpoints. 
```
mkdir checkpoints
```
In this folder, download and store the following checkpoints:
- Model: [MERU ViT-large](https://dl.fbaipublicfiles.com/meru/meru_vit_l.pth)
- Model: [CLIP ViT-large](https://dl.fbaipublicfiles.com/meru/clip_vit_l.pth)
- Model: [MR-ResNet50](https://drive.google.com/file/d/1SnY6H3tbv4OkFZhfq7UgenFQ42faNjih/view?usp=drive_link)
- Model: [FF-Resnet50](https://drive.google.com/drive/folders/1Kb4KwpTPzX6VNZqzh7X6jHjUaicVEmcw?usp=drive_link) (all files)
## Evaluation Files
Evaluate on a dimension by ...

## Evaluate Own Models
To use datasets ... 

## Notebooks
Notebooks ...


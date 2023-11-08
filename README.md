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

Next, create a checkpoint folder to store model checkpoints and an embeds folder for image embeddings.
```
mkdir checkpoints
mkdir embeds
```
In the checkpoints folder, download and store the following checkpoints:
- Model: [MERU ViT-large](https://dl.fbaipublicfiles.com/meru/meru_vit_l.pth)
- Model: [CLIP ViT-large](https://dl.fbaipublicfiles.com/meru/clip_vit_l.pth)
- Model: [MR-ResNet50](https://drive.google.com/file/d/1SnY6H3tbv4OkFZhfq7UgenFQ42faNjih/view?usp=drive_link)
- Model: [FF-Resnet50](https://drive.google.com/drive/folders/1Kb4KwpTPzX6VNZqzh7X6jHjUaicVEmcw?usp=drive_link) (all files)
## Evaluation Files
Run the following command to evaluate a model on all datasets across all dimensions (512 for CLIP/MERU, 8-2048 for MR/FF), storing the results in --results_dir.
*--model* accepts the following arguments: clip, meru, ff, mrl.
```
python evaluate.py --model clip --dataset_dir ./files/dataset_info.pt --imagenet_dir /mnt/nvme0n1p2/data/ImageNet-1K
```
This command does the same for only *--model* being ff, with the exception that embeddings are now PCA reduced from 2048-dim ResNet50.
```
python evaluate_pca.py --model ff --dataset_dir ./files/dataset_info.pt --imagenet_dir /mnt/nvme0n1p2/data/ImageNet-1K --results_dir ./files/ff_pca1_data.pt
```

## Evaluate Own Models

## Notebooks
Notebooks ...


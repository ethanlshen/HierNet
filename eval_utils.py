import torch
import pandas as pd
import numpy as np
import random
import json
import argparse
import copy
import os
from tqdm import tqdm
from collections import OrderedDict

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from torchvision.models import resnet50, ResNet50_Weights

from MRL.utils import load_from_old_ckpt
from MRL.utils import apply_blurpool
from MRL.MRL import FixedFeatureLayer

from meru.config import LazyConfig, LazyFactory
from meru.evaluation.classification import _encode_dataset
from meru.utils.checkpointing import CheckpointManager
from meru.meru.lorentz import pairwise_dist, pairwise_inner

from robustness.tools.breeds_helpers import setup_breeds
from robustness.tools.breeds_helpers import ClassHierarchy
from robustness.tools.breeds_helpers import BreedsDatasetGenerator
from robustness import datasets

from cluster_analysis import *
from embeddings import *
from distances import *
from custom.eval_breeds import ZeroShotClassificationEvaluator


def load_mrl(f, dims, device):
    """
    Return mrl model over the given dimensions.
    Inputs:
    - f: location of mrl weights
    - dims: mrl dimensions
    Output:
    - mrl model
    """
    weights = torch.load(f, map_location='cpu')
    new_weights_dict = OrderedDict()
    for key in weights.keys():
        new_weights_dict[key[7:]] = weights[key]
    mrl_resnet = resnet50()
    mrl_resnet = load_from_old_ckpt(mrl_resnet, False, dims)
    apply_blurpool(mrl_resnet)
    mrl_resnet.load_state_dict(new_weights_dict)
    mrl_resnet.to(device)
    mrl_resnet.eval()
    return mrl_resnet
    
def load_ff(f, dim, device):
    """
    Return ff model over the given dimensions.
    Inputs:
    - f: location of ff weight for desired dimension
    - dims: desired dimension
    Output:
    - ff model desired dimensionality
    """
    weights = torch.load(f, map_location='cpu')
    new_weights_dict = OrderedDict()
    for key in weights.keys():
        new_weights_dict[key[7:]] = weights[key]
    ffresnet = resnet50()
    ffresnet.fc=FixedFeatureLayer(dim, 1000)
    apply_blurpool(ffresnet)
    ffresnet.load_state_dict(new_weights_dict)
    ffresnet.fc.weight = torch.nn.Parameter(ffresnet.fc.weight[:, :dim])
    ffresnet.to(device)
    ffresnet.eval()
    return ffresnet

def load_meru(checkpoint_path, train_config, device):
    """
    Return meru model. 
    Inputs:
    - checkpoint_path: meru checkpoint
    - train_config: meru train config
    Output:
    - meru model
    """
    _C_TRAIN = LazyConfig.load(train_config)
    meru_large = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=meru_large).load(checkpoint_path)
    return meru_large

def load_clip(checkpoint_path, train_config, device):
    """
    Return clip model. 
    Inputs:
    - checkpoint_path: clip checkpoint
    - train_config: clip train config
    Output:
    - clip model
    """
    _C_TRAIN = LazyConfig.load(train_config)
    clip_large = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=clip_large).load(checkpoint_path)
    return clip_large

def load_breeds(dataset_name, dataset_info, data_dir, num_workers=1, batch_size=5):
    """
    Return breeds dataloader for desired dataset.
    Inputs:
    - dataset_name: name of desired dataset in dataset_info
    - dataset_info: file path for dataset configs
    - data_dir: imagenet directory
    Outputs:
    - (breeds_loader, subclass_labels, superclasses, subclass_split, label_map)
    """
    superclasses = dataset_info[dataset_name]['superclasses']
    subclass_split = dataset_info[dataset_name]['subclass_split']
    label_map = dataset_info[dataset_name]['label_map']
    source_subclasses, _ = subclass_split # source
    dataset_source = datasets.CustomImageNet(data_dir, source_subclasses)
    _, breeds_loader = dataset_source.make_loaders(num_workers, batch_size, shuffle_val=False) # take val set
    # add images in increasing order by label
    val_subclass_labels = np.repeat(np.sort(np.ravel(subclass_split[0])), 50).tolist() # if shuffle is false
    return breeds_loader, val_subclass_labels, superclasses, subclass_split, label_map

def load_image_costs(cost_loc, embed_loc, breeds_loader, device):
    if not os.path.isfile(cost_loc):
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
        preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
        resnet.to(device)
        resnet.eval()
        image_embeds = embed_breeds(embed_loc, 
                                    resnet,
                                    breeds_loader, 
                                    2048, 
                                    None, 
                                    device, 
                                    override=True)['embed']
        image_embeds = torch.nn.functional.normalize(image_embeds)
        n = image_embeds.shape[0]
        image_costs = np.zeros((n ,n))
        print(n)
        for i in range(n):
            for j in range(n):
                image_costs[i, j] = image_distance(image_embeds[i], image_embeds[j])
        torch.save(image_costs, cost_loc)
    return torch.load(cost_loc)

def sup_acc(subclass_split, embed_dict):
    s_by_s = subclass_split[0] # list of subclasses by superclass
    correct = 0
    for i in range(len(embed_dict['labels'])):
        if embed_dict['preds'][i] in s_by_s[embed_dict['labels'][i]]:
            correct += 1
    return correct / len(embed_dict['labels'])
    
def sub_acc(embed_dict):
    correct = 0
    val_subclass_labels = embed_dict['subclass_labels']
    for i in range(len(val_subclass_labels)):
        if embed_dict['preds'][i] == val_subclass_labels[i]:
            correct += 1
    return correct / len(val_subclass_labels)

def clip_acc(model, dataset_name, subclass_labels):
    evaluator = ZeroShotClassificationEvaluator(ds_name=dataset_name, subclass_labels=subclass_labels)
    rd = evaluator(model)
    subclass_accuracy = rd['sub_acc']
    superclass_accuracy = rd['sup_acc']
    return subclass_accuracy, superclass_accuracy
    
def generate_cdm(model, embed_tensor, device):
    curv = model.curv.exp()
    dm = pairwise_dist(embed_tensor.to(device), embed_tensor.to(device), curv)
    
    # Condense the distance matrix
    n = dm.shape[0]
    out_size = (n * (n - 1)) // 2
    cdm = np.empty(out_size) 
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            cdm[k] = dm[i, j]
            k += 1
    return cdm

def cluster(cdm, embed_tensor, n_subclasses, n_superclasses):
    Z = linkage(cdm, method="ward")
    subclass_clustering = fcluster(Z, t=n_subclasses, criterion='maxclust')
    superclass_clustering = fcluster(Z, t=n_superclasses, criterion='maxclust')
    for i in range(len(subclass_clustering)):
        subclass_clustering[i] -= 1
        superclass_clustering[i] -= 1
    return subclass_clustering, superclass_clustering

def gen_results(f, dataset_name, names, dims):
    if not os.path.isfile(f):
        print("DATA FILE NOT FOUND")
        data_big = {}
        torch.save(data_big, f)
    data_big = torch.load(f)
    if dataset_name not in data_big:
        data_big[dataset_name] = {}
        data = data_big[dataset_name]
        data_dict = {name: np.full((len(dims)), -1, dtype=np.float64) for name in names}
        if 'hhot_super' not in data:
            data['hhot_super'] = copy.deepcopy(data_dict)
        if 'hhot_sub' not in data:
            data['hhot_sub'] = copy.deepcopy(data_dict)
        if 'acc_super' not in data:
            data['acc_super'] = copy.deepcopy(data_dict)
        if 'acc_sub' not in data:
            data['acc_sub'] = copy.deepcopy(data_dict)
        if 'ami_super' not in data:
            data['ami_super'] = copy.deepcopy(data_dict)   
        if 'ami_sub' not in data:
            data['ami_sub'] = copy.deepcopy(data_dict)
        if 'hhot_h2' not in data:
            data['hhot_h2'] = copy.deepcopy(data_dict)
        if 'purity_super' not in data:
            data['purity_super'] = copy.deepcopy(data_dict)
        if 'purity_sub' not in data:
            data['purity_sub'] = copy.deepcopy(data_dict)
        if 'nr_matches' not in data: # r for reduced
            data['nr_matches'] = copy.deepcopy(data_dict)
        if 'nr_non_matches' not in data: # r for reduced
            data['nr_non_matches'] = copy.deepcopy(data_dict)
        if 'n_matches' not in data: # r for reduced
            data['n_matches'] = copy.deepcopy(data_dict)
        if 'n_non_matches' not in data: # r for reduced
            data['n_non_matches'] = copy.deepcopy(data_dict)
    torch.save(data_big, f)

def save_results(f, model_name, dataset_name, dim, dims, overwrite,
                 superclass_distance,
                 subclass_distance,
                 superclass_accuracy,
                 subclass_accuracy,
                 superclass_ami,
                 subclass_ami,
                 h2_distance,
                 superclass_purity,
                 subclass_purity,
                 nr_matches,
                 nr_non_matches,
                 n_matches,
                 n_non_matches):
    
    data_big = torch.load(f)
    data = data_big[dataset_name]
    dim_idx = dims.index(dim)
    if overwrite or data['hhot_super'][model_name][dim_idx] == -1:
        data['hhot_super'][model_name][dim_idx] = superclass_distance
    if overwrite or data['hhot_sub'][model_name][dim_idx] == -1:
        data['hhot_sub'][model_name][dim_idx] = subclass_distance
    if overwrite or data['acc_super'][model_name][dim_idx] == -1:  
        data['acc_super'][model_name][dim_idx] = superclass_accuracy
    if overwrite or data['acc_sub'][model_name][dim_idx] == -1:  
        data['acc_sub'][model_name][dim_idx] = subclass_accuracy
    if overwrite or data['ami_super'][model_name][dim_idx] == -1: 
        data['ami_super'][model_name][dim_idx] = superclass_ami
    if overwrite or data['ami_sub'][model_name][dim_idx] == -1:
        data['ami_sub'][model_name][dim_idx] = subclass_ami
    if overwrite or data['hhot_h2'][model_name][dim_idx] == -1:
        data['hhot_h2'][model_name][dim_idx] = h2_distance
    if overwrite or data['purity_super'][model_name][dim_idx] == -1:
        data['purity_super'][model_name][dim_idx] = superclass_purity
    if overwrite or data['purity_sub'][model_name][dim_idx] == -1:
        data['purity_sub'][model_name][dim_idx] = subclass_purity
    if overwrite or data['nr_matches'][model_name][dim_idx] == -1:
        data['nr_matches'][model_name][dim_idx] = nr_matches
    if overwrite or data['nr_non_matches'][model_name][dim_idx] == -1:
        data['nr_non_matches'][model_name][dim_idx] = nr_non_matches
    if overwrite or data['n_matches'][model_name][dim_idx] == -1:
        data['n_matches'][model_name][dim_idx] = n_matches
    if overwrite or data['n_non_matches'][model_name][dim_idx] == -1:
        data['n_non_matches'][model_name][dim_idx] = n_non_matches
    torch.save(data_big, f)
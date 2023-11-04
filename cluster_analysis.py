import math
import torch

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pygraphviz as pgv


# metrics
def purity_score(true_labels, pred_labels):
    """
    Returns the purity score of a clustering.

    Inputs:
    - true_labels: ground truth labels of each sample (n_samples, )
    - pred_labels: predicted clusters (n_samples, )
    """
    contingency = contingency_matrix(true_labels, pred_labels)
    return np.sum(np.amax(contingency, axis = 0)) / np.sum(contingency)

def metrics(true_labels, pred_labels):
    """
    Prints cluster metrics (currently AMI and purity).

    Inputs:
    - true_labels: ground truth labels of each sample
    - pred_labels: clustering of each sample

    Outputs:
    - ami, purity
    """
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    purity = purity_score(true_labels, pred_labels)
    return (ami, purity)

def cluster_mode(pred_labels, true_labels):
    """
    Returns most common label within each predicted cluster, sorted by cluster label.
    Input:
    - pred_labels: (n_samples,) predicted clusters
    - true_labels: (n_samples,) true labels
    Output:
    - array-like (n_clusters, 1)
    """
    counts = {}
    for i in range(len(pred_labels)):
        if pred_labels[i] not in counts:
            counts[pred_labels[i]] = {}
        if true_labels[i] not in counts[pred_labels[i]]:
            counts[pred_labels[i]][true_labels[i]] = 0
        counts[pred_labels[i]][true_labels[i]] += 1
    n_clusters = len(np.unique(pred_labels))
    modes = np.zeros((n_clusters, 1), dtype=np.int32)
    for k, v in counts.items():
        mode = list(v.keys())[0]
        for k1 in v:
            if v[mode] < v[k1]:
                mode = k1
        modes[k] = mode
    return modes

def find_merges(c1, c2):
    """
    Returns dictionary of how samples in clustering c1 merge into clustering c2.
    Input:
    - c1: (n_samples, ) original clustering
    - c2: (n_samples, ) cluster being merged into
    Output:
    - m: {c2_i1: {c1_i1: [i1, i2,...], c1_i2: [i1, i2, ...], ...}, ...} 
    """
    m = {}
    for i in range(len(c2)):
        if c2[i] not in m:
            m[c2[i]] = {}
        if c1[i] not in m[c2[i]].keys():
            m[c2[i]][c1[i]] = []
        m[c2[i]][c1[i]].append(i)
    return m

def remove_outlier(merges, count = 5):
    """
    Removes subclasses <= count from each cluster
    Input:
    - merges: {c2_i1: {c1_i1: [i1, i2,...], c1_i2: [i1, i2, ...], ...}, ...}
    - count: int
    Output:
    (merge_copy, outliers, outlier_count)
    - merge_copy: modified merge dictionary
    - outlier_count: number of removed classes
    """
    merge_copy = {}
    outlier_count = 0
    for k, v in merges.items():
        merge_copy[k] = {}
        for k1, v1 in v.items():
            if len(v1) <= count:
                outlier_count += 1
            else:
                merge_copy[k][k1] = v1
    return merge_copy, outlier_count
    
def separate_on_matches(merges, subclass_split):
    """
    Separate merges into two dictionaries containing 1) merges that match a known superclass-superclass
    relationship and 2) non-matches.
    Input:
    - merges: {c2_i1: {c1_i1: [i1, i2,...], c1_i2: [i1, i2, ...], ...}, ...}
    - subclass_split: (n_superclass, 2) ground truth merges sorted by class label
    Output:
    - {c2_i1: {c1_i1: [i1, i2,...], c1_i2: [i1, i2, ...], ...}, ...} for matches
    - {c2_i1: {c1_i1: [i1, i2,...], c1_i2: [i1, i2, ...], ...}, ...} for non-matches
    """
    subclass_split = np.sort(subclass_split, axis=1)
    matches = {}
    non_matches = {}
    for k, v in merges.items():
        match = False
        sorted_subclasses = np.sort(list(v.keys()))
        for i in range(subclass_split.shape[0]):
            if np.array_equal(sorted_subclasses, subclass_split[i]): 
                match = True
        if match:
            matches[k] = v
            # print(sorted_subclasses)
        else:
            # print(sorted_subclasses)
            non_matches[k] = v
    return matches, non_matches

def ltensor_to_lint(l):
    """
    Convert list of single-element tensors to list of numbers.
    Input:
    - l: list of tensors
    Output:
    - k: list of numbers
    """
    k = []
    for i in l:
        if isinstance(i, torch.Tensor):
            k.append(i.cpu().item())
        else:
            k.append(i)
    return k
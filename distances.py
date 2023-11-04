import ot
import numpy as np
import torch

def image_distance(i1, i2):
    """
    Calculate distance between two image representations.
    Input:
    - i1: (x_dim, y_dim)
    - i2: (x_dim, y_dim)
    
    Output:
    - d: distance as a float
    """
    d = torch.linalg.vector_norm(i1 - i2) # euclid distance of embeds
    return d

def hhot(true_labels, pred_labels, cost):
    """
    Returns H1 HHOT distance between true and cluster labels.
    Input:
    - true_labels: (n_samples, ) true labels per sample
    - pred_labels: (n_samples, ) predicted cluster labels per sample
    - cost: (n_samples, n_samples) cost matrix between image samples
    Note: true_labels, cluster_labels, cost should share same order
    Output:
    - float: distance
    - (n_clusters, n_clusters): array-like cluster distance matrix where axis = 0 are true clusters
    and axis = 1 are predicted clusters, ordered in increasing order on both axes
    """
    labels1 = np.array(true_labels) # ground truth
    labels2 = np.array(pred_labels) # clusters
    labels1_groups = np.unique(labels1) # returns sorted by index
    labels2_groups = np.unique(labels2) # returns sorted by index
    hhot_dist = np.zeros((len(labels1_groups), len(labels2_groups)))
    for i, l1 in enumerate(labels1_groups): # ith true cluster
        for j, l2 in enumerate(labels2_groups): # jth pred cluster
            c1_index_b = labels1 == l1
            c2_index_b = labels2 == l2
            c1_index = c1_index_b.nonzero()
            c2_index = c2_index_b.nonzero()
            # print(cost.shape)
            temp = np.take(cost, c1_index, axis = 0)
            # print(temp.shape)
            temp2 = np.take(temp[0], c2_index, axis = 1).squeeze(axis = 1)
            hhot_dist[i, j] = ot.sinkhorn2(ot.unif(temp2.shape[0]), ot.unif(temp2.shape[1]), temp2/temp2.max(), 0.1, numItermax=2000)*temp2.max()
    return ot.sinkhorn2(ot.unif(hhot_dist.shape[0]), ot.unif(hhot_dist.shape[1]), hhot_dist/hhot_dist.max(), 0.1)*hhot_dist.max(), hhot_dist

def hhot_h2(cluster_cost, true_superclass_merge, pred_superclass_merge):
    """
    Input:
    - cluster_cost: (n_clusters, n_clusters): cost matrix where axis = 0 contain true clusters
    and axis = 1 contain predicted clusters, ordered in increasing order of cluster indices in true_superclass_merge 
    and pred_superclass_merge on both axes
    - true_superclass_merge: {} where superclass indices are sorted keys and values are nested {} with subclass 
    indices as keys
    - pred_superclass_merge: {} where superclass indices are sorted keys and values are nested {} with subclass 
    indices as keys
    Output:
    - float: H2 distance between two clusterings
    """
    # array of merged subclasses in order that they appear in cluster_cost
    true_subclass_idx = []
    for k, v in true_superclass_merge.items():
        for k1, v1 in v.items():
            true_subclass_idx.append(k1)
    true_subclass_idx.sort() # indices of classes correspond w/indices on axis 0
    
    pred_subclass_idx = []
    for k, v in pred_superclass_merge.items():
        for k1, v1 in v.items():
            pred_subclass_idx.append(k1)
    pred_subclass_idx.sort() # indices of classes correspond w/indices on axis 1
    
    # cost matrix
    hhot_h2_dist = np.zeros((len(true_superclass_merge), len(pred_superclass_merge)))
    for i, (kt, vt) in enumerate(true_superclass_merge.items()):
        c1_index = []
        for key in vt:
            c1_index.append(true_subclass_idx.index(key)) # add cluster's corresponding index in cost matrix
        for j, (kp, vp) in enumerate(pred_superclass_merge.items()):
            c2_index = []
            for key in vp:
                c2_index.append(pred_subclass_idx.index(key))
            temp = np.take(cluster_cost, c1_index, axis=0) # take rows corresponding to true subclass clusters
            temp2 = np.take(temp, c2_index, axis=1) # take cols corresponding to pred subclass clusters
            hhot_h2_dist[i, j] = ot.sinkhorn2(ot.unif(temp2.shape[0]), ot.unif(temp2.shape[1]), temp2/temp2.max(), 0.2, numItermax=5000)*temp2.max()
    return ot.sinkhorn2(ot.unif(hhot_h2_dist.shape[0]), ot.unif(hhot_h2_dist.shape[1]), hhot_h2_dist/hhot_h2_dist.max(), 0.1, numItermax=2000)*hhot_h2_dist.max()
import os
import torch
import numpy as np

from tqdm import tqdm
from meru.meru.evaluation.classification import _encode_dataset
from torchvision.transforms import ToTensor
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision.models import resnet50, ResNet50_Weights
preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

def embed_breeds(f, model, breeds_loader, dim, subclass_labels, device, mrl=False, override=False):
    """
    Returns dim-dimensional embeddings of breeds_loader using the given resnet model
    Input:
    - f: (str) file path to store embeddings
    - model: mrl or resnet
    - breeds_loader: breeds dataloader
    - dim: (int) dimension
    - subclass_labels: subclass labels
    - device: 
    - mrl: (default False) whether model is mrl
    Output: 
    - results = {'embed': array-like (n_samples, dim), 
                 'labels': [n_samples, ], 
                 'preds': [n_samples, ], 
                 'subclass_labels': subclass_labels}
    """
    if override or not os.path.isfile(f):
        with torch.no_grad():
            print('embedding breeds...')
            extract = create_feature_extractor(model, return_nodes={'flatten': 'second_last'}).to(device)
            extract.eval()
            embed_tensor = torch.zeros((len(breeds_loader.dataset), dim))
            results = {'embed': embed_tensor, 'labels': [], 'preds': [], 'subclass_labels': subclass_labels}
            row_idx = 0
            for image in tqdm(breeds_loader): # batched
                # image in form [tensor of images, tensor of labels]
                img_tensor = image[0].to(device)
                labels = image[1]
                embeds = extract(preprocess(img_tensor.float()))['second_last']
                for i in range(embeds.shape[0]):
                    results['embed'][row_idx, :] = embeds[i][:dim]
                    results['labels'].append(labels[i])
                    if not mrl: # if not mrl
                        results['preds'].append(torch.argmax(model.fc(torch.unsqueeze(embeds[i][:dim], 0))))
                    else: # if mrl 
                        logits = getattr(model.fc, f"nesting_classifier_{model.fc.nesting_list.index(dim)}")(embeds[i][:dim])
                        results['preds'].append(torch.argmax(logits))
                    row_idx += 1
            open(f, 'w').close()
            torch.save(results, f)
    return torch.load(f)
    
def embed_breeds_meip(f, model, breeds_loader, subclass_labels=None):
    """
    Returns dim-dimensional embeddings of breeds_loader using the given model
    
    Input:
    - f: (str) file path to store embeddings
    - model: meru or clip
    - breeds_loader: dataloader from breeds
    - subclass_labels: (optional) subclass labels in same order as breeds_loader
    Output: 
    - results = {'embed': array-like (n_samples, dim), 
                 'labels': [n_samples, ], 
                 'preds': [], 
                 'subclass_labels': subclass_labels}
    """
    if not os.path.isfile(f):
        with torch.no_grad():
            print('embedding breeds...')
            results = {'embed': None, 'labels': [], 'preds': [], 'subclass_labels': subclass_labels}
            # Gather normalized for CLIP or projected for MERU
            all_image_feats, all_labels = _encode_dataset(breeds_loader, model, True) 
            results['embed'] = all_image_feats
            results['labels'] = [i.item() for i in all_labels]
            torch.save(results, f)
    return torch.load(f)

def convert_pca(model, dim, device, embed_dict):
    """
    Perform SVD to reduce the dimensionality of embeddings through the final weight matrix in 
    a resnet model
    Inputs:
    - model: resnet model
    - dim: dimension to reduce to
    - embed_dict: {'embed': array-like (n_samples, 2048), 
                 'labels': [n_samples, ], 
                 'preds': [n_samples, ], 
                 'subclass_labels': subclass_labels}
    Outputs:
    - results = {'embed': array-like (n_samples, dim), 
                 'labels': [n_samples, ], 
                 'preds': [n_samples, ], 
                 'subclass_labels': subclass_labels}
    """
    with torch.no_grad():
        # svd
        u, s, v = torch.linalg.svd(model.fc.weight.T, full_matrices=False) # 1000, 2048 -> 2048, 1000
        # print(u.shape, s.shape, v.shape)
        s[dim:] = 0
        r = u * s # reduction matrix # 2048, 1000 -> cols past dim are 0
        embed_old = embed_dict['embed'].to(device) # n_samples, 2048
        embed_red = embed_old @ r # n_samples, 1000, w/0s on end
        embed_dict['embed'] = embed_red[:, :dim].cpu()
        embed_dict['preds'] = torch.argmax(embed_red @ v, dim=1).tolist()
        return embed_dict

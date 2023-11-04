# can improve by pass in dirs, and have names be hardcoded
import argparse

from cluster_analysis import *
from embeddings import *
from distances import *
from eval_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model name between mrl, ff, meru, clip')
parser.add_argument('--dataset_dir', help='location of dataset info')
parser.add_argument('--imagenet_dir', help='location of imagenet dataset')

def main(dim, dataset_dir, imagenet_dir, model_name, device, overwrite=True):
    # these two lists are to be used to set up results dict
    names = ['mrl', 'ff', 'meru', 'clip']
    dims = [2**n for n in range(3, 12)] 
    
    # model initialization
    if model_name == 'mrl':
        tf = './pt/r50_mrl1_e0_ff2048.pt' # location of mrl checkpoint
        model = load_mrl(tf, dims, device)
    elif model_name == 'ff':
        tf = f'./pt/r50_mrl0_e0_ff{dim}.pt' # location of ff checkpoint
        model = load_ff(tf, dim, device)
    elif model_name == 'meru':
        checkpoint_path = f'./checkpoints/meru{dim}_vit_l.pth' # location of meru checkpoint
        train_config = f'./meru/configs/train_meru{dim}_vit_l.py' # location of meru model config
        model = load_meru(checkpoint_path, train_config, device)   
    elif model_name == 'clip':
        checkpoint_path = f'./checkpoints/clip{dim}_vit_l.pth' # location of clip checkpoint
        train_config = f'./meru/configs/train_clip{dim}_vit_l.py' # location of clip config
        model = load_clip(checkpoint_path, train_config, device)
    
    # load datasets
    imagenet_labels = []
    with open('../datasets/imagenet-labels.txt') as f: # location of imagenet labels
        for row in f:
            imagenet_labels.append(row.strip())
    dataset_info = torch.load(dataset_dir)
    for dataset_name in dataset_info:
        print(f'{model_name}, {dim}, {dataset_name}')
        breeds_loader, val_subclass_labels, superclasses, subclass_split, label_map = load_breeds(dataset_name, dataset_info, imagenet_dir)
        if model_name == 'mrl':
            embed_dict = embed_breeds(f'./embeds/mrl_rn{dim}_spherical_{dataset_name}.pt', # location of stored embeds
                                      model, 
                                      breeds_loader, 
                                      dim, 
                                      val_subclass_labels, 
                                      device, 
                                      True)
            embed_dict['embed'] = torch.nn.functional.normalize(embed_dict['embed'])
        elif model_name == 'ff':
            embed_dict = embed_breeds(f'./embeds/ff_rn{dim}_spherical_{dataset_name}.pt', 
                                      model, 
                                      breeds_loader, 
                                      dim, 
                                      val_subclass_labels, 
                                      device)
            embed_dict['embed'] = torch.nn.functional.normalize(embed_dict['embed'])
        elif model_name == 'meru':
            embed_dict = embed_breeds_meip(f'./embeds/meru{dim}_hyperbolic_{dataset_name}.pt', 
                                           model, 
                                           breeds_loader, 
                                           val_subclass_labels)
        elif model_name == 'clip':
            embed_dict = embed_breeds_meip(f'./embeds/clip{dim}_spherical_{dataset_name}.pt', 
                                           model, 
                                           breeds_loader, 
                                           val_subclass_labels)
            
        
        # convert tensor elements into integers
        embed_dict['labels'] = ltensor_to_lint(embed_dict['labels'])
        embed_dict['subclass_labels'] = ltensor_to_lint(embed_dict['subclass_labels'])
        
        # accuracy
        n_subclasses = dataset_info[dataset_name]['n_subclasses']
        n_superclasses = dataset_info[dataset_name]['n_superclasses'] 
        if model_name == 'meru' or model_name == 'clip':
            subclass_accuracy, superclass_accuracy = clip_acc(model, dataset_name)
        else:
            superclass_accuracy = sup_acc(subclass_split, embed_dict)
            subclass_accuracy = sub_acc(embed_dict)
        
        # clustering
        if model_name == 'meru':
            cdm = generate_cdm(model, embed_dict['embed'], device)
        else:
            cdm = pdist(embed_dict['embed'])
        subclass_clustering, superclass_clustering = cluster(cdm, 
                                                             embed_dict['embed'],
                                                             n_subclasses,
                                                             n_superclasses)
        subclass_ami, subclass_purity = metrics(embed_dict['subclass_labels'], subclass_clustering)
        superclass_ami, superclass_purity = metrics(embed_dict['labels'], superclass_clustering)
        
        # merges
        m = dict(sorted(find_merges(embed_dict['subclass_labels'], superclass_clustering).items())) # defualt sorts kv by key, create new dict()       
        matches, non_matches = separate_on_matches(m, subclass_split[0])
        reduced_m, _, _ = remove_outlier(m, 10)
        rmatches, rnon_matches = separate_on_matches(reduced_m, subclass_split[0])
        nr_matches = len(rmatches)
        nr_non_matches = len(rnon_matches)
        n_matches = len(matches)
        n_non_matches = len(non_matches)
        
        # ot distance
        cost_loc = f'./files_new/{dataset_name}_image_costs.pt' # location for dataset image costs
        embed_loc = f'./files_new/resnet_{dataset_name}.pt' # location to store embeds
        image_costs = load_image_costs(cost_loc, embed_loc, breeds_loader, device)
        superclass_distance, _ = hhot(embed_dict['labels'], superclass_clustering, image_costs)
        subclass_distance, subclass_dist = hhot(embed_dict['subclass_labels'], subclass_clustering, image_costs)
        true_superclass_merge = dict(sorted(find_merges(embed_dict['subclass_labels'], embed_dict['labels']).items()))
        pred_superclass_merge = dict(sorted(find_merges(subclass_clustering, superclass_clustering).items()))
        h2_distance = hhot_h2(subclass_dist, true_superclass_merge, pred_superclass_merge)
        
        # save results
        f = './files/results.pt'
        gen_results(f, dataset_name, names, dims)
        save_results(f, model_name, dataset_name, dim, dims, overwrite,
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
                     n_non_matches)

# evaluate chosen model against all dimensions of all datasets
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    _A = parser.parse_args()
    model_name = _A.model
    dataset_dir = _A.dataset_dir
    imagenet_dir = _A.imagenet_dir
    if (model_name != 'mrl' and 
       model_name != 'ff' and 
       model_name != 'meru' and 
       model_name != 'clip'):
        raise RuntimeError()
    if model_name == 'meru' or model_name == 'clip':
        dims = [512]
    else:
        dims = [2**n for n in range(3, 12)]
    for dim in dims:
        main(dim, dataset_dir, imagenet_dir, model_name, device)
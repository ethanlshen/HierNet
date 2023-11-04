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
parser.add_argument('--results_dir', help='where to store results')

def main(dim, dims, model_name, dataset_info, imagenet_dir, results_dir, device, overwrite=True):
    # model initialization
    if model_name == 'ff':
        tf = f'./pt/r50_mrl0_e0_ff2048.pt' # location of ff checkpoint
        model = load_ff(tf, 2048, device)
    else:
        raise(RuntimeError())
    
    # load datasets
    imagenet_labels = []
    with open('../datasets/imagenet-labels.txt') as f: # location of imagenet labels
        for row in f:
            imagenet_labels.append(row.strip())
    for dataset_name in dataset_info:
        print(f'{model_name}, {dim}, {dataset_name}')
        breeds_loader, val_subclass_labels, superclasses, subclass_split, label_map = load_breeds(dataset_name, dataset_info, imagenet_dir)
        
        # load embeds
        if model_name == 'mrl':
            embed_dict = torch.load(f'./embeds/mrl_rn2048_spherical_{dataset_name}.pt') # location of base embed
        elif model_name == 'ff':
            embed_dict = torch.load(f'./embeds/ff_rn2048_spherical_{dataset_name}.pt') 

        # convert tensor elements into integers
        embed_dict['labels'] = ltensor_to_lint(embed_dict['labels'])
        embed_dict['subclass_labels'] = ltensor_to_lint(embed_dict['subclass_labels'])

        # PCA
        n_subclasses = dataset_info[dataset_name]['n_subclasses']
        n_superclasses = dataset_info[dataset_name]['n_superclasses'] 
        embed_dict = convert_pca(model, dim, device, embed_dict)
        embed_dict['embed'] = torch.nn.functional.normalize(embed_dict['embed'])
        
        # accuracy
        superclass_accuracy = sup_acc(subclass_split, embed_dict)
        subclass_accuracy = sub_acc(embed_dict)
        print(superclass_accuracy, subclass_accuracy)
        
        # clustering
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
        gen_results(results_dir, dataset_name, [model_name], dims)
        save_results(results_dir, model_name, dataset_name, dim, dims, overwrite,
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
    results_dir = _A.results_dir
    
    names = ['mrl', 'ff']
    if model_name not in names:
        raise RuntimeError()
    dims = [2**n for n in range(3, 12)]
    # running
    dataset_info = torch.load(dataset_dir)
    for dim in dims:
        if dim <= 1000:
            main(dim, dims, model_name, dataset_info, imagenet_dir, results_dir, device)
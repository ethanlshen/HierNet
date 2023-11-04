# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from loguru import logger
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from meru import lorentz as L
from meru.evaluation.catalog import DatasetCatalog
from meru.evaluation.class_names import CLASS_NAMES
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
from meru.evaluation.classification import _encode_dataset

from robustness import datasets
from robustness.tools.vis_tools import show_image_row


class ZeroShotClassificationEvaluator:
    """
    Evaluate trained models for zero-shot image classification, wherein the entire
    model is transferred to the downstream task without additional training. This
    protocol is similar to CLIP: the classifier weights are constructed by encoding
    text prompts of class labels using text encoder.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        ds_name,
        subclass_labels,
        num_workers: int = 4,
        image_size: int = 224,
        info_dir='./files/dataset_info.pt',
        data_dir='/mnt/nvme0n1p2/data/ImageNet-1K',
        datasets_and_prompts={
            "imagenet": [
                "i took a picture : itap of a {}.",
                "pics : a bad photo of the {}.",
                "pics : a origami {}.",
                "pics : a photo of the large {}.",
                "pics : a {} in a video game.",
                "pics : art of the {}.",
                "pics : a photo of the small {}.",
            ],
        },
    ):
        """
        Args:
            datasets_and_prompts: Dictionary mapping between dataset name and
                a list of prompt templates to fill using its class names. Add
                a single `{}` in prompt to fill with class name.
            data_dir: Dataloader object for desired dataset. 
            num_workers: Number of CPU works to parallelize data loading for
                extracting features.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
        """
        self._datasets_and_prompts = datasets_and_prompts
        
        # breeds loader creation
        dataset_info = torch.load(info_dir)
        self.subclass_split = dataset_info[ds_name]['subclass_split']
        source_subclasses, _ = self.subclass_split # pick source or target
        dataset_source = datasets.CustomImageNet(data_dir, source_subclasses)
        _, self.loader = dataset_source.make_loaders(num_workers, 128, shuffle_val=False)
        self.subclass_labels = subclass_labels
        self._num_workers = num_workers
        self._image_transform = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    @torch.inference_mode()
    def __call__(self, model: MERU | CLIPBaseline) -> dict[str, float]:
        model = model.eval()
        tokenizer = Tokenizer()

        # Collect results per task in this dict:
        results_dict = {}

        for dname, prompts in self._datasets_and_prompts.items():
            # ----------------------------------------------------------------
            # Make zero-shot classifier using class name and prompts.
            # ----------------------------------------------------------------
            class_names = CLASS_NAMES[dname] # NEED MAINTAIN IMAGENET NAME

            # Collect text features of each class.
            all_class_feats: list[torch.Tensor] = []

            for name in class_names:
                # Fill prompt templates with class name and tokenize them.
                class_prompts = [_pt.format(name) for _pt in prompts]

                class_prompt_tokens = tokenizer(class_prompts)
                class_feats = model.encode_text(class_prompt_tokens, project=False)

                if isinstance(model, MERU):
                    # Ensemble in the tangent space, then project to Hyperboloid.
                    class_feats = class_feats.mean(dim=0)
                    class_feats = class_feats * model.textual_alpha.exp()
                    class_feats = L.exp_map0(class_feats, model.curv.exp())
                else:
                    # Ensemble prompt features: normalize -> average -> normalize.
                    class_feats = F.normalize(class_feats, dim=-1)
                    class_feats = class_feats.mean(dim=0)
                    class_feats = F.normalize(class_feats, dim=-1)

                all_class_feats.append(class_feats)

            # shape: (num_classes, embed_dim)
            classifier = torch.stack(all_class_feats, dim=0) # store prompt embeds as one class per row
            # ----------------------------------------------------------------

            # Extract image features and labels from the test split of required dataset.
            image_feats, labels = _encode_dataset(self.loader, model, project=True)

            # Features returned by this function will be on CPU, move to device:
            image_feats = image_feats.to(model.device)

            # Measure model performance according to accuracy metric of the dataset.
            acc_meter = MulticlassAccuracy(DatasetCatalog.NUM_CLASSES[dname])

            # Evaluate in small batches of 256 instances against subclass labels.
            preds = None
            for _feats, _labels in zip(image_feats.split(256), torch.as_tensor(self.subclass_labels).split(256)): 
                # Compute pairwise similarity depending on model type:
                if isinstance(model, MERU):
                    scores = L.pairwise_inner(_feats, classifier, model.curv.exp()) # outputs inner product
                else:
                    scores = _feats @ classifier.T
                    
                # Store predictions
                if preds == None:
                    preds = torch.argmax(scores, dim=1)
                else:
                    preds = torch.cat([preds, torch.argmax(scores, dim=1)])

                acc_meter(scores.cpu(), _labels)

            subclass_accuracy = acc_meter.compute() * 100.0
            results_dict['sub_acc'] = subclass_accuracy
            
            # Calculate superclass accuracy
            correct = 0
            for n_i, i in enumerate(labels):
                if preds[n_i].item() in self.subclass_split[0][i]:
                    correct += 1
            results_dict['sup_acc'] = 100.0 * correct / len(labels)
            
            # Calculate subclass accuracy
            correct = 0
            for n_i, i in enumerate(self.subclass_labels):
                if preds[n_i].item() == i:
                    correct += 1
            results_dict['sub_acc'] = 100.0 * correct / len(labels)

            results_dict['preds'] = preds
            # logger.info(
            #     f"Zero-shot evaluation: {dname}, {len(image_feats)} images, "
            #     f"{len(class_names)} classes [acc.: {accuracy:.1f}%] "
            # )

        return results_dict

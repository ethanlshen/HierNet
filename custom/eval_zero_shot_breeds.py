# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from meru.config import LazyCall as L
from meru.evaluation.classification import ZeroShotClassificationEvaluator


evaluator = L(ZeroShotClassificationEvaluator)(
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
    data_dir="datasets/eval",
    image_size=224,
    num_workers=4,
)

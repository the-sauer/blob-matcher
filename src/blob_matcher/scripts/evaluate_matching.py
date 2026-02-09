# Copyright 2025 Hendrik Sauer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re


import torch
import torchvision


from blob_matcher.hardnet.eval_metrics import ErrorRateAt95Recall
from blob_matcher.hardnet.losses import distance_matrix_vector
from blob_matcher.hardnet.models import HardNet


def main():
    image_descriptions = {
        1: "very oblique, close",
        3: "very oblique, far",
        5: "medium oblique, close",
        7: "medium oblique, far",
        9: "fronto-parallel, close",
        11: "fronto-parallel, far",
        13: "fronto-parallel, very far",
    }

    for date in [
        "2025_11_05",
        "2025_11_06",
    ]:
        for scale in [96, 128]:
            checkpoint = 199
            print(f"\n# Evaluating {date} with lambda={scale}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HardNet(patch_size=32)
            model.load_state_dict(torch.load(
                f"./src/blob_matcher/data/models/default.pth",
                weights_only=False,
                map_location=device
            )["state_dict"])
            model.to(device)
            model.eval()
            positive_path_regex = re.compile("(\\d+)_(\\d+).png")
            dataset_path = "./data/datasets/new/real/validation"
            patch_files = os.listdir(os.path.join(dataset_path, f"patches/{scale}/positives"))

            fpr95_sum = 0
            fpr95_num = 0
            for i in range(0, 123):
                
                regex = re.compile(f"{i:04}_\\d+\\.png")
                patches = list(filter(lambda f: regex.match(f) is not None, patch_files))
                if len(patches) == 0:
                    continue

                anchor_patches = torch.empty((len(patches), 1, 32, 32)).to(device)
                positive_patches = torch.empty((len(patches), 1, 32, 32)).to(device)

                for j, patch_file in enumerate(patches):
                    match = positive_path_regex.search(os.path.basename(patch_file))
                    board_idx, blob_idx = match.group(1), match.group(2)
                    positive_patches[j] = torchvision.io.decode_image(os.path.join(dataset_path, f"patches/{scale}/positives/{board_idx}_{blob_idx}.png"), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255
                    anchor_patches[j] = torchvision.io.decode_image(os.path.join(dataset_path, f"patches/{scale}/anchors/{board_idx}_{blob_idx}.png"), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255
                garbage_patch_files = os.listdir(os.path.join(dataset_path, f"patches/{scale}/garbage"))
                garbage_patch_files = list(filter(lambda f: regex.match(f) is not None, garbage_patch_files))
                garbage_patches = torch.empty((len(garbage_patch_files), 1, 32, 32)).to(device)
                for j, patch_file in enumerate(garbage_patch_files):
                    garbage_patches[j] = torchvision.io.decode_image(os.path.join(dataset_path, f"patches/{scale}/garbage/{patch_file}"), torchvision.io.ImageReadMode.GRAY)

                anchor_features, _ = model(anchor_patches)
                positive_features, _ = model(positive_patches)
                garbage_features, _ = model(garbage_patches)

                distances = distance_matrix_vector(anchor_features, torch.concat((positive_features, garbage_features))).detach().cpu().numpy().flatten()
                labels = torch.eye(anchor_features.size(0), positive_features.size(0) + garbage_features.size(0)).cpu().numpy().flatten()
                fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
                print(f"{fpr95=} for image {i} with {anchor_features.size(0)} features")
                fpr95_num += distances.size
                fpr95_sum += distances.size * fpr95

            avg_fpr95 = fpr95_sum / fpr95_num
            print(f"{avg_fpr95=}")

# Copyright 2019 EPFL, Google LLC
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


import numpy as np
import torch
import torchvision


class Augmentor(object):
    """Class to augment data by randomly jittering
       the anchor keypoints' coordinates, orientations and scales at training time
    """
    def __init__(
            self,
            cfg,
            device,
    ):
        self.cfg = cfg
        self.padTo = cfg.TRAINING.PAD_TO
        self.device = device

        # location augmentation l ~ min(G(0.5), 5) in pixels, which is a bit more skewed than a binomial distribution
        #self.geometr = torch.distributions.geometric.Geometric(probs=0.5)
        self.binom = torch.distributions.binomial.Binomial(total_count=3,
                                                           probs=0.1)
        # orientation augmentation o ~ N(0,25/2) in degrees, such that 95% of samples fall within \pm 25
        self.normal = torch.distributions.normal.Normal(loc=0, scale=25.0)
        # scale ratio augmentation r ~ Gamma(shape=0.5,rate=1/scale=1.0), or skew it even more to the left ...
        self.gamma = torch.distributions.gamma.Gamma(1.5, 3.5)

        self.maxHeight = self.padTo  # standardized size of the square images
        self.pi = torch.Tensor([np.pi]).to(self.device)

    def augmentLoc(self, kpLoc):
        # perform augmentation on anchor keypoint coordinates
        batchSize = kpLoc.shape[0]
        #augmLoc = torch.min(self.binom.sample((batchSize, 2)).squeeze(), torch.tensor(5.0)).to(self.device)
        augmLoc = self.binom.sample((batchSize, 2)).to(self.device)
        kpLoc = self.maxHeight * (
            kpLoc + 1
        ) / 2  # invert mapping from pixel space to standard grid space (or sample directly in standard space)
        kpLoc += augmLoc  # augment in pixel-space
        kpLoc = 2 * kpLoc / self.maxHeight - 1  # re-map to standardized grid space [-1, +1]
        return kpLoc

    def augmentRot(self, rotation):
        # perform augmentation on anchor keypoint orientations
        batchSize = rotation.shape[0]
        augmRot = self.normal.sample((batchSize, 1)).squeeze().to(self.device)
        augmRot = self.deg2rad(
            augmRot)  # map to radians (or sample directly in radian space)
        rotation += augmRot
        return rotation % (2 * np.pi)

    def augmentScale(self, scaling_a, scaling_p):
        # perform augmentation on keypoint scale ratios
        batchSize = scaling_a.shape[0]
        # elementwise comparison to check which scale dominates
        maxAnchor = (scaling_a >= scaling_p).float()
        augmRatio = self.gamma.sample((batchSize, 1)).squeeze().to(self.device)

        # augment the ratio, this currently only increases the ratio (additively)
        scaling_a += maxAnchor * scaling_p * augmRatio
        scaling_p += (1 - maxAnchor) * scaling_a * augmRatio
        return scaling_a, scaling_p

    def deg2rad(self, deg):
        rad = deg * self.pi / 180.0
        return rad

class BlobinatorTrainingData(torch.utils.data.Dataset):
    def __init__(self, cfg, path):
        self.cfg = cfg
        self.path = path
        patch_size = self.cfg.INPUT.IMAGE_SIZE
        self.resize = torchvision.transforms.Resize((patch_size, patch_size))
        self.positive_path_regex = re.compile("(\\d+)_(\\d+).png")
        self.patch_paths = list(filter(
            lambda p: self.positive_path_regex.match(p) is not None,
            os.listdir(os.path.join(path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "positives"))
        ))

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        filename = self.patch_paths[idx]
        match = self.positive_path_regex.search(os.path.basename(filename))
        try:
            board_idx, blob_idx = match.group(1), match.group(2)
        except AttributeError as e:
            print(f"{os.path.basename(filename)} is not a valid positive patch name")
            raise e
        positive_patch = self.resize(
            torchvision.io.decode_image(
                os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "positives", filename),
                torchvision.io.ImageReadMode.GRAY
            ).to(torch.float32) / 255
        )
        anchor_patch = self.resize(
            torchvision.io.decode_image(
                os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "anchors", f"{board_idx}_{blob_idx}.png"),
                torchvision.io.ImageReadMode.GRAY
            ).to(torch.float32) / 255)
        try:
            garbage_patch = self.resize(torchvision.io.decode_image(os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "garbage", f"{board_idx}_{blob_idx}.png"), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255)
            garbage_available = True
        except (FileNotFoundError, RuntimeError):
            garbage_patch = torch.zeros(1, 32, 32)
            garbage_available = False

        return positive_patch, anchor_patch, garbage_patch, garbage_available


class BlobinatorValidationData(torch.utils.data.Dataset):
    def __init__(self, cfg, path):
        self.cfg = cfg
        self.path = path
        patch_size = self.cfg.INPUT.IMAGE_SIZE
        self.resize = torchvision.transforms.Resize((patch_size, patch_size))
        self.positive_path_regex = re.compile("(\\d+)_(\\d+).png")
        self.patch_paths = list(filter(
            lambda p: self.positive_path_regex.match(p) is not None,
            os.listdir(os.path.join(path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "positives"))
        ))
        self.permutation = torch.randperm(len(self.patch_paths) * 3)[
            :min(len(self.patch_paths) * 3, self.cfg.TEST.MAX_SAMPLES)
        ]

    def __len__(self):
        return len(self.permutation)

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        filename = self.patch_paths[idx // 3]
        match = self.positive_path_regex.search(filename)
        board_idx, blob_idx = match.group(1), match.group(2)
        positive_patch = self.resize(torchvision.io.decode_image(os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "positives", filename), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255)
        if idx % 3 == 0:
            anchor_patch = self.resize(torchvision.io.decode_image(os.path.join(os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}",  "anchors", f"{board_idx}_{blob_idx}.png")), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255)
            return positive_patch, anchor_patch, 1
        elif idx % 3 == 1:
            anchor_patch = self.resize(torchvision.io.decode_image(os.path.join(os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}",  "false_anchors", filename)), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255)
            return positive_patch, anchor_patch, 0
        else:
            garbage_patch = self.resize(torchvision.io.decode_image(os.path.join(os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}",  "garbage", filename)), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255)
            return positive_patch, garbage_patch, 0
        

class BlobinatorValidationPreBatchedData(torch.utils.data.Dataset):
    def __init__(self, cfg, path):
        self.cfg = cfg
        self.path = path
        patch_size = self.cfg.INPUT.IMAGE_SIZE
        self.resize = torchvision.transforms.Resize((patch_size, patch_size))
        self.positive_path_regex = re.compile("(\\d+)_(\\d+).png")
        self.image_ids = list(set(map(
            lambda m: m.group(1),
            filter(
                lambda m: m is not None,
                map(
                    lambda p: self.positive_path_regex.match(p),
                    os.listdir(os.path.join(path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "positives"))
                )
            )
        )))
        self.positive_ids = {
            i: list(map(
                lambda m: m.group(2),
                filter(
                    lambda m: m is not None and m.group(1) == i,
                    map(
                        lambda p: self.positive_path_regex.match(p),
                        os.listdir(os.path.join(path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "positives"))
                    )
                )
            )) for i in self.image_ids
        }
        self.garbage_ids = {
            i: list(map(
                lambda m: m.group(2),
                filter(
                    lambda m: m is not None and m.group(1) == i,
                    map(
                        lambda p: self.positive_path_regex.match(p),
                        os.listdir(os.path.join(path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}", "garbage"))
                    )
                )
            )) for i in self.image_ids
        }
        pass

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_idx = self.image_ids[idx]
        anchor_patches = torch.empty((len(self.positive_ids[image_idx]), 1, self.cfg.INPUT.IMAGE_SIZE,  self.cfg.INPUT.IMAGE_SIZE))
        positive_patches = torch.empty((len(self.positive_ids[image_idx]), 1, self.cfg.INPUT.IMAGE_SIZE,  self.cfg.INPUT.IMAGE_SIZE))
        garbage_patches = torch.empty((len(self.garbage_ids[image_idx]), 1, self.cfg.INPUT.IMAGE_SIZE,  self.cfg.INPUT.IMAGE_SIZE))
        for blob_idx in range(anchor_patches.size(0)):
            anchor_patches[blob_idx] = self.resize(
                torchvision.io.decode_image(
                    os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}",  "anchors", f"{image_idx}_{self.positive_ids[image_idx][blob_idx]}.png"),
                    torchvision.io.ImageReadMode.GRAY
                ).to(torch.float32) / 255
            )
            positive_patches[blob_idx] = self.resize(
                torchvision.io.decode_image(
                    os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}",  "positives", f"{image_idx}_{self.positive_ids[image_idx][blob_idx]}.png"),
                    torchvision.io.ImageReadMode.GRAY
                ).to(torch.float32) / 255
            )
        for blob_idx in range(garbage_patches.size(0)):
            garbage_patches[blob_idx] = self.resize(
                torchvision.io.decode_image(
                    os.path.join(self.path, "patches", f"{int(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR)}",  "garbage", f"{image_idx}_{self.positive_ids[image_idx][blob_idx]}.png"),
                    torchvision.io.ImageReadMode.GRAY
                ).to(torch.float32) / 255
            )
        return anchor_patches, positive_patches, garbage_patches

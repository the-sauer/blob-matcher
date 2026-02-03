# Copyright 2026 Hendrik Sauer
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

import time

import torch
import torchvision

from blob_matcher.hardnet.models import HardNet
from blob_matcher.keypoints import (
    conic_to_ellipse,
    ellipse_to_affine,
    get_patch,
    keypoint_to_mapped_conic,
    keypoints_to_torch
)
from blob_matcher.utils import read_json
from blob_matcher.utils.functional import curry


def measure(model: HardNet, img: torch.Tensor, num_patches: int, keypoints, device):
    iterations = 100
    A = torch.stack(list(map(ellipse_to_affine, keypoints[:num_patches]))).to(device)
    patches = get_patch(img, A, None, psf=96, resolution=model.patch_size)
    start = time.time()
    for _ in range(iterations):
        model(patches)
    end = time.time()
    print(f"Forward pass for {num_patches} patches with size {model.patch_size} took on average {((end-start) / iterations) * 1000} ms")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torchvision.io.decode_image(
        "./data/datasets/2025_11_04/easy/validation/warped_images/0008.png",
        torchvision.io.ImageReadMode.GRAY
    ).to(torch.float32).to(device) / 255
    homography = torch.load("./data/datasets/2025_11_04/easy/validation/homographies.pt")[8].to(device)
    blobboard_info = read_json("./data/patterns/blob_board_Wn4.json")
    keypoints = list(
        filter(
            lambda x: x is not None,
            map(
                conic_to_ellipse,
                map(curry(keypoint_to_mapped_conic)(homography),  keypoints_to_torch(blobboard_info))
            )
        )
    )

    for patch_size in [32, 64]:
        model = HardNet(patch_size=patch_size)
        model.load_state_dict(torch.load(
            f"./data/models/2025_11_11/matrix_train/scale_96_res_{patch_size}_br_min_loss_triplet_margin_optimizer_sgd_real/_best_model_checkpoint.pth",
            weights_only=False,
            map_location=device
        )["state_dict"])
        model.to(device)
        # model = model.half().to(device)

        # Compile for max speed
        model = torch.compile(model, mode="max-autotune")
        model.eval()
        for num_patches in [10 ** i for i in range(4)]:
            measure(model, img, num_patches, keypoints, device)


if __name__ == "__main__":
    main()

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

import importlib.resources as resources
import os
import typing


import pdf2image
import torch
import torchvision


from blob_matcher.hardnet.losses import distance_matrix_vector
from blob_matcher.hardnet.models import HardNet
from blob_matcher.keypoints import ellipse_to_affine, get_patch, keypoints_to_torch
from blob_matcher.utils import read_json


class BlobMatcher:
    """The `BlobMatcher` uses log-polar descriptors to match the blobs on BlobBoards."""
    def __init__(
            self,
            board_dir: str,
            model_path: typing.Optional[str] = None,
            scale: int = 96,
            patch_size=32
    ):
        """
        Initialize the `BlobMatcher`.

        Arguments:
            board_dir: A path to a directory containing BlobBoards. Note that it must contain the pdf or png file and
                the json for each BlobBoard, nothing else.
            model_path: Optional path to the Model used for description.
            scale: The scale of the patches. Remember to use an appropriate model when changing the scale. The default
                model supports a scale of 96.
            patcht_size: Resolution of the patches the model expect. The default is 32x32.
        """
        device = torch.device("cpu")
        self.model = HardNet(patch_size=32)
        if model_path is not None:
            weights = torch.load(model_path, weights_only=False, map_location=device)
        else:
            with resources.files("blob_matcher").joinpath("data/models/default.pth").open("rb") as f:
                weights = torch.load(f, weights_only=False, map_location=device)
        self.model.load_state_dict(weights["state_dict"])
        self.model.eval()

        self.boards: list[tuple[typing.Optional[int], float, int]] = []
        blob_descriptors: list[torch.Tensor] = []
        board_files = os.listdir(board_dir)
        board_files.sort()
        for i in range(0, len(board_files), 2):
            assert board_files[i][:-4] == board_files[i+1][:-3]
            blob_meta = read_json(os.path.join(board_dir, board_files[i]))
            if board_files[i+1].endswith("png"):
                blob_image = torchvision.io.decode_image(
                    os.path.join(board_dir, board_files[i+1]),
                    torchvision.io.ImageReadMode.GRAY
                ).to(torch.float32) / 255
            elif board_files[i+1].endswith("pdf"):
                blob_image = torchvision.transforms.functional.pil_to_tensor(pdf2image.convert_from_path(
                    os.path.join(board_dir, board_files[i+1]),
                    dpi=blob_meta["preamble"]["board_config"]["print_density"]["value"],
                    grayscale=True
                )[0]).to(torch.float32) / 255
            keypoints = keypoints_to_torch(blob_meta)[:100] # TODO: Restore
            anchor_transforms = torch.stack(list(map(
                ellipse_to_affine,
                map(lambda k: (k[:2], (k[2], k[2]), 0), keypoints)
            )))
            anchor_patches = get_patch(
                blob_image,
                anchor_transforms,
                cfg=None,
                psf=scale
            )
            anchor_patches = torchvision.transforms.Resize((patch_size, patch_size))(anchor_patches)
            blob_descriptors.extend(map(lambda batch: self.model(batch)[0], anchor_patches.split(200)))
            self.boards.append((
                int(blob_meta["hashes"]["config"]) if "hashes" in blob_meta else None,
                float(blob_meta["preamble"]["pattern_config"]["seed"]),
                len(blob_meta["blobs"])
            ))
        self.blob_descriptors = torch.concat(blob_descriptors)

    def query(self, patches: torch.Tensor, k: int = 1) -> list[list[tuple[int, float, int]]]:
        """
        Query a sequence of patches for their nearest neighbours in the reference boards.

        Arguments:
            patches: A `torch.Tensor` with size either (B, 1, P, P) or (1, P, P) containing the image data of the
                patches normalized between [0, 1].
            k: number of nearest neighbours returned for each patch.

        Returns:
            A list of length B containing for every patch a list of length k with the nearest neigbours in ascending
            distance. A nearest neighbour is identified by a tuple consisting of
            - the BlobBoard config hash,
            - the BlobBoard seed and
            - the blob index within the board.
        """
        if patches.ndim == 3:
            patches.unsqueeze(0)
        patches.to(torch.float32)
        descriptors = self.model(patches)[0]
        distances = distance_matrix_vector(self.blob_descriptors, descriptors)
        indices = distances.argsort(dim=0)[:k]
        results = []
        for i in range(indices.size(1)):
            results.append(list(map(self._get_board_and_blob_id, indices[:, i])))
        return results

    def _get_board_and_blob_id(self, index):
        index = int(index)
        for board_hash, board_id, num_blobs in self.boards:
            if index < num_blobs:
                return board_hash, board_id, index
            index -= num_blobs
        return None


def main():
    path_to_boards = "./data/patterns"
    path_to_patches = "./data/datasets/2025_11_06/real/training/patches/96/positives"
    blob_matcher = BlobMatcher(board_dir=os.path.join(os.getcwd(), path_to_boards))
    patch_paths = list(filter(
        lambda f: f.startswith("0124"),
        os.listdir(path_to_patches)
    ))
    patch_paths.sort()
    print(patch_paths)
    patches = torch.empty((len(patch_paths), 1, 32, 32))
    for i, p in enumerate(patch_paths):
        patches[i] = torchvision.io.decode_image(
            os.path.join(path_to_patches, p),
            torchvision.io.ImageReadMode.GRAY
        ).to(torch.float32) / 255
    for [(board_hash, board_seed, blob_idx)] in blob_matcher.query(patches, k=1):
        if board_seed == 1.4467754675085926e19:
            print(f"Correct board {blob_idx} {board_hash}")
        else:
            print("Incorrect board {board_hash}")


if __name__ == "__main__":
    main()

import os
import sys


import pdf2image
import torch
import torchvision


sys.path.insert(0, os.getcwd())
from generate_dataset import ellipse_to_affine, get_patch, physical_to_logical_coordinates, physical_to_logical_distance, read_json
from modules.hardnet.losses import distance_matrix_vector
from modules.hardnet.models import HardNet


def keypoint_to_torch(resolution, border_width, canvas_offset):
    def _keypoint_to_torch(keypoint):
        return torch.tensor(
            [
                *physical_to_logical_coordinates(
                    (keypoint["center"][0]["value"], keypoint["center"][1]["value"]),
                    resolution,
                    border_width,
                    canvas_offset
                ),
                physical_to_logical_distance(keypoint["σ"]["value"], resolution),
                0
            ],
            dtype=torch.float32
        )
    return _keypoint_to_torch


def keypoints_to_torch(blob_info):
    resolution = blob_info["preamble"]["board_config"]["print_density"]["value"]
    border_width = blob_info["preamble"]["board_config"]["border_width"]["value"]
    canvas_offset = (
        (blob_info["preamble"]["board_config"]["canvas_size"]["width"]["value"]
            - blob_info["preamble"]["board_config"]["board_size"]["width"]["value"]) / 2,
        (blob_info["preamble"]["board_config"]["canvas_size"]["height"]["value"]
            - blob_info["preamble"]["board_config"]["board_size"]["height"]["value"]) / 2,
    )
    return torch.stack(list(map(keypoint_to_torch(resolution, border_width, canvas_offset), blob_info["blobs"])))


class BlobMatcher:
    def __init__(self, board_dir, model_path=os.path.join(os.path.dirname(__file__), "../data/models/default.pth"), scale=96):
        self.model = HardNet(transform="PTN", coords="log", patch_size=32, scale=scale)
        self.model.load_state_dict(torch.load(model_path, weights_only=False)["state_dict"])
        self.model.eval()

        self.boards = []
        blob_descriptors = []
        board_files = os.listdir(board_dir)
        board_files.sort()
        for i in range(0, len(board_files), 2):
            assert board_files[i][:-4] == board_files[i+1][:-3]
            blob_meta = read_json(os.path.join(board_dir, board_files[i]))
            if board_files[i+1].endswith("png"):
                blob_image = torchvision.io.decode_image(os.path.join(board_dir, board_files[i+1])).to(torch.float32) / 255
            elif board_files[i+1].endswith("pdf"):
                blob_image = torchvision.transforms.functional.pil_to_tensor(pdf2image.convert_from_path(
                    os.path.join(board_dir, board_files[i+1]),
                    dpi=blob_meta["preamble"]["board_config"]["print_density"]["value"],
                    grayscale=True
                )[0]).to(torch.float32) / 255
            keypoints = keypoints_to_torch(blob_meta)
            anchor_transforms = torch.stack(list(map(ellipse_to_affine, map(lambda k: (k[:2], (k[2], k[2]), 0), keypoints))))
            anchor_patches = get_patch(blob_image.unsqueeze(0).expand(anchor_transforms.size(0), -1, -1, -1), anchor_transforms, cfg=None, psf=scale)
            blob_descriptors.extend(map(lambda batch: self.model(batch)[0], anchor_patches.split(200)))
            self.boards.append((blob_meta["preamble"]["pattern_config"]["seed"], len(blob_meta["blobs"])))
        self.blob_descriptors = torch.concat(blob_descriptors)

    def query(self, patches, k=1):
        if patches.ndim == 3:
            patches.unsqueeze(0)
        descriptors = self.model(patches)[0]
        distances = distance_matrix_vector(self.blob_descriptors, descriptors)
        indices = distances.argsort(dim=0)[:k]
        results = []
        for i in range(indices.size(1)):
            results.append(list(map(self._get_board_and_blob_id, indices[:, i])))
        return results

    def _get_board_and_blob_id(self, index):
        index = int(index)
        for board_id, num_blobs in self.boards:
            if index < num_blobs:
                return board_id, index
            index -= num_blobs
        return None


if __name__ == "__main__":
    blob_matcher = BlobMatcher(board_dir=os.path.join(os.getcwd(), "data/real_image_data/2025_11_05/boards"))
    patch_paths = os.listdir("./data/datasets/new/real/validation/patches/96/positives")[:10]
    print(patch_paths)
    patches = torch.empty((10, 1, 32, 32))
    for i in range(10):
        patches[i] = torchvision.io.decode_image(
            os.path.join("./data/datasets/new/real/validation/patches/96/positives", patch_paths[i]),
            torchvision.io.ImageReadMode.GRAY
        ).to(torch.float32) / 255
    print(blob_matcher.query(patches, k=3))

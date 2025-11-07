import importlib.resources as resources
import os
import typing


import pdf2image
import torch
import torchvision


from blob_matcher.scripts.generate_dataset import ellipse_to_affine, get_patch, physical_to_logical_coordinates, physical_to_logical_distance, read_json
from blob_matcher.modules.hardnet.losses import distance_matrix_vector
from blob_matcher.modules.hardnet.models import HardNet


class BlobMatcher:
    """The `BlobMatcher` uses log-polar descriptors to match the blobs on BlobBoards."""
    def __init__(
            self,
            board_dir: str,
            model_path: typing.Optional[str] = None,
            scale: int = 96
    ):
        """
        Initialize the `BlobMatcher`.

        Arguments:
            board_dir: A path to a directory containing BlobBoards. Note that it must contain the pdf or png file and the json for each BlobBoard, nothing else.
            model_path: Optional path to the Model used for description.
            scale: The scale of the patches. Remember to use an appropriate model when changing the scale. The default model supports a scale of 96.
        """
        self.model = HardNet(transform="PTN", coords="log", patch_size=32, scale=scale)
        if model_path is not None:
            weights = torch.load(model_path, weights_only=False)
        else:
            with resources.files("blob_matcher").joinpath("data/models/default.pth").open("rb") as f:
                weights = torch.load(f, weights_only=False)
        self.model.load_state_dict(weights["state_dict"])
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
            keypoints = self._keypoints_to_torch(blob_meta)
            anchor_transforms = torch.stack(list(map(ellipse_to_affine, map(lambda k: (k[:2], (k[2], k[2]), 0), keypoints))))
            anchor_patches = get_patch(blob_image.unsqueeze(0).expand(anchor_transforms.size(0), -1, -1, -1), anchor_transforms, cfg=None, psf=scale)
            blob_descriptors.extend(map(lambda batch: self.model(batch)[0], anchor_patches.split(200)))
            self.boards.append((blob_meta["preamble"]["pattern_config"]["seed"], len(blob_meta["blobs"])))
        self.blob_descriptors = torch.concat(blob_descriptors)

    def query(self, patches: torch.Tensor, k: int = 1):
        """
        Query a sequence of patches for their nearest neighbours in the reference boards.

        Arguments:
            patches: A `pytorch.Tensor` with size either (B, 1, P, P) or (1, P, P) containing the image data of the patches normalized between [0, 1].
            k: number of nearest neighbours returned for each patch.

        Returns:
            A list of length B containing for every patch a list of length k with the nearest neigbours in ascending distance. A nearest neighbour is identified by a tuple consisting of the BlobBoard seed and the blob index within the board.
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
        for board_id, num_blobs in self.boards:
            if index < num_blobs:
                return board_id, index
            index -= num_blobs
        return None

    def _keypoint_to_torch(self, resolution, border_width, canvas_offset):
        def __keypoint_to_torch(keypoint):
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
        return __keypoint_to_torch

    def _keypoints_to_torch(self, blob_info):
        resolution = blob_info["preamble"]["board_config"]["print_density"]["value"]
        border_width = blob_info["preamble"]["board_config"]["border_width"]["value"]
        canvas_offset = (
            (blob_info["preamble"]["board_config"]["canvas_size"]["width"]["value"]
                - blob_info["preamble"]["board_config"]["board_size"]["width"]["value"]) / 2,
            (blob_info["preamble"]["board_config"]["canvas_size"]["height"]["value"]
                - blob_info["preamble"]["board_config"]["board_size"]["height"]["value"]) / 2,
        )
        return torch.stack(list(map(self._keypoint_to_torch(resolution, border_width, canvas_offset), blob_info["blobs"])))


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

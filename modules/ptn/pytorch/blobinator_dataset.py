"""
This module provides an extension to learn the features of the Blobinator calibration sheet.
"""

from abc import ABC
from functools import reduce
from itertools import chain, repeat
import json
import logging
import math
import os
import random
import sys
from typing import Any, Generator, NotRequired, Optional, TypeAlias, TypedDict


import cv2 as cv
import kagglehub
import kornia
import numpy as np
import pdf2image
import torch
import torchvision
from torchvision.transforms import v2


sys.path.insert(0, os.getcwd())

from modules.hardnet.utils import curry, fchain, flip


Ellipse: TypeAlias = tuple[np.typing.NDArray, tuple[float, float], float]

class BlobParam(TypedDict):
    value: float
    unit: str


class IsoBlob(TypedDict):
    center: list[BlobParam]
    σ: BlobParam


class BoardBundle(TypedDict):
    preamble: NotRequired[Any]
    blobs: list[IsoBlob]


class BlobinatorDataset:
    """
    This is an abstract base class for Blobinator datasets.

    It warps the Blobinator sheet and its keypoints into a number of background images via randomly generated
    homographies.
    """

    def __init__(self, cfg, temp_path) -> None:
        self.cfg = cfg
        self.temp_path = temp_path

        # res = blob_pattern(
        #     cfg.BLOBINATOR.PATTERN_HEIGHT,
        #     cfg.BLOBINATOR.PATTERN_WIDTH,
        #     min_scale=3,
        #     seed=cfg.BLOBINATOR.SEED,
        #     output_dir=cfg.BLOBINATOR.BOARD_DIR
        # )
        # files = res["files_created"]
        files = ["blob_pattern_1DA.json", "blob_pattern_1DA.png"]
        blob_metadata: BoardBundle = {"blobs": []}
        self.blobs = torch.zeros(
            size=(1, cfg.BLOBINATOR.PATTERN_WIDTH, cfg.BLOBINATOR.PATTERN_HEIGHT),
            dtype=torch.float32
        )
        for file in files:
            full_path = os.path.join(cfg.BLOBINATOR.BOARD_DIR, file)
            if file.endswith(".json"):
                with open(full_path) as f:
                    blob_metadata = json.loads(f.read())
            elif file.endswith(".png"):
                read_blobs = torchvision.io.decode_image(full_path, torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255
                if read_blobs is not None:
                    self.blobs[0] = read_blobs
                else:
                    logging.error("Could not read blob board")
            elif file.endswith(".pdf"):
                self.blobs[0] = torch.tensor(pdf2image.convert_from_path(full_path, grayscale=True)[0], dtype=torch.float32) / 255
            else:
                logging.warning(f"Unrecognised file '{file}'")

        keypoints = list(map(
            lambda k: torch.tensor([k["center"][0]["value"] - 1, k["center"][1]["value"] - 1, k["σ"]["value"], 0]),
            blob_metadata["blobs"]
        ))
        random.shuffle(keypoints)
        self.training_keypoints = torch.stack(keypoints[int(len(keypoints) * self.cfg.BLOBINATOR.VALIDATION.KEYPOINT_SPLIT):])
        self.validation_keypoints = torch.stack(keypoints[:int(len(keypoints) * self.cfg.BLOBINATOR.VALIDATION.KEYPOINT_SPLIT)])
        #path = os.path.join(kagglehub.dataset_download(self.cfg.BLOBINATOR.BACKGROUND_DATASET), "indoorCVPR_09/images")
        path = "./data/backgrounds/openloris-location"
        background_filenames = fchain(
            list,
            curry(filter)(curry(flip(str.endswith))(".png")),
            curry(reduce)(list.__add__),
            curry(map)(lambda x: list(map(lambda f: os.path.join(x[0], f), x[2])))
        )(os.walk(path))
        random.shuffle(background_filenames)
        self.training_background_filenames = background_filenames[int(len(background_filenames) * self.cfg.BLOBINATOR.VALIDATION.BACKGROUND_SPLIT):]
        self.training_background_filenames = self.training_background_filenames[:self.cfg.BLOBINATOR.TRAINING_DATASET_SIZE // self.cfg.BLOBINATOR.BLOBS_PER_IMAGE]
        self.validation_background_filenames = background_filenames[:int(len(background_filenames) * self.cfg.BLOBINATOR.VALIDATION.BACKGROUND_SPLIT)]

        self.background_index = 0

    def load_background(self, img_path):
        img = torchvision.io.decode_image(img_path, torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255
        if img.shape[1] > img.shape[2]:
            crop1 = (img.shape[1] - img.shape[2]) // 2
            crop2 = crop1 if 2 * crop1 + img.shape[2] == img.shape[1] else crop1 + 1
            img = img[:, crop1:-crop2, :]
        elif img.shape[2] > img.shape[1]:
            crop1 = (img.shape[2] - img.shape[1]) // 2
            crop2 = crop1 if 2 * crop1 + img.shape[1] == img.shape[2] else crop1 + 1
            img = img[:, :, crop1:-crop2]
        return torchvision.transforms.Resize((self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO))(img)

    def sample_homography(
        self,
        original_shape,
        patch_shape,
        perspective=True,
        scaling=True,
        rotation=True,
        translation=True,
        n_scales=5,
        n_angles=25,
        scaling_amplitude=32,
        perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1,
        patch_ratio=0.5,
        max_angle=np.pi / 2,
        allow_artifacts=False,
        translation_overflow=0.0,
    ) -> torch.Tensor:
        """Sample a random valid homography.

        **Note:** This function is an adapted version from [SuperPoints](github.com/rpautrat/SuperPoint) homography
        sampling.

        Computes the homography transformation between a random patch in the original image and a warped projection
        with the same image size. It maps the output point (warped patch) to a transformed input point (original
        patch). The original patch, which is initialized with a simple half-size centered crop, is iteratively
        projected, scaled, rotated and translated.

        Arguments:
            original_shape: A rank-2 `Tensor` specifying the height and width of the original image.
            patch_shape: A rank-2 `Tensor` specifying the height and width of the patch image.
            perspective: A boolean that enables the perspective and affine transformations.
            scaling: A boolean that enables the random scaling of the patch.
            rotation: A boolean that enables the random rotation of the patch.
            translation: A boolean that enables the random translation of the patch.
            n_scales: The number of tentative scales that are sampled when scaling.
            n_angles: The number of tentatives angles that are sampled when rotating.
            scaling_amplitude: Controls the amount of scale.
            perspective_amplitude_x: Controls the perspective effect in x direction.
            perspective_amplitude_y: Controls the perspective effect in y direction.
            patch_ratio: Controls the size of the patches used to create the homography.
            max_angle: Maximum angle used in rotations.
            allow_artifacts: A boolean that enables artifacts when applying the homography.
            translation_overflow: Amount of border artifacts caused by translation.

        Returns:
            An `array` of shape `(3,3)` corresponding to the homography.
        """
        def _truncated_normal(loc, scale, shape):
            result = np.random.normal(loc, scale, shape)
            while any(result < loc - 2 * scale) or any(result > loc + 2 * scale):
                result = np.random.normal(loc, scale, shape)
                logging.debug("Recalculated truncated normal")
            return result

        # Corners of the output image
        margin = (1 - patch_ratio) / 2
        pts1 = margin + np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]], np.float32)
        pts1.flags.writeable = False
        # Corners of the input patch
        pts2 = pts1.copy()

        # Random perspective and affine perturbations
        if perspective:
            if not allow_artifacts:
                perspective_amplitude_x = min(perspective_amplitude_x, margin)
                perspective_amplitude_y = min(perspective_amplitude_y, margin)
            perspective_displacement = np.random.normal(0.0, perspective_amplitude_y / 2, (1,))
            h_displacement_left = np.random.normal(0.0, perspective_amplitude_x / 2, (1,))
            h_displacement_right = np.random.normal(0.0, perspective_amplitude_x / 2, (1,))
            pts2 += np.stack([
                np.concatenate([h_displacement_left, perspective_displacement], axis=0),
                np.concatenate([h_displacement_left, -perspective_displacement], axis=0),
                np.concatenate([h_displacement_right, perspective_displacement], axis=0),
                np.concatenate([h_displacement_right, -perspective_displacement], axis=0),
            ])

        # Random scaling
        # sample several scales, check collision with borders, randomly pick a valid one
        if scaling:
            scales = np.concatenate([[1.0], np.random.normal(1, scaling_amplitude / 2, (n_scales,))], 0)
            center = np.mean(pts2, axis=0, keepdims=True)
            scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center
            if allow_artifacts:
                valid = np.arange(start=1, stop=n_scales + 1)  # all scales are valid except scale=1
            else:
                valid = np.where(np.all((scaled >= 0.0) & (scaled <= 1.0), axis=(1, 2)))[0]
            idx = valid[int(np.random.uniform(low=0, high=valid.shape[0]))]
            pts2 = scaled[idx]

        # Random translation
        if translation:
            t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
            if allow_artifacts:
                t_min += translation_overflow
                t_max += translation_overflow
            pts2 += np.expand_dims(
                np.stack([
                    np.random.uniform(low=-t_min[0], high=t_max[0]),
                    np.random.uniform(low=-t_min[1], high=t_max[1])
                ]),
                axis=0,
            )

        # Random rotation
        # sample several rotations, check collision with borders, randomly pick a valid one
        if rotation:
            angles = np.linspace(-max_angle, max_angle, n_angles)
            angles = np.concat([[0.0], angles], axis=0)  # in case no rotation is valid
            center = np.mean(pts2, axis=0, keepdims=True)
            rot_mat = np.reshape(
                np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], axis=1),
                (-1, 2, 2)
            )
            rotated = np.tile(np.expand_dims(pts2 - center, axis=0), (n_angles + 1, 1, 1)) @ rot_mat + center
            if allow_artifacts:
                valid = np.arange(1, n_angles + 1)  # all angles are valid, except angle=0
            else:
                valid = np.where(np.all((rotated >= 0.0) & (rotated <= 1.0), axis=(1, 2)))[0]
            idx = valid[int(np.random.uniform(low=0, high=valid.shape[0]))]
            pts2 = rotated[idx]

        # Rescale to actual size
        original_shape = tuple(map(float, original_shape[::-1]))  # different convention [y, x]
        pts1 = pts1 * np.expand_dims(original_shape, axis=0)
        pts2 = pts2 * np.expand_dims(original_shape, axis=0)

        def ax(p, q):
            return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

        def ay(p, q):
            return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

        a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
        p_mat = np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0).T
        flat_homography = np.linalg.lstsq(a_mat, p_mat)[0].T
        homography = np.ones(shape=(3, 3))
        homography[0, :] = flat_homography[0][:3]
        homography[1, :] = flat_homography[0][3:6]
        homography[2, :2] = flat_homography[0][6:]
        correct_translation = np.identity(3)
        correct_translation[0, 2] = (patch_shape[0] - original_shape[0]) // 2
        correct_translation[1, 2] = (patch_shape[1] - original_shape[1]) // 2
        return torch.tensor(correct_translation @ homography, dtype=torch.float32)

    def map_blobs(self, background: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        """
        Warps the blobsheet into the background according to a homography.

        The blobboard is taken from `self`.

        Argsument:
            background: A 2-D array containing the image data for the background.
            homography: A valid homography with shape (3,3) that describes the transformation from the blobboard into
            the image.

        Returns:
            An image with the blobsheet mapped in front of a background.
        """
        blobs = self.blobs.unsqueeze(0).expand(background.size(0), -1, -1, -1)
        warped_blobs = kornia.geometry.transform.warp_perspective(
            blobs,
            homography,
            (self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO),
        )
        #warped_blobs.reshape(background.size())
        mask = kornia.geometry.transform.warp_perspective(
            torch.ones_like(blobs),
            homography,
            (self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO)
        )
        mask = (mask > 0.999).float()
        warped_blobs = warped_blobs * mask + background * (1 - mask)
        return warped_blobs

    def keypoint_to_mapped_conic(self, homography: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Transforms a keypoint into a conic section and warps it via a homography.

        When ignoring orientation a keypoint can be repersented by a circle. If the circle is represented as a conic
        section, it can be transformed with a homography. We expect to obtain an elliptical conic section as a result,
        since the perspective change of the used homographies are not that large.

        Arguments:
            homography: The homography with which the keypoint will be mapped.
            k: The keypoint.

        Returns:
            An array of shape (3,3) representing the conic section of the mapped keypoint.
        """
        #location = k[:2]
        assert k.shape == (4,)
        scale = k[2]

        x = k[0]
        y = k[1]
        conic = torch.tensor([[1, 0, -x], [0, 1, -y], [-x, -y, x**2 + y**2 - scale**2]], dtype=torch.float32)
        inverse_homography = torch.linalg.inv(homography)
        mapped_conic = torch.transpose(inverse_homography, 0, 1) @ conic @ inverse_homography
        mapped_conic = (mapped_conic + torch.transpose(mapped_conic, 0, 1)) / 2
        return mapped_conic

    def conic_to_ellipse(self, conic) -> Optional[tuple[np.typing.NDArray, tuple[float, float], float]]:
        """
        Extracts ellipse parameters from a conic section.

        Arguments:
            conic: A (3,3) array with the conic section.

        Returns:
            A tuple consisting of
                - a 2 entry array with the location,
                - a tuple containing the semi-major axis and the semi-minor axis, and
                - the angle of the semi-major axis in radians.

        Raises:
            AssertionError: When the conic does not represent a valid ellipse.
        """
        location = -torch.linalg.inv(conic[:2, :2] * 2) @ (conic[:2, 2] * 2)
        A = conic[0, 0].item()
        B = conic[0, 1].item() * 2
        C = conic[1, 1].item()
        D = conic[0, 2].item() * 2
        E = conic[1, 2].item() * 2
        F = conic[2, 2].item()
        x_0 = location[0].item()
        y_0 = location[1].item()

        F_c = A * x_0 * x_0 + B * x_0 * y_0 + C * y_0 * y_0 + D * x_0 + E * y_0 + F
        if F_c >= 0:
            return None

        semi_axis_factor1 = 2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F)
        semi_minor_axis_factor2 = (A + C) - np.sqrt((A - C) ** 2 + B ** 2)
        semi_major_axis_factor2 = (A + C) + np.sqrt((A - C) ** 2 + B ** 2)
        semi_axis_quotient = B ** 2 - 4 * A * C
        semi_minor_axis = -np.sqrt(semi_axis_factor1 * semi_minor_axis_factor2) / semi_axis_quotient
        semi_major_axis = -np.sqrt(semi_axis_factor1 * semi_major_axis_factor2) / semi_axis_quotient
        angle = np.atan2(-B, C - A) / 2

        return location, (semi_major_axis, semi_minor_axis), angle

    def ellipse_to_affine(self, ellipse: Ellipse) -> torch.Tensor:
        """
        Finds an affine transformation that transforms the unit circle into the ellipse.

        Arguments:
            ellipse: The ellipse in which the unit circle should be transformed.

        Returns:
            An (3,3) array representing the affine transformation.
        """
        location, (semi_major_axis, semi_minor_axis), angle = ellipse
        scale = np.diag([semi_major_axis, semi_minor_axis, 1])
        rotation = np.identity(3)
        rotation[:2, :] = cv.getRotationMatrix2D((0, 0), int(-angle * 180 / np.pi), 1)
        translation = np.identity(3)
        translation[:2, 2] = location
        return torch.tensor(translation @ rotation @ scale, dtype=torch.float32)

    def get_patch(self, img: torch.Tensor, A: torch.Tensor, pad_with=1.0):
        """
        Extract a log-polar interpolated patch from an affine patch.

        The method used to extract the patch is now described in more detail. Given patch coordinates $(x,y)$ with
        $0 \\leq x, y < P$ for a given patch size $P$ we first normalize them into the half-open interval $[0,1)$:
        $\\overline{x} = \\frac{x}{P}, \\overline{x} = \\frac{x}{P}$. We set $x$ to be the radial dimension and $y$ to
        be the angular dimension of the patch. We can obtain the log-polar coordinates by:
        $$r = e^{\\ln(PSF) \\cdot \\overline{x}}$$
        and
        $$\\theta = 2 \\cdot \\pi \\cdot \\overline{x}$$
        where $PSF$ is the *Patch-Scale-Factor*, a configuration value that controls the size of the patch in relation
        to the radius of the blob. Note that $r \\in [1, PSF)$ and $\\theta \\in [0, 2\\pi)$.

        Arguments:
            img: The source image.
            A: An affine transform that maps the unit circle into the keypoints.
            pad_with: A color value to use for padding if part of the patch lie outside the source image.

        Returns:
            A 2-dim `numpy` Array containing the image data of the patch.
            """
        device = img.device
        P = self.cfg.INPUT.IMAGE_SIZE
        PSF = self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR
        B, _, Hs, Ws = img.shape

        # Create normalized grid for output patch
        y, x = torch.meshgrid(
            torch.linspace(0, 1, P, device=device, dtype=torch.float32),
            torch.linspace(0, 1, P, device=device, dtype=torch.float32),
            indexing='ij'
        )
        # Each (x,y) is a location in patch coords, shape (P, P)
        # We treat x as radial and y as angular dimension

        # Compute log-polar coordinates
        r = torch.exp(torch.log(torch.tensor(PSF, device=device)) * x)  # [1, PSF)
        theta = 2 * math.pi * y                                         # [0, 2π)

        # Convert to Cartesian coordinates in source patch
        xs = r * torch.cos(theta)
        ys = r * torch.sin(theta)

        # Stack homogeneous coords
        ones = torch.ones_like(xs)
        coords = torch.stack([xs, ys, ones], dim=-1)
        coords = coords.view(1, P, P, 3)
        coords = coords.expand(B, -1, -1, -1)  # (B, P, P, 3)

        # Flatten spatial grid for batch multiplication
        coords_flat = coords.view(B, -1, 3)                     # (B, P*P, 3)
        # Apply affine transform A to each batch item
        coords_flat = torch.bmm(coords_flat, A.transpose(1, 2)) # (B, P*P, 3)
        # Reshape back
        coords = coords_flat.view(B, P, P, 3)

        x_src, y_src = coords[..., 0], coords[..., 1]

        # Normalize coordinates to [-1, 1] for grid_sample
        x_norm = 2 * (x_src / (Ws - 1)) - 1
        y_norm = 2 * (y_src / (Hs - 1)) - 1
        grid = torch.stack([x_norm, y_norm], dim=-1)  # (B, P, P, 2)

        # Sample the image
        warped = torch.nn.functional.grid_sample(
            img, grid,
            mode='bilinear',
            padding_mode='zeros',  # outside = 0
            align_corners=True
        )

        # Blend with constant background
        mask = torch.nn.functional.grid_sample(
            torch.ones((B, 1, Hs, Ws), device=device),
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        )

        background = torch.full_like(warped, pad_with)
        output = warped * mask + background * (1 - mask)
        return output


class BlobinatorTrainDataset(BlobinatorDataset):
    def convert_cv_keypoint(self, keypoint: cv.KeyPoint) -> torch.Tensor:
        return torch.tensor([*keypoint.pt, keypoint.size, keypoint.angle * np.pi / 180])
    
    def preprocess(self):
        os.makedirs(os.path.join(self.temp_path, "warped_image"), exist_ok=True)
        resizing = torchvision.transforms.Resize((self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO))
        transforms = v2.Compose([
            v2.ColorJitter(),
            v2.GaussianBlur(kernel_size=(5, 5)),
            v2.GaussianNoise()
        ])
        self.homographies = [
            self.sample_homography(original_shape=self.blobs.shape[1:], patch_shape=(self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO))
            for _ in range(len(self.training_background_filenames))
        ]
        for i, (img_path, homography) in enumerate(zip(self.training_background_filenames, self.homographies)):
            print(img_path)
            img = torchvision.io.decode_image(img_path, torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255
            if img.shape[1] > img.shape[2]:
                crop1 = (img.shape[1] - img.shape[2]) // 2
                crop2 = crop1 if 2 * crop1 + img.shape[2] == img.shape[1] else crop1 + 1
                img = img[:, crop1:-crop2, :]
            elif img.shape[2] > img.shape[1]:
                crop1 = (img.shape[2] - img.shape[1]) // 2
                crop2 = crop1 if 2 * crop1 + img.shape[1] == img.shape[2] else crop1 + 1
                img = img[:, :, crop1:-crop2]
            img = resizing(img)
            img = self.map_blobs(img.unsqueeze(0), homography.unsqueeze(0))
            img = transforms(img)
            torchvision.utils.save_image(img, os.path.join(self.temp_path, "warped_image", f"{i:04}.png"))

        anchor_patch_transforms = torch.stack(list(map(lambda keypoint: self.ellipse_to_affine((keypoint[:2], (keypoint[2], keypoint[2]), 0)), self.training_keypoints)))
        self.anchor_patches = self.get_patch(self.blobs.unsqueeze(0).expand(anchor_patch_transforms.size(0), -1, -1, -1), anchor_patch_transforms)

        self.keypoint_map = {}
        for i, homography in enumerate(self.homographies):
            warped_keypoints = []
            for j in range(self.training_keypoints.size(0)):
                warped_keypoints.append(self.conic_to_ellipse(self.keypoint_to_mapped_conic(homography, self.training_keypoints[j])))
            indices = torch.where(torch.tensor(list(map(lambda x: x is not None, warped_keypoints))))[0]
            transforms = torch.stack(list(map(self.ellipse_to_affine, filter(lambda x: x is not None, warped_keypoints))))
            assert indices.size(0) == transforms.size(0)
            self.keypoint_map[i] = (indices, transforms)
        self.length = sum(map(lambda x: len(x[0]), self.keypoint_map.values()))

    def get_image_and_patch_index(self, idx):
        image_idx = 0
        while idx >= len(self.keypoint_map[image_idx][0]):
            idx -= len(self.keypoint_map[image_idx][0])
            image_idx += 1
        patch_idx = self.keypoint_map[image_idx][0][idx]
        return image_idx, patch_idx

    def __getitem__(self, idx):
        image_idx, patch_idx = self.get_image_and_patch_index(idx)
        base_image = torchvision.io.decode_image(
            os.path.join(self.temp_path, "warped_image", f"{image_idx:04}.png"),
            torchvision.io.ImageReadMode.GRAY
        )
        positive_transform = self.keypoint_map[image_idx][1][int(torch.where(self.keypoint_map[image_idx][0] == patch_idx)[0])]
        positive_patch = self.get_patch(base_image.to(torch.float).unsqueeze(0) / 255, positive_transform.unsqueeze(0))
        return (self.anchor_patches[patch_idx], positive_patch.squeeze(0), torch.zeros((1, 32, 32)), False)

    def __len__(self):
        return self.length


class BlobinatorBlobToBlobValidationDataset(torch.utils.data.IterableDataset, BlobinatorDataset):
    def __iter__(self):
        dataset_size = 10_000
        self.num_background_images = dataset_size // (2 * self.validation_keypoint.size(0))
        random.shuffle(self.validation_background_filenames)
        for _, background_filename in zip(range(self.num_background_images), self.validation_background_filenames):
            background = self.load_background(background_filename).unsqueeze(0)
            homography = self.sample_homography.unsqueeze(0)
            warped_image = self.map_blobs(background, self.homographies)
            transforms = v2.Compose([
                v2.ColorJitter(),
                v2.GaussianBlur(kernel_size=(3, 3)),
                v2.GaussianNoise()
            ])
            warped_image = transforms(warped_image)
            for (homography, warped_image) in zip(self.homographies, warped_images):
                anchor_patch_transforms = torch.stack(list(map(lambda keypoint: self.ellipse_to_affine((keypoint[:2], (keypoint[2], keypoint[2]), 0)), self.validation_keypoints)))
                anchor_patches = self.get_patch(self.blobs.unsqueeze(0).expand(anchor_patch_transforms.size(0), -1, -1, -1), anchor_patch_transforms)
                keypoint_indices = torch.randperm(len(self.training_keypoints))
                for keypoint, anchor_idx in zip(self.training_keypoints[keypoint_indices], keypoint_indices):
                    try:
                        positive_patch_transform = self.ellipse_to_affine(self.conic_to_ellipse(self.keypoint_to_mapped_conic(
                            homography,
                            keypoint
                        )))
                    except AssertionError:
                        continue    # Encountered invalid ellipse
                    positive_patch = self.get_patch(warped_image.unsqueeze(0), positive_patch_transform.unsqueeze(0))
                    yield (
                        anchor_patches[anchor_idx],
                        positive_patch,
                        1.
                    )
                    random_idx = int(np.random.uniform(low=0, high=anchor_patches.size(0)))
                    while random_idx == anchor_idx:
                        random_idx = int(np.random.uniform(low=0, high=anchor_patches.size(0)))
                    yield (
                        anchor_patches[random_idx],
                        positive_patch,
                        0.
                    )

    def __len__(self):
        return 1
    


def main():
    import sys
    import time
    sys.path.insert(0, os.getcwd())
    from configs.defaults import _C as cfg

    directory = "./data/patches"
    start = time.time()

    dataset = BlobinatorTrainDataset(cfg, "data/training")
    dataset.preprocess()

    os.makedirs(os.path.join(directory, "anchor"), exist_ok=True)
    os.makedirs(os.path.join(directory, "positive"), exist_ok=True)
    os.makedirs(os.path.join(directory, "garbage"), exist_ok=True)

    for i in range(len(dataset)):
        (anchor_patch, positive_patch, garbage_patch, garbage_available) = dataset[i]
        torchvision.utils.save_image(anchor_patch, os.path.join(directory, "anchor", f"{i:04}.png"))
        torchvision.utils.save_image(positive_patch, os.path.join(directory, "positive", f"{i:04}.png"))
        if garbage_available:
            torchvision.utils.save_image(garbage_patch, os.path.join(directory, "garbage", f"{i:04}.png"))

    end = time.time()
    print(f"Generated {i*3} patches in {end - start:.1f} s")  # pyright: ignore[reportPossiblyUnboundVariable]


if __name__ == "__main__":
    main()

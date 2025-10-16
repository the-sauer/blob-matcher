"""
This module provides an extension to learn the features of the Blobinator calibration sheet.
"""

from abc import ABC
from itertools import chain, repeat
import json
import logging
import os
from typing import Any, Generator, NotRequired, Sequence, TypeAlias, TypedDict


import cv2 as cv
import kornia
import numpy as np
import pdf2image
import skimage
import torch
from torchvision.transforms import v2


Ellipse: TypeAlias = tuple[np.typing.NDArray, tuple[float, float], float]


Keypoint: TypeAlias = tuple[np.typing.NDArray, float, float]


class BlobParam(TypedDict):
    value: float
    unit: str


class IsoBlob(TypedDict):
    center: list[BlobParam]
    σ: BlobParam


class BoardBundle(TypedDict):
    preamble: NotRequired[Any]
    blobs: list[IsoBlob]


class BlobinatorDataset(ABC):
    """
    This is an abstract base class for Blobinator datasets.

    It warps the Blobinator sheet and its keypoints into a number of background images via randomly generated
    homographies.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

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
                read_blobs = torch.tensor(cv.imread(full_path, cv.IMREAD_GRAYSCALE) / 255, dtype=torch.float32)
                if read_blobs is not None:
                    self.blobs[0] = read_blobs
                else:
                    logging.error("Could not read blob board")
            elif file.endswith(".pdf"):
                self.blobs[0] = torch.tensor(pdf2image.convert_from_path(full_path, grayscale=True)[0] / 255, dtype=torch.float32)
            else:
                logging.warning(f"Unrecognised file '{file}'")

        self.keypoints: Sequence[Keypoint] = list(map(
            lambda k: (np.array([k["center"][0]["value"] - 1, k["center"][1]["value"] - 1]), k["σ"]["value"], 0),
            blob_metadata["blobs"]
        ))

        self.backgrounds = torch.zeros(
            size=(
                len(list(filter(lambda x: x.endswith(".jpg"), os.listdir((cfg.BLOBINATOR.BACKGROUND_DIR))))),
                1,
                self.cfg.TRAINING.PAD_TO,
                self.cfg.TRAINING.PAD_TO
            ),
            dtype=torch.float32
        )
        for i, img_path in enumerate(filter(lambda x: x.endswith(".jpg"), os.listdir(cfg.BLOBINATOR.BACKGROUND_DIR))):
            img = cv.imread(os.path.join(cfg.BLOBINATOR.BACKGROUND_DIR, img_path), cv.IMREAD_GRAYSCALE)
            if img is None:
                logging.error(f"Failed to load image '{img_path}'")
                continue
            if img.shape[0] > img.shape[1]:
                crop1 = (img.shape[0] - img.shape[1]) // 2
                crop2 = crop1 if 2 * crop1 + img.shape[1] == img.shape[0] else crop1 + 1
                img = img[crop1:-crop2, :]
            elif img.shape[1] > img.shape[0]:
                crop1 = (img.shape[1] - img.shape[0]) // 2
                crop2 = crop1 if 2 * crop1 + img.shape[0] == img.shape[1] else crop1 + 1
                img = img[:, crop1:-crop2]
            self.backgrounds[i, 0, :, :] = torch.as_tensor(
                cv.resize(img, dsize=(cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO)) / 255,
                dtype=torch.float32
            )

        self.homographies = torch.stack([
            self.sample_homography(
                original_shape=self.blobs.shape[1:],
                patch_shape=(cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO)
            )
            for _ in range(len(self.backgrounds))
        ])

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
        scaling_amplitude=0.1,
        perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1,
        patch_ratio=0.5,
        max_angle=np.pi / 2,
        allow_artifacts=False,
        translation_overflow=0.0,
    ) -> np.typing.NDArray:
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
            perspective_displacement = _truncated_normal(0.0, perspective_amplitude_y / 2, (1,))
            h_displacement_left = _truncated_normal(0.0, perspective_amplitude_x / 2, (1,))
            h_displacement_right = _truncated_normal(0.0, perspective_amplitude_x / 2, (1,))
            pts2 += np.stack([
                np.concatenate([h_displacement_left, perspective_displacement], axis=0),
                np.concatenate([h_displacement_left, -perspective_displacement], axis=0),
                np.concatenate([h_displacement_right, perspective_displacement], axis=0),
                np.concatenate([h_displacement_right, -perspective_displacement], axis=0),
            ])

        # Random scaling
        # sample several scales, check collision with borders, randomly pick a valid one
        if scaling:
            scales = np.concatenate([[1.0], _truncated_normal(1, scaling_amplitude / 2, (n_scales,))], 0)
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

    def keypoint_to_mapped_conic(self, homography: torch.Tensor, k: Keypoint) -> torch.Tensor:
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
        location, scale, _ = k
        x = location[0]
        y = location[1]
        conic = torch.tensor([[1, 0, -x], [0, 1, -y], [-x, -y, x**2 + y**2 - scale**2]], dtype=torch.float32)
        inverse_homography = torch.linalg.inv(homography)
        mapped_conic = torch.transpose(inverse_homography, 0, 1) @ conic @ inverse_homography
        mapped_conic = (mapped_conic + torch.transpose(mapped_conic, 0, 1)) / 2
        return mapped_conic

    def conic_to_ellipse(self, conic) -> tuple[np.typing.NDArray, tuple[float, float], float]:
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
        assert F_c < 0

        semi_axis_factor1 = 2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F)
        semi_minor_axis_factor2 = (A + C) - np.sqrt((A - C) ** 2 + B ** 2)
        semi_major_axis_factor2 = (A + C) + np.sqrt((A - C) ** 2 + B ** 2)
        semi_axis_quotient = B ** 2 - 4 * A * C
        semi_minor_axis = -np.sqrt(semi_axis_factor1 * semi_minor_axis_factor2) / semi_axis_quotient
        semi_major_axis = -np.sqrt(semi_axis_factor1 * semi_major_axis_factor2) / semi_axis_quotient
        angle = np.atan2(-B, C - A) / 2

        return location, (semi_major_axis, semi_minor_axis), angle

    def ellipse_to_affine(self, ellipse: Ellipse) -> np.typing.NDArray:
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
        return torch.tensor(translation @ rotation @ scale)

    def get_patch(self, img, A, pad_with=1.0):
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
        def backwards_map(cr):
            coords = torch.empty(cr.shape)
            for i in range(cr.shape[0]):
                normalized_x = cr[i, 0] / self.cfg.INPUT.IMAGE_SIZE     # x in [0,1)
                normalized_y = cr[i, 1] / self.cfg.INPUT.IMAGE_SIZE     # y in [0,1)
                r = np.exp(np.log(self.cfg.BLOBINATOR.PATCH_SCALE_FACTOR) * normalized_x)   # r in [1, PATCH_SCALE_FACTOR)
                x_source = r * np.cos(2 * np.pi * normalized_y)
                y_source = r * np.sin(2 * np.pi * normalized_y)

                coords[i] = (A @ torch.tensor([x_source, y_source, 1]))[:2]
            return coords
        return torch.from_numpy(skimage.transform.warp(
            img[0],
            backwards_map,
            output_shape=(self.cfg.INPUT.IMAGE_SIZE, self.cfg.INPUT.IMAGE_SIZE),
            mode="constant",
            cval=pad_with
        )).unsqueeze(0)


class BlobinatorTrainDataset(torch.utils.data.IterableDataset, BlobinatorDataset):
    def convert_cv_keypoint(self, keypoint: cv.KeyPoint) -> Keypoint:
        return np.array(keypoint.pt), keypoint.size, keypoint.angle * np.pi / 180

    def __iter__(self) -> Generator[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            bool
        ],
        None,
        None
    ]:
        """
        Yields:
            A tuple consisting of
                - the anchor patch,
                - a positive patch (i.e.) the corresponding patch to the anchor patch in the warped image,
                - an optional garbage patch, which describes a random detection in the background, and
                - a boolean flag whether there is a garbage patch available.
        """
        sift = cv.SIFT_create()
        warped_images = self.map_blobs(self.backgrounds, self.homographies)
        transforms = v2.Compose([
            v2.ColorJitter(),
            v2.GaussianBlur(kernel_size=(3, 3)),
            v2.GaussianNoise()
        ])
        warped_images = transforms(warped_images)
        for homography, background, warped_image in zip(self.homographies, self.backgrounds, warped_images):
            # warped_image = self.map_blobs(background, homography)
            # cv.imwrite("warped_image.png", np.clip(warped_image[0].numpy() * 255, min=0, max=255).astype(np.uint8))
            garbage_mask = np.zeros(shape=background[0].shape, dtype=np.uint8)
            garbage_mask[100:-100, 100:-100] = np.ones(
                shape=(background.shape[1] - 200, background.shape[2] - 200),
                dtype=np.uint8
            )
            detections = sift.detect((background[0].numpy() * 255).astype(np.uint8), garbage_mask)
            garbage_keypoints = map(self.convert_cv_keypoint, detections)
            for keypoint, garbage_keypoint in zip(self.keypoints, chain(garbage_keypoints, repeat(None))):
                # TODO: Augmentation
                anchor_patch_transform = self.ellipse_to_affine((keypoint[0], (keypoint[1], keypoint[1]), 0))

                anchor_patch = self.get_patch(self.blobs, anchor_patch_transform)
                # TODO: Augmentation
                positive_patch_transform = self.ellipse_to_affine(self.conic_to_ellipse(self.keypoint_to_mapped_conic(
                    homography,
                    keypoint
                )))
                positive_patch = self.get_patch(warped_image, positive_patch_transform)

                garbage_available = False
                garbage_patch = np.zeros(shape=(self.cfg.INPUT.IMAGE_SIZE, self.cfg.INPUT.IMAGE_SIZE))
                if garbage_keypoint is not None:
                    garbage_patch_transform = self.ellipse_to_affine(self.conic_to_ellipse(
                        self.keypoint_to_mapped_conic(homography, garbage_keypoint)
                    ))
                    garbage_patch = self.get_patch(background, garbage_patch_transform)
                    garbage_available = True

                yield (
                    anchor_patch.reshape(1, self.cfg.INPUT.IMAGE_SIZE, self.cfg.INPUT.IMAGE_SIZE),
                    positive_patch.reshape(1, self.cfg.INPUT.IMAGE_SIZE, self.cfg.INPUT.IMAGE_SIZE),
                    garbage_patch.reshape(1, self.cfg.INPUT.IMAGE_SIZE, self.cfg.INPUT.IMAGE_SIZE),
                    garbage_available
                )


class BlobinatorTestDataset(torch.utils.data.Dataset, BlobinatorDataset):
    def __getitem__(self, index) -> tuple[np.typing.NDArray, np.typing.NDArray, float]:
        # TODO: Implement
        return (
            np.zeros(shape=(self.cfg.INPUT.IMAGE_SIZE, self.cfg.INPUT.IMAGE_SIZE)),
            np.zeros(shape=(self.cfg.INPUT.IMAGE_SIZE, self.cfg.INPUT.IMAGE_SIZE)),
            1.
        )

    def __len__(self):
        return 1


def main():
    import sys
    sys.path.insert(0, os.getcwd())
    from configs.defaults import _C as cfg

    directory = "./data/patches"

    dataset = BlobinatorTrainDataset(cfg)

    for i, (anchor_patch, positive_patch, garbage_patch, garbage_available) in zip(range(3), dataset):
        os.makedirs(os.path.join(directory, f"{i:02}"), exist_ok=True)
        cv.imwrite(os.path.join(directory, f"{i:02}", "anchor.png"), (anchor_patch[0] * 255).numpy().astype(np.uint8))
        cv.imwrite(os.path.join(directory, f"{i:02}", "positive.png"), (positive_patch[0] * 255).numpy().astype(np.uint8))
        if garbage_available:
            cv.imwrite(os.path.join(directory, f"{i:02}", "garbage.png"), (garbage_patch[0] * 255).numpy().astype(np.uint8))


if __name__ == "__main__":
    main()

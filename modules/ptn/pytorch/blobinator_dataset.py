"""
This module provides an extension to learn the features of the Blobinator calibration sheet.
"""

#from BlobBoards import blob_pattern

from abc import ABC
import json
import logging
import os
from typing import Sequence, TypeAlias

import cv2 as cv
import numpy as np
import pdf2image
import tensorflow as tf
import torch



Keypoint: TypeAlias = tuple[np.ndarray, float, float]


class BlobinatorDataset(torch.utils.data.Dataset, ABC):
    """
    This is the Blobinator dataset.

    It warps the Blobinator sheet into a number of background images via randomly generated homographies.
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
        blob_metadata = None
        self.blobs = None
        for file in files:
            full_path = os.path.join(cfg.BLOBINATOR.BOARD_DIR, file)
            if file.endswith(".json"):
                assert(blob_metadata is None)
                with open(full_path) as f:
                    blob_metadata = json.loads(f.read())
            elif file.endswith(".png"):
                assert(self.blobs is None)
                self.blobs = np.zeros(shape=(1, cfg.BLOBINATOR.PATTERN_WIDTH, cfg.BLOBINATOR.PATTERN_HEIGHT))
                self.blobs[0, :, :] = cv.imread(full_path, cv.IMREAD_GRAYSCALE)
            elif file.endswith(".pdf"):
                assert(self.blobs is None)
                self.blobs = np.zeros(shape=(1, cfg.BLOBINATOR.PATTERN_WIDTH, cfg.BLOBINATOR.PATTERN_HEIGHT))
                self.blobs[0, :, :] = np.array(pdf2image.convert_from_path(full_path, grayscale=True)[0])
            else:
                logging.warning(f"Unrecognised file '{file}'")

        self.keypoints: Sequence[Keypoint] = list(map(
            lambda k: (np.array([k["center"][0]["value"] - 1, k["center"][1]["value"] - 1]), k["σ"]["value"], 0),
            blob_metadata["blobs"]
        ))
        self.backgrounds: np.ndarray = np.zeros(
            shape=(len(os.listdir((cfg.BLOBINATOR.BACKGROUND_DIR))), self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO)
        )
        for i, img_path in enumerate(os.listdir(cfg.BLOBINATOR.BACKGROUND_DIR)):
            img = cv.imread(os.path.join(cfg.BLOBINATOR.BACKGROUND_DIR, img_path), cv.IMREAD_GRAYSCALE)
            if img.shape[0] > img.shape[1]:
                crop1 = (img.shape[0] - img.shape[1]) // 2
                crop2 = crop1 if 2 * crop1 + img.shape[1] == img.shape[0] else crop1 + 1
                img = img[crop1:-crop2, :]
            elif img.shape[1] > img.shape[0]:
                crop1 = (img.shape[1] - img.shape[0]) // 2
                crop2 = crop1 if 2 * crop1 + img.shape[0] == img.shape[1] else crop1 + 1
                img = img[:, crop1:-crop2]
            try:
                self.backgrounds[i, :, :] = cv.resize(img, dsize=(cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO))
            except:
                logging.error(f"Failed to resize image {img_path}")

        self.homographies = [
            self.sample_homography(
                original_shape=(cfg.BLOBINATOR.PATTERN_WIDTH, cfg.BLOBINATOR.PATTERN_HEIGHT),
                patch_shape=(cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO)
            )
            for _ in range(len(self.backgrounds))
        ]

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
    ):
        """Sample a random valid homography.

        **Note:** This function is an adapted version from [SuperPoints](github.com/rpautrat/SuperPoint) homography sampling.

        Computes the homography transformation between a random patch in the original image
        and a warped projection with the same image size.
        As in `tf.contrib.image.transformtransform`, it maps the output point (warped patch) to a
        transformed input point (original patch).
        The original patch, which is initialized with a simple half-size centered crop, is
        iteratively projected, scaled, rotated and translated.

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

        # Corners of the output image
        margin = (1 - patch_ratio) / 2
        pts1 = margin + tf.constant(
            [[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]],
            tf.float32,
        )
        # Corners of the input patch
        pts2 = pts1

        # Random perspective and affine perturbations
        if perspective:
            if not allow_artifacts:
                perspective_amplitude_x = min(perspective_amplitude_x, margin)
                perspective_amplitude_y = min(perspective_amplitude_y, margin)
            perspective_displacement = tf.random.truncated_normal(
                [1], 0.0, perspective_amplitude_y / 2
            )
            h_displacement_left = tf.random.truncated_normal(
                [1], 0.0, perspective_amplitude_x / 2
            )
            h_displacement_right = tf.random.truncated_normal(
                [1], 0.0, perspective_amplitude_x / 2
            )
            pts2 += tf.stack(
                [
                    tf.concat([h_displacement_left, perspective_displacement], 0),
                    tf.concat([h_displacement_left, -perspective_displacement], 0),
                    tf.concat([h_displacement_right, perspective_displacement], 0),
                    tf.concat([h_displacement_right, -perspective_displacement], 0),
                ]
            )

        # Random scaling
        # sample several scales, check collision with borders, randomly pick a valid one
        if scaling:
            scales = tf.concat(
                [[1.0], tf.random.truncated_normal([n_scales], 1, scaling_amplitude / 2)], 0
            )
            center = tf.reduce_mean(pts2, axis=0, keepdims=True)
            scaled = (
                tf.expand_dims(pts2 - center, axis=0)
                * tf.expand_dims(tf.expand_dims(scales, 1), 1)
                + center
            )
            if allow_artifacts:
                valid = tf.range(1, n_scales + 1)  # all scales are valid except scale=1
            else:
                valid = tf.where(
                    tf.reduce_all((scaled >= 0.0) & (scaled <= 1.0), [1, 2])
                )[:, 0]
            idx = valid[
                tf.random.uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)
            ]
            pts2 = scaled[idx]

        # Random translation
        if translation:
            t_min, t_max = tf.reduce_min(pts2, axis=0), tf.reduce_min(1 - pts2, axis=0)
            if allow_artifacts:
                t_min += translation_overflow
                t_max += translation_overflow
            pts2 += tf.expand_dims(
                tf.stack(
                    [
                        tf.random.uniform((), -t_min[0], t_max[0]),
                        tf.random.uniform((), -t_min[1], t_max[1]),
                    ]
                ),
                axis=0,
            )

        # Random rotation
        # sample several rotations, check collision with borders, randomly pick a valid one
        if rotation:
            angles = tf.linspace(
                tf.constant(-max_angle), tf.constant(max_angle), n_angles
            )
            angles = tf.concat([[0.0], angles], axis=0)  # in case no rotation is valid
            center = tf.reduce_mean(pts2, axis=0, keepdims=True)
            rot_mat = tf.reshape(
                tf.stack(
                    [tf.cos(angles), -tf.sin(angles), tf.sin(angles), tf.cos(angles)],
                    axis=1,
                ),
                [-1, 2, 2],
            )
            rotated = (
                tf.matmul(
                    tf.tile(
                        tf.expand_dims(pts2 - center, axis=0), [n_angles + 1, 1, 1]
                    ),
                    rot_mat,
                )
                + center
            )
            if allow_artifacts:
                valid = tf.range(
                    1, n_angles + 1
                )  # all angles are valid, except angle=0
            else:
                valid = tf.where(
                    tf.reduce_all((rotated >= 0.0) & (rotated <= 1.0), axis=[1, 2])
                )[:, 0]
            idx = valid[
                tf.random.uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)
            ]
            pts2 = rotated[idx]

        # Rescale to actual size
        original_shape = tf.cast(original_shape[::-1], tf.float32)  # different convention [y, x]
        pts1 *= tf.expand_dims(original_shape, axis=0)
        pts2 *= tf.expand_dims(original_shape, axis=0)

        def ax(p, q):
            return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

        def ay(p, q):
            return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

        a_mat = tf.stack(
            [f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0
        )
        p_mat = tf.transpose(
            tf.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0)
        )
        flat_homography = tf.transpose(tf.linalg.lstsq(a_mat, p_mat, fast=True)).numpy()
        homography = np.ones(shape=(3, 3))
        homography[0, :] = flat_homography[0][:3]
        homography[1, :] = flat_homography[0][3:6]
        homography[2, :2] = flat_homography[0][6:]
        correct_translation = np.identity(3)
        correct_translation[0,2] = (patch_shape[0] - original_shape[0]) // 2
        correct_translation[1,2] = (patch_shape[1] - original_shape[1]) // 2
        return correct_translation @ homography

    def normalize_keypoint(self, keypoint, into):
        loc, scale, rotation = keypoint
        normalized_x = 2 * loc[0] / into[0] - 1
        normalized_y = 2 * loc[1] / into[1] - 1
        return np.array([normalized_x, normalized_y]), scale, rotation

    def map_blobs(self, background, homography):
        """
        Warps the blobsheet into the background according to a homography.
        """
        warped_blobs = cv.warpPerspective(
            self.blobs[0, :, :],
            homography,
            (self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO),
            background,
            borderMode=cv.BORDER_TRANSPARENT
        )
        warped_blobs.reshape(background.shape)
        return warped_blobs
    
    def keypoint_to_mapped_conic(self, homography: np.ndarray, k: Keypoint) -> np.ndarray:
        location, scale, rotation = k
        x = location[0]
        y = location[1]
        conic = np.array(
            [[1, 0, -x],
            [0, 1, -y],
            [-x, -y, x**2 + y**2 - scale**2]],
            dtype=float
        )
        mapped_conic = np.linalg.inv(homography).transpose() @ conic @ np.linalg.inv(homography)
        mapped_conic = (mapped_conic + mapped_conic.transpose()) / 2
        return mapped_conic
    
    def conic_to_ellipse(self, conic: np.ndarray) -> tuple[np.ndarray, tuple[float, float], float]:
        location = -np.linalg.inv(conic[:2,:2] * 2) @ (conic[:2,2] * 2)
        Fc = conic[0,0] * location[0] ** 2 \
            + 2 * conic[0,1] * location[0] * location[0] \
            + conic[1,1] * location[1] ** 2 \
            + 2 * conic[0,2] * location[0] \
            + 2 * conic[1,2] * location[1] \
            + conic[2,2]
        eigenvalues, eigenvectors = np.linalg.eigh(conic[:2,:2])
        a = np.sqrt(-Fc / eigenvalues[1])
        b = np.sqrt(-Fc / eigenvalues[0])
        angle = np.arccos(
            np.dot(eigenvectors[1], np.array([1, 0]))
            / np.linalg.norm(eigenvectors[1])
        )
       
        return location, (a,b), angle

    def map_keypoints(self, homography: np.ndarray) -> Sequence[Keypoint]:
        """
        Maps the list of keypoints according to the homography.
        """
        # TODO: Map ellipse instead of point
        def map_keypoint(k: Keypoint) -> Keypoint:
            location, scale, rotation = k
            x = location[0]
            y = location[1]
            mapped_location = homography @ np.array([x, y, 1])
            mapped_location = mapped_location[:2] / mapped_location[2]
            second_location = homography @ np.array(
                [x - np.sin(rotation) * scale, y + np.cos(rotation) * scale, 1]
            )
            second_location = second_location[:2] / second_location[2]
            new_rotation = np.arccos(
                np.dot(second_location - mapped_location, np.array([0, 1]))
                / np.linalg.norm(second_location - mapped_location)
            )
            if second_location[0] > mapped_location[0]:
                new_rotation = (2 * np.pi - new_rotation) % (2 * np.pi)
            new_scale = float(np.linalg.norm(second_location - mapped_location))
            return mapped_location, new_scale, new_rotation.item()

        return list(map(map_keypoint, self.keypoints))


class BlobinatorTrainDataset(BlobinatorDataset):
    def __getitem__(self, index):
        background = self.backgrounds[index // len(self.keypoints)]
        transform = self.homographies[index // len(self.keypoints)]
        mapped_image = self.map_blobs(background, transform)
        mapped_keypoints = self.map_keypoints(transform)
        k = self.keypoints[index % len(self.keypoints)]
        k_mapped = mapped_keypoints[index % len(self.keypoints)]
        normalized_k = self.normalize_keypoint(
            k, into=(self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO)
        )
        normalized_k_mapped = self.normalize_keypoint(
            k_mapped, into=(self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO)
        )
        ellipse = self.conic_to_ellipse(self.keypoint_to_mapped_conic(transform, k))
        return (
            {
                "img": self.blobs,
                # TODO: Return meaningful values here
                "padLeft": 0,
                "padUp": 0,
            },
            {"img": mapped_image, "padLeft": 0, "padUp": 0},
            k,
            k_mapped,
            "img0000",
            f"img{(index % 4) + 1:04}",
            1.0,
            0.0,
            ellipse
        )

    def __len__(self):
        return len(self.backgrounds) * len(self.keypoints)

    def get_pairs(self, _):
        # Do nothing for now. This method could load new background images or so.
        pass


class BlobinatorTestDataset(BlobinatorDataset):
    def __getitem__(self, index):
        background = self.backgrounds[index // (len(self.keypoints) ** 2)]
        transform = self.homographies[index // (len(self.keypoints) ** 2)]
        mapped_image = self.map_blobs(background, transform)
        mapped_keypoints = self.map_keypoints(transform)
        keypoint_1_idx = index % (len(self.keypoints) ** 2) // len(self.keypoints)
        keypoint_2_idx = index % len(self.keypoints)
        k = self.keypoints[keypoint_1_idx]
        k_mapped = mapped_keypoints[keypoint_2_idx]
        # TODO: Normalize
        normalized_k = self.normalize_keypoint(
            k, into=(self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO)
        )
        normalized_k_mapped = self.normalize_keypoint(
            k_mapped, into=(self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO)
        )
        return (
            {
                "img": self.blobs,
                "padLeft": self.blob_sheet_pad_left,
                "padUp": self.blob_sheet_pad_up,
            },
            {"img": mapped_image, "padLeft": 0, "padUp": 0},
            normalized_k,
            normalized_k_mapped,
            "img0000",
            f"img{(index % 4) + 1:04}",
            k,
            k_mapped,
            1.0,
            1,
            0,
            1 if keypoint_1_idx == keypoint_2_idx else 0,
        )

    def __len__(self):
        return len(self.backgrounds) * len(self.keypoints) * len(self.keypoints)

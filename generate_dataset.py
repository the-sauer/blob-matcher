import argparse
from functools import reduce
import json
import logging
import math
import os
import random
import sys
from typing import Optional

import cv2 as cv
import kornia
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2

from modules.hardnet.utils import curry, fchain, flip


def read_json(path):
    with open(path, encoding="UTF-8") as f:
        return json.loads(f.read())


def sample_homography(
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


def map_blobs(background: torch.Tensor, homography: torch.Tensor, blobs: torch.Tensor, cfg) -> torch.Tensor:
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
    blobs = blobs.unsqueeze(0).expand(background.size(0), -1, -1, -1)
    warped_blobs = kornia.geometry.transform.warp_perspective(
        blobs,
        homography,
        (cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO),
    )
    #warped_blobs.reshape(background.size())
    mask = kornia.geometry.transform.warp_perspective(
        torch.ones_like(blobs),
        homography,
        (cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO)
    )
    mask = (mask > 0.999).float()
    warped_blobs = warped_blobs * mask + background * (1 - mask)
    return warped_blobs


def keypoint_to_mapped_conic(homography: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
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

def conic_to_ellipse(conic) -> Optional[tuple[np.typing.NDArray, tuple[float, float], float]]:
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

def ellipse_to_affine(ellipse) -> torch.Tensor:
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

def get_patch(img: torch.Tensor, A: torch.Tensor, cfg, pad_with=1.0):
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
    P = cfg.INPUT.IMAGE_SIZE
    PSF = cfg.BLOBINATOR.PATCH_SCALE_FACTOR
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


def create_manifest(cfg, path, background_files, keypoint_indices, blobboards):
    def keypoint_to_torch(keypoint):
        return torch.tensor(
            [keypoint["center"][0]["value"], keypoint["center"][1]["value"], keypoint["σ"]["value"], 0],
            dtype=torch.float32
        )

    def keypoints_to_torch(keypoints):
        return torch.stack(list(map(keypoint_to_torch, keypoints)))

    j = read_json(blobboards[0][1])["pattern_config"]["pattern_resolution"]
    blobboard_shape = (j["height"]["value"], j["height"]["value"])
    homographies = torch.stack([
        sample_homography(blobboard_shape, (cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO))
        for _ in range(len(background_files))
    ])
    torch.save(homographies, os.path.join(path, "homographies.pt"))
    with open(os.path.join(path, "backgrounds.txt"), "x", encoding="UTF-8") as background_list_file:
        background_list_file.writelines(map(
            lambda line: line[1] + f" {line[0] % len(blobboards):04}\n",
            enumerate(background_files)
        ))
    with open(os.path.join(path, "blobboards.txt"), "x", encoding="UTF-8") as blobs_file:
        blobs_file.writelines(map(lambda b: b[0] + "\n", blobboards))
    torch.save(torch.stack(list(map(
        lambda k: keypoints_to_torch(k)[keypoint_indices],
        map(lambda b: read_json(b[1])["blobs"], blobboards)
    ))), os.path.join(path, "keypoints.pt"))


def generate_dataset(cfg, path, is_validation=False):
    resize = torchvision.transforms.Resize((cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO))
    with open(os.path.join(path, "blobboards.txt"), "r", encoding="UTF-8") as blobs_file:
        blobboard_paths = list(map(str.strip, blobs_file.readlines()))
    os.makedirs(os.path.join(path, "warped_images"), exist_ok=True)
    blobboards = torch.empty((len(blobboard_paths), 1, 600, 800), dtype=torch.float32)
    for i, blobboard_path in enumerate(blobboard_paths):
        blobboards[i] = torchvision.io.decode_image(blobboard_path, torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255
    with open(os.path.join(path, "backgrounds.txt"), "r", encoding="UTF-8") as backgrounds_file:
        backgrounds_paths, blobboards_indices = list(zip(*map(str.split, backgrounds_file.readlines())))
        blobboards_indices = list(map(int, blobboards_indices))
    homographies = torch.load(os.path.join(path, "homographies.pt"))

    #warped_images = torch.empty(len(backgrounds_paths), 1, cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO)

    transforms = v2.Compose([
        v2.ColorJitter(),
        v2.GaussianBlur(kernel_size=(5, 5)),
        v2.GaussianNoise()
    ])
    for i, (background_path, blobboard_idx, homography) in enumerate(zip(backgrounds_paths, blobboards_indices, homographies)):
        if os.path.exists(os.path.join(path, "warped_images", f"{i:04}.png")):
            continue
        img = torchvision.io.decode_image(background_path, torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255
        assert img.size(0) == 1
        if img.shape[1] > img.shape[2]:
            crop1 = (img.shape[1] - img.shape[2]) // 2
            crop2 = crop1 if 2 * crop1 + img.shape[2] == img.shape[1] else crop1 + 1
            img = resize(img[:, crop1:-crop2, :])
        elif img.shape[2] > img.shape[1]:
            crop1 = (img.shape[2] - img.shape[1]) // 2
            crop2 = crop1 if 2 * crop1 + img.shape[1] == img.shape[2] else crop1 + 1
            img = resize(img[:, :, crop1:-crop2])
        else:
            img = resize(img)
        warped_image = map_blobs(img.unsqueeze(0), homography.unsqueeze(0), blobboards[blobboard_idx], cfg)
        warped_image = transforms(warped_image)
        torchvision.utils.save_image(warped_image, os.path.join(path, "warped_images", f"{i:04}.png"))

    keypoints = torch.load(os.path.join(path, "keypoints.pt"))

    os.makedirs(os.path.join(path, "patches", "anchors"), exist_ok=True)
    anchor_patches = torch.empty((*keypoints.shape[:2], 1, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE))
    for board_idx, blobboard in enumerate(blobboards):
        anchor_patch_transforms = torch.stack(list(map(lambda keypoint: ellipse_to_affine((keypoint[:2], (keypoint[2], keypoint[2]), 0)), keypoints[board_idx])))
        anchor_patches[board_idx] = get_patch(blobboard.unsqueeze(0).expand(anchor_patch_transforms.size(0), -1, -1, -1), anchor_patch_transforms, cfg)
        for blob_idx, patch in enumerate(anchor_patches[board_idx]):
            torchvision.utils.save_image(patch, os.path.join(path, "patches", "anchors", f"{board_idx:04}_{blob_idx:04}.png"))

    os.makedirs(os.path.join(path, "patches", "positives"), exist_ok=True)
    for background_idx, homography in enumerate(homographies):
        background = torchvision.io.decode_image(os.path.join(path, "warped_images", f"{background_idx:04}.png")).to(torch.float32) / 255
        mapped_keypoints = list(map(conic_to_ellipse, map(curry(keypoint_to_mapped_conic)(homography), keypoints[blobboards_indices[background_idx]])))
        positive_indices = torch.where(torch.tensor(list(map(lambda x: x is not None, mapped_keypoints))))[0]
        positive_transforms = torch.stack(list(map(ellipse_to_affine, filter(lambda x: x is not None, mapped_keypoints))))
        positive_patches = get_patch(background.unsqueeze(0).expand(positive_transforms.size(0), -1, -1, -1), positive_transforms, cfg)
        if is_validation:
            os.makedirs(os.path.join(path, "patches", "false_anchors"), exist_ok=True)
            false_anchor_map = torch.empty((2, keypoints[blobboards_indices[background_idx]].size(0)))
            false_anchor_map[0] = torch.randperm(keypoints[blobboards_indices[background_idx]].size(0))
            false_anchor_map[1, 1:] = false_anchor_map[0, :-1]
            false_anchor_map[1, 0] = false_anchor_map[0, -1]
        for blob_idx, patch in enumerate(positive_patches):
            torchvision.utils.save_image(patch, os.path.join(path, "patches", "positives", f"{background_idx:04}_{blobboards_indices[background_idx]:04}_{int(positive_indices[blob_idx]):04}.png"))
            if is_validation:
                false_anchor_idx = int(false_anchor_map[1, int(torch.where(false_anchor_map[0] == int(positive_indices[blob_idx]))[0])])
                assert false_anchor_idx != int(positive_indices[blob_idx])
                torchvision.utils.save_image(anchor_patches[board_idx, false_anchor_idx], os.path.join(path, "patches", "false_anchors", f"{background_idx:04}_{blobboards_indices[background_idx]:04}_{int(positive_indices[blob_idx]):04}.png"))



def main():
    sys.path.insert(0, os.getcwd())
    from configs.defaults import _C as cfg

    config_path = os.path.join(os.getcwd(), "configs", "init.yml")
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path", default=os.path.join(os.getcwd(), "data"))
    argument_parser.add_argument("--config_file",
                        default=config_path,
                        help="path to config file",
                        type=str)
    argument_parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = argument_parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    if not cfg.TRAINING.NO_CUDA:
        torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
        torch.backends.cudnn.deterministic = True

    # set random seeds
    random.seed(cfg.TRAINING.SEED)
    torch.manual_seed(cfg.TRAINING.SEED)
    np.random.seed(cfg.TRAINING.SEED)

    os.makedirs(args.path, exist_ok=True)
    os.makedirs(os.path.join(args.path, "training"), exist_ok=True)
    os.makedirs(os.path.join(args.path, "validation"), exist_ok=True)

    if not os.path.exists(os.path.join(args.path, "training", "blobboards.txt")) \
            or not os.path.exists(os.path.join(args.path, "training", "homographies.pt")) \
            or not os.path.exists(os.path.join(args.path, "training", "keypoints.pt")) \
            or not os.path.exists(os.path.join(args.path, "training", "backgrounds.txt")) \
            or not os.path.exists(os.path.join(args.path, "validation", "blobboards.txt")) \
            or not os.path.exists(os.path.join(args.path, "validation", "homographies.pt")) \
            or not os.path.exists(os.path.join(args.path, "validation", "keypoints.pt")) \
            or not os.path.exists(os.path.join(args.path, "validation", "backgrounds.txt")):

        background_filenames = fchain(
            list,
            curry(filter)(curry(flip(str.endswith))(".png")),
            curry(reduce)(list.__add__),
            curry(map)(lambda x: list(map(lambda f: os.path.join(x[0], f), x[2]))),
        )(os.walk(os.path.join("./data/backgrounds/openloris-location")))
        random.shuffle(background_filenames)
        training_background_filenames = background_filenames[int(len(background_filenames) * cfg.BLOBINATOR.VALIDATION.BACKGROUND_SPLIT):]
        training_background_filenames = training_background_filenames[:cfg.BLOBINATOR.TRAINING_DATASET_SIZE // cfg.BLOBINATOR.BLOBS_PER_IMAGE]
        validation_background_filenames = background_filenames[:int(len(background_filenames) * cfg.BLOBINATOR.VALIDATION.BACKGROUND_SPLIT)]
        validation_background_filenames = training_background_filenames[:cfg.BLOBINATOR.VALIDATION_DATASET_SIZE // cfg.BLOBINATOR.BLOBS_PER_IMAGE]

        blobboards = [
            (
                os.path.join(os.getcwd(), "data/patterns/blob_pattern_1DA.png"),
                os.path.join(os.getcwd(), "data/patterns/blob_pattern_1DA.json")
            )
        ]

        keypoint_indices = torch.randperm(min(
            len(blobs) for blobs in map(lambda x: read_json(x[1])["blobs"], blobboards)
        ))

        create_manifest(
            cfg,
            os.path.join(args.path, "training"),
            training_background_filenames,
            keypoint_indices[:int(keypoint_indices.size(0) * 0.8)],
            blobboards
        )
        create_manifest(
            cfg,
            os.path.join(args.path, "validation"),
            validation_background_filenames,
            keypoint_indices[int(keypoint_indices.size(0) * 0.8):],
            blobboards
        )
    # generate_dataset(cfg, os.path.join(args.path, "training"))
    generate_dataset(cfg, os.path.join(args.path, "validation"), is_validation=True)


if __name__ == "__main__":
    main()

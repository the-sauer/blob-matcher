import argparse
from functools import reduce
import json
import logging
import math
import os
import random
import sys
from typing import Optional


import kornia
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2


from modules import curry, fchain, flip
path_to_blobboards = os.path.join(os.getcwd(), "../BlobBoards.jl/python")
sys.path.insert(0, path_to_blobboards)
from BlobBoards.core import ureg, BlobGenerator
dpi = 1 / ureg.inch


def physical_to_logical_distance(x, resolution):
    return ((x * ureg.millimeter).to(ureg.inch) * (resolution * dpi)).magnitude


def physical_to_logical_coordinates(x, resolution, border_width, canvas_offset):
    return (
        (physical_to_logical_distance(border_width + canvas_offset[0], resolution)
            + (x[0] * ureg.millimeter).to(ureg.inch) * (resolution * dpi)).magnitude,
        (physical_to_logical_distance(border_width + canvas_offset[1], resolution)
            + (x[1] * ureg.millimeter).to(ureg.inch) * (resolution * dpi)).magnitude
    )


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
    patch_shape = tuple(map(float, patch_shape[::-1]))  # different convention [y, x]
    pts1 = pts1 * np.expand_dims(patch_shape, axis=0)
    pts2 = pts2 * np.expand_dims(patch_shape, axis=0)

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
    translation1 = np.identity(3)
    translation1[0, 2] = -original_shape[0] / 2
    translation1[1, 2] = -original_shape[1] / 2
    translation2 = np.identity(3)
    translation2[0, 2] = patch_shape[0] / 2
    translation2[1, 2] = patch_shape[1] / 2
    scale = min(patch_shape[0] / original_shape[0], patch_shape[1] / original_shape[1])
    return torch.tensor(homography @ translation2 @ np.diag([scale, scale, 1]) @ translation1, dtype=torch.float32)


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
    warped_blobs = kornia.geometry.transform.warp_perspective(
        blobs,
        homography,
        (cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO),
    )
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
    scale = torch.diag(torch.tensor([semi_major_axis, semi_minor_axis, 1], dtype=torch.float32))
    rotation = torch.eye(3)
    angle = torch.tensor(angle)
    rotation[:2, :2] = torch.tensor(
        [[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]],
        dtype=torch.float32
    )
    translation = torch.eye(3)
    translation[:2, 2] = location
    return translation @ rotation @ scale


def get_patch(img: torch.Tensor, A: torch.Tensor, cfg, sigma_cutoff=1.0, psf=None, pad_with=1.0):
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
    PSF = psf if psf is not None else cfg.BLOBINATOR.PATCH_SCALE_FACTOR
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
    r = torch.exp(torch.log(torch.tensor(PSF, device=device) / sigma_cutoff) * x) * sigma_cutoff  # [sigma_cutoff, PSF)
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
        torch.ones((1, 1, 1, 1), device=device).expand(B, 1, Hs, Ws),
        grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )

    background = torch.full_like(warped, pad_with)
    output = warped * mask + background * (1 - mask)
    return output


def create_manifest(cfg, path, background_files):
    gen = BlobGenerator()
    max_num_blobs = 0
    num_keypoints = torch.empty((len(background_files),))
    seeds = torch.randperm(10*len(background_files))[:len(background_files)]
    with open(os.path.join(path, "blobboards.txt"), "x", encoding="UTF-8") as blob_file:
        for i, seed in enumerate(seeds):
            res = gen.blob_board(
                paper_size="A4",
                dpi=600,
                board_size=(150*ureg.mm, 150*ureg.mm),
                sigma_cutoff=2.0,
                alpha=1.6,
                max_diameter_fraction=0.9,
                min_scale=0.2*ureg.mm,
                border_width=10.0*ureg.mm,
                ruler_width=5*ureg.mm,
                major_tick_spacing=10.0*ureg.mm,
                minor_tick_spacing=1.0*ureg.mm,
                dir=os.path.join(path, "patterns"),
                seed=seed,
                format="png"
            )
            max_num_blobs = max(max_num_blobs, res["num_blobs"])
            num_keypoints[i] = res["num_blobs"]
            blobboard_shape = (
                physical_to_logical_distance(res["height_mm"], 600),
                physical_to_logical_distance(res["width_mm"], 600)
            )
            blob_file.write(list(filter(lambda f: f.endswith(".png"), res["files_created"]))[0] + " ")
            blob_file.write(list(filter(lambda f: f.endswith(".json"), res["files_created"]))[0] + "\n")

    homographies = torch.stack([
        sample_homography(blobboard_shape, (cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO))
        for _ in range(len(background_files))
    ])
    torch.save(homographies, os.path.join(path, "homographies.pt"))
    with open(os.path.join(path, "backgrounds.txt"), "x", encoding="UTF-8") as background_list_file:
        background_list_file.writelines(map(
            lambda line: line[1] + "\n",
            enumerate(background_files)
        ))


def generate_dataset(cfg, path, is_validation=False):
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

    def keypoints_to_torch(keypoints, resolution, border_width, canvas_offset):
        return torch.stack(list(map(keypoint_to_torch(resolution, border_width, canvas_offset), keypoints)))

    resize = torchvision.transforms.Resize((cfg.TRAINING.PAD_TO, cfg.TRAINING.PAD_TO))
    os.makedirs(os.path.join(path, "warped_images"), exist_ok=True)
    with open(os.path.join(path, "backgrounds.txt"), "r", encoding="UTF-8") as backgrounds_file:
        backgrounds_paths = list(map(str.strip, backgrounds_file.readlines()))
    homographies = torch.load(os.path.join(path, "homographies.pt"))

    with open(os.path.join(path, "blobboards.txt"), "r", encoding="UTF-8") as blob_file:
        lines = blob_file.readlines()
        blobboard_paths, blobboard_info_paths = list(zip(*map(str.split, lines)))

    blobboard_json = read_json(os.path.join(path, "patterns", blobboard_info_paths[0]))
    blobboard_shape = blobboard_json["preamble"]["board_config"]["canvas_size"]
    blobboard_shape = (
        physical_to_logical_distance(
            blobboard_shape["height"]["value"],
            blobboard_json["preamble"]["board_config"]["print_density"]["value"]
        ),
        physical_to_logical_distance(
            blobboard_shape["height"]["value"],
            blobboard_json["preamble"]["board_config"]["print_density"]["value"]
        )
    )

    transforms = v2.Compose([
        v2.ColorJitter(),
        v2.GaussianBlur(kernel_size=(5, 5)),
        v2.GaussianNoise()
    ])
    keypoints = []
    for i in range(len(backgrounds_paths)):
        if os.path.exists(os.path.join(path, "warped_images", f"{i:04}.png")):
            continue
        img = torchvision.io.decode_image(backgrounds_paths[i], torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255
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
        blobboard = torchvision.io.decode_image(
            os.path.join(path, "patterns", blobboard_paths[i]),
            torchvision.io.ImageReadMode.GRAY).to(torch.float32
        ) / 255
        warped_image = map_blobs(img.unsqueeze(0), homographies[i].unsqueeze(0), blobboard.unsqueeze(0), cfg)
        warped_image = transforms(warped_image)
        torchvision.utils.save_image(warped_image, os.path.join(path, "warped_images", f"{i:04}.png"))

        blobboard_info = read_json(os.path.join(path, "patterns", blobboard_info_paths[i]))
        keypoints = keypoints_to_torch(
            blobboard_info["blobs"],
            blobboard_info["preamble"]["board_config"]["print_density"]["value"],
            blobboard_info["preamble"]["board_config"]["border_width"]["value"],
            (
                (blobboard_info["preamble"]["board_config"]["canvas_size"]["width"]["value"]
                    - blobboard_info["preamble"]["board_config"]["board_size"]["width"]["value"]) / 2,
                (blobboard_info["preamble"]["board_config"]["canvas_size"]["height"]["value"]
                    - blobboard_info["preamble"]["board_config"]["board_size"]["height"]["value"]) / 2,
            )
        )
        sigma_cutoff = blobboard_info["preamble"]["pattern_config"]["sigma_cutoff"]

        keypoint_pairs = zip(
            map(lambda k: (k[:2], (k[2], k[2]), 0), keypoints),
            map(conic_to_ellipse, map(curry(keypoint_to_mapped_conic)(homographies[i]), keypoints))
        )
        anchor_keypoints, positive_keypoints = list(zip(*filter(
            lambda x: x[1][0][0] >= 0 and x[1][0][0] < cfg.TRAINING.PAD_TO and x[1][0][1] >= 0 and x[1][0][1] < cfg.TRAINING.PAD_TO,
            filter(
                lambda x: x[0] is not None and x[1] is not None,
                keypoint_pairs
            )
        )))
        garbage_locations = torch.rand((len(positive_keypoints), 2)) * cfg.TRAINING.PAD_TO
        positive_keypoint_scales = torch.tensor(list(map(lambda k: k[1][0], positive_keypoints)))
        garbage_scales = torch.normal(torch.mean(positive_keypoint_scales).unsqueeze(0).expand(garbage_locations.size(0)), torch.std(positive_keypoint_scales).unsqueeze(0).expand(garbage_locations.size(0)))
        garbage_orientations = torch.rand((len(positive_keypoints),)) * 2 * np.pi
        garbage_keypoints = list(zip(garbage_locations, zip(garbage_scales, garbage_scales), garbage_orientations))

        anchor_transforms = torch.stack(list(map(ellipse_to_affine, anchor_keypoints)))
        positive_transforms = torch.stack(list(map(ellipse_to_affine, positive_keypoints)))
        garbage_transforms = torch.stack(list(map(ellipse_to_affine, garbage_keypoints)))

        for psf in [4, 8, 16, 32, 64, 96, 128]:
            os.makedirs(os.path.join(path, "patches", f"{int(psf)}", "anchors"), exist_ok=True)
            os.makedirs(os.path.join(path, "patches", f"{int(psf)}", "positives"), exist_ok=True)
            os.makedirs(os.path.join(path, "patches", f"{int(psf)}", "garbage"), exist_ok=True)

            anchor_patches = get_patch(
                blobboard.unsqueeze(0).expand(anchor_transforms.size(0), -1, -1, -1),
                anchor_transforms,
                cfg,
                sigma_cutoff=sigma_cutoff,
                psf=psf
            )
            positive_patches = get_patch(
                warped_image.expand(positive_transforms.size(0), -1, -1, -1),
                positive_transforms,
                cfg,
                sigma_cutoff=sigma_cutoff,
                psf=psf
            )
            garbage_patches = get_patch(
                img.unsqueeze(0).expand(garbage_transforms.size(0), -1, -1, -1),
                garbage_transforms,
                cfg,
                psf=psf
            )
            for j in range(anchor_patches.size(0)):
                torchvision.utils.save_image(
                    anchor_patches[j],
                    os.path.join(path, "patches", f"{int(psf)}", "anchors", f"{i:04}_{j:04}.png")
                )
                torchvision.utils.save_image(
                    positive_patches[j],
                    os.path.join(path, "patches", f"{int(psf)}", "positives", f"{i:04}_{j:04}.png")
                )
                if j < garbage_patches.size(0):
                    torchvision.utils.save_image(
                        garbage_patches[j],
                        os.path.join(path, "patches", f"{int(psf)}", "garbage", f"{i:04}_{j:04}.png")
                    )


def main():
    sys.path.insert(0, os.getcwd())
    from configs.defaults import _C as cfg

    config_path = os.path.join(os.getcwd(), "configs", "init.yml")
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--path",
        default=os.path.join(os.getcwd(), "data")
    )
    argument_parser.add_argument(
        "--config_file",
        default=config_path,
        help="path to config file",
        type=str
    )
    argument_parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
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
            or not os.path.exists(os.path.join(args.path, "training", "backgrounds.txt")) \
            or not os.path.exists(os.path.join(args.path, "validation", "blobboards.txt")) \
            or not os.path.exists(os.path.join(args.path, "validation", "homographies.pt")) \
            or not os.path.exists(os.path.join(args.path, "validation", "backgrounds.txt")):

        background_filenames = fchain(
            list,
            curry(filter)(curry(flip(str.endswith))(".png")),
            curry(reduce)(list.__add__),
            curry(map)(lambda x: list(map(lambda f: os.path.join(x[0], f), x[2]))),
        )(os.walk(os.path.join("./data/backgrounds/openloris-location")))
        random.shuffle(background_filenames)
        # Fairly distribute the available backgrounds between training and validation datasets
        validation_split = cfg.BLOBINATOR.VALIDATION_NUM_BACKGROUNDS / cfg.BLOBINATOR.TRAINING_NUM_BACKGROUNDS
        training_background_filenames = background_filenames[int(len(background_filenames) * validation_split):]
        training_background_filenames = training_background_filenames[:cfg.BLOBINATOR.TRAINING_NUM_BACKGROUNDS]
        validation_background_filenames = background_filenames[:int(len(background_filenames) * validation_split)]
        validation_background_filenames = validation_background_filenames[:cfg.BLOBINATOR.VALIDATION_NUM_BACKGROUNDS]

        create_manifest(
            cfg,
            os.path.join(args.path, "training"),
            training_background_filenames,
        )
        create_manifest(
            cfg,
            os.path.join(args.path, "validation"),
            validation_background_filenames,
        )
    generate_dataset(cfg, os.path.join(args.path, "training"))
    generate_dataset(cfg, os.path.join(args.path, "validation"), is_validation=True)


if __name__ == "__main__":
    main()

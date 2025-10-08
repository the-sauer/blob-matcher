#import juliacall    # Has to be imported before pytorch (https://github.com/pytorch/pytorch/issues/78829)

import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from configs.defaults import _C as cfg
from modules.ptn.pytorch.blobinator_dataset import BlobinatorTrainDataset

def repeat(s):
    while True:
        for e in s:
            yield e

dataset = BlobinatorTrainDataset(cfg)
dataset.__getitem__(0)
img = None
for i, data, color in zip(range(238), dataset, repeat(matplotlib.colors.BASE_COLORS.values())):
    _, mapped_image, _, mapped_keypoint, *_ , ellipse = data
    if img is None:
        img = np.stack((mapped_image["img"].reshape((1500, 1500)),) * 3, axis=-1).astype(np.uint8) # Convert to RGB
    location, (semi_major_axis, semi_minor_axis), rotation = ellipse
    x = location[0]
    y = location[1]
    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    if i == 20:
        cv.ellipse(img, (int(x), int(y)), (int(semi_major_axis), int(semi_minor_axis)), rotation / np.pi * 180, 0, 360, color)
        A = dataset.ellipse_to_affine(ellipse)
        for p1, p2 in zip(
            [np.array([-0.5,-0.5,1]), np.array([-0.5,0.5,1]), np.array([0.5,0.5,1]), np.array([0.5,-0.5,1])],
            [np.array([-0.5,0.5,1]), np.array([0.5,0.5,1]), np.array([0.5,-0.5,1]), np.array([-0.5,-0.5,1])]
        ):
            A_inv = np.identity(3)
            A_inv[:2,:] = cv.invertAffineTransform(A[:2,:])
            patch_size = 32
            patch_transform = np.diag([patch_size, patch_size, 1]) @ np.array([[1,0,0.5], [0,1,0.5], [0,0,1]]) @ A_inv
            patch = cv.warpAffine(img, patch_transform[:2,:], (patch_size, patch_size))
            cv.imwrite(f"patch_{i}.png", patch)
            p1 = A @ p1
            p2 = A @ p2
            print(p1)
            cv.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255))
cv.imwrite("transformed.png", img)

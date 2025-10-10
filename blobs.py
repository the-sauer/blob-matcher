#import juliacall    # Has to be imported before pytorch (https://github.com/pytorch/pytorch/issues/78829)

import os

import cv2 as cv

from configs.defaults import _C as cfg
from modules.ptn.pytorch.blobinator_dataset import BlobinatorTrainDataset

dir = "./data/patches"

dataset = BlobinatorTrainDataset(cfg)

for i, (anchor_patch, positive_patch, garbage_patch, garbage_available) in zip(range(3), dataset):
    os.makedirs(os.path.join(dir, f"{i:02}"), exist_ok=True)
    cv.imwrite(os.path.join(dir, f"{i:02}", "anchor.png"), anchor_patch)
    cv.imwrite(os.path.join(dir, f"{i:02}", "positive.png"), positive_patch)
    if garbage_available:
        cv.imwrite(os.path.join(dir, f"{i:02}", "garbage.png"), garbage_patch)

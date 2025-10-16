import argparse
import json
import os
import sys

import numpy as np
import cv2 as cv

sys.path.insert(0, os.getcwd())

from configs.defaults import _C as cfg
from modules.ptn.pytorch.blobinator_dataset import BlobinatorDataset
from modules.hardnet.utils import curry, fchain


class BlobinatorDetectorDataset(BlobinatorDataset):
    def __iter__(self):
        for background, homography in zip(self.backgrounds, self.homographies):
            warped_image = self.map_blobs(background, homography)
            meta_data = {}
            meta_data["keypoints"] = list(map(
                fchain(
                    self.elllipse_to_detection,
                    self.conic_to_ellipse,
                    curry(self.keypoint_to_mapped_conic)(homography)
                ),
                self.keypoints
            ))
            yield warped_image, meta_data

    def elllipse_to_detection(self, ellipse):
        return {
            "center": {
                "x": ellipse[0][0],
                "y": ellipse[0][1]
            },
            "semimajor": ellipse[1][0],
            "semiminor": ellipse[1][1],
            "angle": ellipse[2]
        }


def main():
    argument_parser = argparse.ArgumentParser()
    config_path = os.path.join(os.getcwd(), "configs", "init.yml")

    argument_parser.add_argument(
        "--config_file",
        default=config_path,
        help="path to config file",
        type=str
    )
    argument_parser.add_argument(
        "--out-dir",
        default="out",
        type=str,
        help="The directory in which to write the image and metadata files"
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

    dataset = BlobinatorDetectorDataset(cfg)

    for i, (image, meta_data) in enumerate(dataset):
        cv.imwrite(os.path.join(args.out_dir, f"img_{i:04}.png"), (image * 255).astype(np.uint8))
        with open(os.path.join(args.out_dir, f"metadata_{i:04}.json"), "w") as f:
            f.write(json.dumps(meta_data, indent=4))


if __name__ == "__main__":
    main()

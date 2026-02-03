# Copyright 2025 Hendrik Sauer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from datetime import datetime
import os


from blob_matcher.configs.defaults import _C as cfg
from blob_matcher.scripts.hardnet import run_training


def main():
    cfg.LOGGING.LOG_DIR = os.path.join('data/logs/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
    cfg.LOGGING.MODEL_DIR = os.path.join('data/models/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
    cfg.LOGGING.IMGS_DIR = os.path.join('data/images/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")

    for loss in ["triplet_margin"]:
        for batch_reduce in ["min"]:
            for optimizer in ["sgd", "adam"]:
                for scale in [96, 128, 64]:
                    for resolution in [32, 64, 128]:
                        experiment_name = f"scale_{scale}_res_{resolution}_br_{batch_reduce}_loss_{loss}_optimizer_{optimizer}"
                        config = copy.deepcopy(cfg)
                        if resolution == 64:
                            config.TRAINING.BATCH_SIZE = 800
                        elif resolution == 128:
                            config.TRAINING.BATCH_SIZE = 200
                        config.TRAINING.EXPERIMENT_NAME = experiment_name
                        config.TRAINING.SCALE = scale
                        config.TEST.SCALE = scale
                        config.BLOBINATOR.PATCH_SCALE_FACTOR = scale
                        config.INPUT.IMAGE_SIZE = resolution
                        config.TEST.IMAGE_SIZE = resolution
                        config.TRAINING.IMAGE_SIZE = resolution
                        config.TRAINING.LOSS = loss
                        config.TRAINING.BATCH_REDUCE = batch_reduce
                        config.TRAINING.OPTIMIZER = optimizer

                        config.TRAINING.EXPERIMENT_NAME = f"{experiment_name}_easy"
                        config.BLOBINATOR.DATASET_PATH = "./data/datasets/new/easy"
                        config.TRAINING.EPOCHS = 50
                        print(f"Running {experiment_name} now")
                        try:
                            if not os.path.exists(os.path.join(
                                config.LOGGING.MODEL_DIR,
                                f"{experiment_name}_easy",
                                "model_checkpoint_49.pth"
                            )):
                                run_training(config)
                            config.TEST.EVAL_INTERVAL = 20
                            config.LOGGING.LOG_DIR = os.path.join('data/logs/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
                            config.LOGGING.MODEL_DIR = os.path.join('data/models/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
                            config.LOGGING.IMGS_DIR = os.path.join('data/images/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
                            config.TRAINING.EXPERIMENT_NAME = f"{experiment_name}_real"
                            config.BLOBINATOR.DATASET_PATH = "./data/datasets/new/real"
                            config.TRAINING.EPOCHS = 200
                            config.TRAINING.RESUME = os.path.join(
                                config.LOGGING.MODEL_DIR,
                                f"{experiment_name}_easy",
                                "model_checkpoint_49.pth"
                            )
                            if not os.path.exists(os.path.join(
                                config.LOGGING.MODEL_DIR,
                                f"{experiment_name}_real",
                                "model_checkpoint_199.pth"
                            )):
                                run_training(config)
                        except Exception as e:
                            print(e)
                            continue


if __name__ == "__main__":
    main()

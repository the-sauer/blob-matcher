import copy
from datetime import datetime
import os


from blob_matcher.configs.defaults import _C as cfg
from blob_matcher.scripts.hardnet import run_training


def main():
    with open("completed.txt", "r", encoding="utf-8") as f:
        completed = f.readlines()

    cfg.LOGGING.LOG_DIR = os.path.join('data/logs/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
    cfg.LOGGING.MODEL_DIR = os.path.join('data/models/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
    cfg.LOGGING.IMGS_DIR = os.path.join('data/images/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")

    for loss in ["triplet_margin"]:
        for batch_reduce in ["min"]:
            for optimizer in ["sgd", "adam"]:
                for scale in [96, 128, 64]:
                    for resolution in [32, 64, 128]:
                        experiment_name = f"scale_{scale}_res_{resolution}_br_{batch_reduce}_loss_{loss}_optimizer_{optimizer}"
                        if experiment_name + "\n" in completed:
                            continue
                        config = copy.deepcopy(cfg)
                        config.TRAINING.EXPERIMENT_NAME = experiment_name
                        config.TRAINING.SCALE = scale
                        config.TEST.SCALE = scale
                        config.BLOBINATOR.PATCH_SCALE_FACTOR = scale
                        config.INPUT.IMAGE_SIZE = resolution
                        config.TEST.IMAGE_SIZE = resolution
                        config.TRAINING.IMAGE_SIZE = resolution
                        config.TRAINING.LOSS = loss
                        config.TRAINING.BATCH_REDUCE = loss
                        config.TRAINING.OPTIMIZER = optimizer

                        config.BLOBINATOR.DATASET_PATH = "./data/datasets/new/easy"
                        config.BLOBINATOR.EPOCHS = 50
                        run_training(config)
                        config.BLOBINATOR.DATASET_PATH = "./data/datasets/new/hard"
                        config.BLOBINATOR.EPOCHS = 100
                        config.BLOBINATOR.RESUME_TRAINING = os.path.join(
                            config.TRAINING.MODEL_DIR,
                            experiment_name,
                            "model_checkpoint_49.pth"
                        )
                        run_training(config)
                        config.BLOBINATOR.DATASET_PATH = "./data/datasets/new/real"
                        config.BLOBINATOR.EPOCHS = 300
                        config.BLOBINATOR.RESUME_TRAINING = os.path.join(
                            config.TRAINING.MODEL_DIR,
                            experiment_name,
                            "model_checkpoint_99.pth"
                        )
                        run_training(config)

                        with open("completed.txt", "w", encoding="utf-8") as f:
                            f.write(experiment_name + "\n")


if __name__ == "__main__":
    main()

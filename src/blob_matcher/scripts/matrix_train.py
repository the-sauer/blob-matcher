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
            for optimizer in ["sgd"]:
                for scale in [96, 128]:
                    for resolution in [32, 64]:
                        experiment_name = f"scale_{scale}_res_{resolution}_br_{batch_reduce}_loss_{loss}_optimizer_{optimizer}"
                        config = copy.deepcopy(cfg)

                        config.TRAINING.BATCH_SIZE = 200
                        config.TRAINING.TEST_BATCH_SIZE = 200
                        config.TEST.TEST_BATCH_SIZE = 200

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
                            config.TEST.EVAL_INTERVAL = 100
                            config.LOGGING.LOG_DIR = os.path.join('data/logs/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
                            config.LOGGING.MODEL_DIR = os.path.join('data/models/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
                            config.LOGGING.IMGS_DIR = os.path.join('data/images/', datetime.today().strftime('%Y_%m_%d'), "matrix_train")
                            config.TRAINING.EXPERIMENT_NAME = f"{experiment_name}_real"
                            config.BLOBINATOR.DATASET_PATH = "./data/datasets/2025_11_14/real"
                            config.TRAINING.EPOCHS = 200
                            config.TRAINING.RESUME = os.path.join(
                                "./data/models/2025_11_11/matrix_train",
                                f"{experiment_name}_easy",
                                "model_checkpoint_49.pth"
                            )
                            if not os.path.exists(os.path.join(
                                config.LOGGING.MODEL_DIR,
                                f"{experiment_name}_real",
                                "model_checkpoint_199.pth"
                            )):
                                for i in range(199, 0, -1):
                                    if os.path.exists(os.path.join(
                                        config.LOGGING.MODEL_DIR,
                                        f"{experiment_name}_real",
                                        f"model_checkpoint_{i}.pth"
                                    )):
                                        config.TRAINING.RESUME = os.path.join(
                                            config.LOGGING.MODEL_DIR,
                                            f"{experiment_name}_real",
                                            f"model_checkpoint_{i}.pth"
                                        )
                                        break
                                run_training(config)
                        except KeyboardInterrupt:
                            continue
                        except Exception as e:
                            print(e)
                            continue


if __name__ == "__main__":
    main()

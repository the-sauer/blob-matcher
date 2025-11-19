import os
import re


import torch
import torchvision


from blob_matcher.hardnet.eval_metrics import ErrorRateAt95Recall
from blob_matcher.hardnet.losses import distance_matrix_vector
from blob_matcher.hardnet.models import HardNet


def evaluate(model_path, scale, resolution):
    resize = torchvision.transforms.Resize((resolution, resolution))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HardNet(patch_size=resolution)
    model.load_state_dict(torch.load(
        model_path,
        weights_only=False,
        map_location=device
    )["state_dict"])
    model.to(device)
    model.eval()
    positive_path_regex = re.compile("(\\d+)_(\\d+).png")
    dataset_path = "./data/datasets/2025_11_19/real/validation"
    patch_files = os.listdir(os.path.join(dataset_path, f"patches/{scale}/positives"))

    print(f"\n# Evaluating for λ={scale} and resolution=({resolution}×{resolution})")

    fpr95_sum = 0
    fpr95_num = 0
    for i in range(0, 123):
        regex = re.compile(f"{i:04}_\\d+\\.png")
        patches = list(filter(lambda f: regex.match(f) is not None, patch_files))
        if len(patches) == 0:
            continue

        anchor_patches = torch.empty((len(patches), 1, resolution, resolution)).to(device)
        positive_patches = torch.empty((len(patches), 1, resolution, resolution)).to(device)

        for j, patch_file in enumerate(patches):
            match = positive_path_regex.search(os.path.basename(patch_file))
            board_idx, blob_idx = match.group(1), match.group(2)
            positive_patches[j] = resize(torchvision.io.decode_image(os.path.join(dataset_path, f"patches/{scale}/positives/{board_idx}_{blob_idx}.png"), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255)
            anchor_patches[j] = resize(torchvision.io.decode_image(os.path.join(dataset_path, f"patches/{scale}/anchors/{board_idx}_{blob_idx}.png"), torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255)
        garbage_patch_files = os.listdir(os.path.join(dataset_path, f"patches/{scale}/garbage"))
        garbage_patch_files = list(filter(lambda f: regex.match(f) is not None, garbage_patch_files))
        garbage_patches = resize(torch.empty((len(garbage_patch_files), 1, 32, 32)).to(device))
        for j, patch_file in enumerate(garbage_patch_files):
            garbage_patches[j] = resize(torchvision.io.decode_image(os.path.join(dataset_path, f"patches/{scale}/garbage/{patch_file}"), torchvision.io.ImageReadMode.GRAY))

        anchor_features, _ = model(anchor_patches)
        positive_features, _ = model(positive_patches)
        garbage_features, _ = model(garbage_patches)

        distances = distance_matrix_vector(anchor_features, torch.concat((positive_features, garbage_features))).detach().cpu().numpy().flatten()
        labels = torch.eye(anchor_features.size(0), positive_features.size(0) + garbage_features.size(0)).cpu().numpy().flatten()
        fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
        print(f"{fpr95=} for image {i} with {anchor_features.size(0)} features")
        fpr95_num += distances.size
        fpr95_sum += distances.size * fpr95

    avg_fpr95 = fpr95_sum / fpr95_num
    print(f"{avg_fpr95=}")


def main():
    base_path = "./data/models/2025_11_11/matrix_train"
    for resolution in [32, 64]:
        for scale in [96, 128]:
            experiment_name = f"scale_{scale}_res_{resolution}_br_min_loss_triplet_margin_optimizer_sgd_real"
            try:
                evaluate(os.path.join(base_path, experiment_name, "model_checkpoint_199.pth"), scale, resolution)
            except FileNotFoundError as e:
                print("\n", e)
                continue


if __name__ == "__main__":
    main()

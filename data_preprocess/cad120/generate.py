import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
import pickle
import shutil
from tqdm import tqdm
import yaml


def get_keypoint(keypoints, file_name, num_classes):
    assert len(file_name) == 7
    image_id = int(file_name[1:5])
    bb_id = int(file_name[6])

    keypoint_list = []
    for i in range(num_classes):
        mask = (
            (keypoints[:, 0].astype(np.int32) == image_id)
            & (keypoints[:, 1].astype(np.int32) == bb_id)
            & (keypoints[:, 2].astype(np.int32) == i + 1)
        )
        coords = keypoints[mask][:, 3:].reshape(-1, 2)
        coords = np.flip(coords, axis=1)

        # assert coords.shape[0] == 1, f"{file_name}: {coords.shape[0]}"
        keypoint_list.append((i, coords.tolist()))

    # assert len(keypoint_dict) > 0, file_name

    return keypoint_list


def train_set(cad120_path, save_path, split_mode):
    keypoints = np.loadtxt("keypoints.txt", delimiter=",")

    os.makedirs(save_path, exist_ok=True)

    images_path = os.path.join(save_path, "affordance", "images")
    labels_path = os.path.join(save_path, "affordance", "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    keypoint_dict = dict()

    fb = open(os.path.join(save_path, "train_affordance.txt"), "w")
    for line in tqdm(open(f"train_{split_mode}_split_id.txt", "r"), ncols=80):
        line = line.strip()

        image_path = os.path.join(cad120_path, "object_crop_images", line + ".png")
        image_save_path = os.path.join(images_path, line + ".png")
        label_path = os.path.join(
            cad120_path, "segmentation_mat", line + "_binary_multilabel.mat"
        )
        label_save_path = os.path.join(labels_path, line + ".pkl")

        image = Image.open(image_path)
        bands = image.getbands()
        if bands == ("L"):
            image = np.asarray(image)
            image = np.stack([image, image, image], axis=2)
            image = Image.fromarray(image)
        elif bands != ("R", "G", "B"):
            raise ValueError(f"{image_path}: {bands}")
        image.save(image_save_path)
        
        label = loadmat(label_path)["data"]

        # assert label.max() == 1

        label = (label > 0).astype(np.uint8)
        with open(label_save_path, "wb") as f:
            pickle.dump(label, f)

        keypoint_dict[line] = get_keypoint(keypoints, line, num_classes=6)

        fb.write(
            os.path.relpath(image_save_path, save_path)
            + ","
            + os.path.relpath(label_save_path, save_path)
            + "\n"
        )
    fb.close()

    with open(os.path.join(save_path, "train_affordance_keypoint.yaml"), "w") as fb:
        yaml.safe_dump(keypoint_dict, fb)


def val_set(cad120_path, save_path, split_mode):
    os.makedirs(save_path, exist_ok=True)

    images_path = os.path.join(save_path, "affordance", "images")
    labels_path = os.path.join(save_path, "affordance", "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    fb = open(os.path.join(save_path, "val_affordance.txt"), "w")
    for line in tqdm(open(f"test_{split_mode}_split_id.txt", "r"), ncols=80):
        line = line.strip()

        image_path = os.path.join(cad120_path, "object_crop_images", line + ".png")
        image_save_path = os.path.join(images_path, line + ".png")
        label_path = os.path.join(
            cad120_path, "segmentation_mat", line + "_binary_multilabel.mat"
        )
        label_save_path = os.path.join(labels_path, line + ".pkl")

        # assert not os.path.exists(image_save_path) and not os.path.exists(
        #     label_save_path
        # )

        image = Image.open(image_path)
        bands = image.getbands()
        if bands == ("L"):
            image = np.asarray(image)
            image = np.stack([image, image, image], axis=2)
            image = Image.fromarray(image)
        elif bands != ("R", "G", "B"):
            raise ValueError(f"{image_path}: {bands}")
        image.save(image_save_path)

        label = loadmat(label_path)["data"]

        # assert label.max() == 1

        label = (label > 0).astype(np.uint8)
        with open(label_save_path, "wb") as f:
            pickle.dump(label, f)

        fb.write(
            os.path.relpath(image_save_path, save_path)
            + ","
            + os.path.relpath(label_save_path, save_path)
            + "\n"
        )
    fb.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    split_mode = "object"

    source_path = "/home/DISCOVER_summer2022/cuily/dataset/CAD120"
    output_path = os.path.join(
        "/home/DISCOVER_summer2022/cuily/dataset/cad120", split_mode
    )

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    train_set(source_path, output_path, split_mode)
    val_set(source_path, output_path, split_mode)


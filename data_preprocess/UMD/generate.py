import os
import numpy as np
from scipy.io import loadmat
import pickle
from tqdm import tqdm
import yaml
from glob import glob
from PIL import Image
import cv2
import shutil


def get_keypoint(labels):
    keypoint_dict = {}
    for idx, label in enumerate(labels):
        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        points = []
        for contour in contours:
            mm = cv2.moments(contour)
            point = [
                int(mm["m10"] / mm["m00"]),
                int(mm["m01"] / mm["m00"]),
            ]  # (x, y) -> (w, h) in skimage
            if label[point[1], point[0]] == 1:
                points.append(point)
        keypoint_dict[idx] = points

    return keypoint_dict


def split_dataset(umd_path, split_mode):
    cnt_train = 0
    cnt_test = 0

    fb_train = open(f"train_{split_mode}_split_id.txt", "w")
    fb_test = open(f"test_{split_mode}_split_id.txt", "w")
    for line in open(os.path.join(f"{umd_path}", f"{split_mode}_split.txt"), "r"):
        line = line.strip()
        sort, dir_name = line.split(" ")

        img_paths = sorted(glob(os.path.join(umd_path, "tools", dir_name, "*_rgb.jpg")))

        if sort == "1":
            for img_path in img_paths:
                file_id = os.path.basename(img_path).split(".")[0]
                file_id = "_".join(file_id.split("_")[:-1])
                file_id = os.path.join(dir_name, file_id)

                fb_train.write(file_id + "\n")
                cnt_train += 1
        elif sort == "2":
            for img_path in img_paths:
                file_id = os.path.basename(img_path).split(".")[0]
                file_id = "_".join(file_id.split("_")[:-1])
                file_id = os.path.join(dir_name, file_id)

                fb_test.write(file_id + "\n")
                cnt_test += 1
    fb_train.close()
    fb_test.close()

    print(f"The number of train dataset: {cnt_train}")
    print(f"The number of test dataset: {cnt_test}")


def gen_dataset(umd_path, save_path, split_mode):
    os.makedirs(save_path, exist_ok=True)

    images_path = os.path.join(save_path, "images")
    labels_path = os.path.join(save_path, "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    keypoint_dict = {}

    fb = open(os.path.join(save_path, "train_affordance.txt"), "w")
    for line in tqdm(open(f"train_{split_mode}_split_id.txt", "r"), ncols=80):
        dir_name, file_id = line.strip().split("/")

        image_path = os.path.join(umd_path, "tools", dir_name, f"{file_id}_rgb.jpg")
        image_save_path = os.path.join(images_path, f"{file_id}.png")

        image = Image.open(image_path).convert("RGB")
        image.save(image_save_path)

        label_path = os.path.join(
            umd_path, "tools", dir_name, f"{file_id}_label_rank.mat"
        )
        label_save_path = os.path.join(labels_path, f"{file_id}.pkl")

        label = loadmat(label_path)["gt_label"].astype(np.uint8).transpose(2, 0, 1)
        for i, it in enumerate(label):
            it[it > 3] = 0
            it[it > 0] = 1
            label[i] = it

        keypoint_dict[file_id] = get_keypoint(label)

        label = label.transpose(1, 2, 0)
        with open(label_save_path, "wb") as f:
            pickle.dump(label, f)

        fb.write(
            os.path.relpath(image_save_path, save_path)
            + ","
            + os.path.relpath(label_save_path, save_path)
            + "\n"
        )
    fb.close()

    yaml_path = os.path.join(save_path, "train_affordance_keypoint.yaml")
    with open(yaml_path, "w") as fb:
        yaml.safe_dump(keypoint_dict, fb)

    fb = open(os.path.join(save_path, "val_affordance.txt"), "w")
    for line in tqdm(open(f"test_{split_mode}_split_id.txt", "r"), ncols=80):
        dir_name, file_id = line.strip().split("/")

        image_path = os.path.join(umd_path, "tools", dir_name, f"{file_id}_rgb.jpg")
        image_save_path = os.path.join(images_path, f"{file_id}.png")

        image = Image.open(image_path).convert("RGB")
        image.save(image_save_path)

        label_path = os.path.join(
            umd_path, "tools", dir_name, f"{file_id}_label_rank.mat"
        )
        label_save_path = os.path.join(labels_path, f"{file_id}.pkl")

        label = loadmat(label_path)["gt_label"].astype(np.uint8).transpose(2, 0, 1)
        for i, it in enumerate(label):
            it[it > 3] = 0
            it[it > 0] = 1
            label[i] = it
        label = label.transpose(1, 2, 0)
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

    split_mode = "novel"
    source_path = "../../../dataset/UMD/part-affordance-dataset"
    output_path = os.path.join("../../../dataset/umd", split_mode)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    split_dataset(source_path, split_mode)
    gen_dataset(source_path, output_path, split_mode)

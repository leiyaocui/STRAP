import os
import numpy as np
from scipy.io import loadmat
import pickle
from tqdm import tqdm
import yaml
from glob import glob
from PIL import Image


def get_keypoint(keypoints, visible_info, file_id, num_classes):
    assert len(file_id) == 7
    image_id = int(file_id[1:5])
    bb_id = int(file_id[6])

    keypoint_dict = {}
    for i in range(num_classes):
        if visible_info[i] == 0:
            continue
        mask = (
            (keypoints[:, 0].astype(np.int32) == image_id)
            & (keypoints[:, 1].astype(np.int32) == bb_id)
            & (keypoints[:, 2].astype(np.int32) == i + 1)
        )
        coords = keypoints[mask][:, 3:].reshape(-1, 2)
        coords = np.flip(
            coords, axis=1
        )  # because the shape style of pillow is (w, h, c).
        coords = coords - 1
        new_coords = np.round(coords, decimals=0).astype(np.int32)
        
        del_coords = np.argwhere(new_coords >= 320)
        if del_coords.size > 0:
            print()
            for it in del_coords:
                print(new_coords)
                new_coords[it[0], it[1]] -= 1
                print(new_coords)
        
        keypoint_dict[i] = np.array(new_coords).tolist()

    return keypoint_dict


def split_dataset(cad120_path, split_mode):
    file_all_list = sorted(
        glob(os.path.join(cad120_path, "object_crop_images", "*.png"))
    )
    for i in range(len(file_all_list)):
        file_all_list[i] = os.path.basename(file_all_list[i])

    print(f"The number of total dataset: {len(file_all_list)}")

    train_file_list = []
    test_file_list = file_all_list.copy()

    fb = open(f"train_{split_mode}_split_id.txt", "w")
    for line in open(
        os.path.join(f"{cad120_path}", "lists", f"train_{split_mode}_split.txt"), "r"
    ):
        line = line.strip()

        file_name = os.path.basename(line.split(" ")[0])
        file_id = file_name.split(".")[0]

        train_file_list.append(file_name)
        test_file_list.remove(file_name)
        fb.write(file_id + "\n")
    fb.close()

    print(f"The number of train dataset: {len(train_file_list)}")
    print(f"The number of test dataset: {len(test_file_list)}")

    assert len(train_file_list) + len(test_file_list) == len(file_all_list)

    fb = open(f"test_{split_mode}_split_id.txt", "w")
    for it in test_file_list:
        file_id = it.split(".")[0]
        fb.write(file_id + "\n")
    fb.close()


def gen_keypoint_list(cad120_path, save_path, split_mode):
    os.makedirs(save_path, exist_ok=True)

    visible_info_dict = dict()
    fb = open(os.path.join(cad120_path, "visible_affordance_info.txt"), "r")
    for line in fb:
        line = line.strip().split(" ")
        file_id = line[0].split(".")[0]
        visible_info_dict[file_id] = np.array(line[1:], dtype=np.uint8).tolist()
    fb.close()

    keypoints = np.loadtxt("keypoints.txt", delimiter=",")
    keypoint_dict = dict()

    for line in tqdm(open(f"train_{split_mode}_split_id.txt", "r"), ncols=80):
        file_id = line.strip()

        visible_info = visible_info_dict[file_id]
        keypoint_dict[file_id] = get_keypoint(
            keypoints, visible_info, file_id, num_classes=6
        )

    yaml_path = os.path.join(save_path, "train_affordance_keypoint.yaml")
    with open(yaml_path, "w") as fb:
        yaml.safe_dump(keypoint_dict, fb)


def gen_dataset(cad120_path, save_path, split_mode):
    os.makedirs(save_path, exist_ok=True)

    images_path = os.path.join(save_path, "images")
    labels_path = os.path.join(save_path, "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    fb = open(os.path.join(save_path, "train_affordance.txt"), "w")
    for line in tqdm(open(f"train_{split_mode}_split_id.txt", "r"), ncols=80):
        file_id = line.strip()

        image_path = os.path.join(cad120_path, "object_crop_images", f"{file_id}.png")
        image_save_path = os.path.join(images_path, f"{file_id}.png")

        image = Image.open(image_path).convert("RGB").crop((0, 0, 320, 320))
        image.save(image_save_path)

        label_path = os.path.join(
            cad120_path, "segmentation_mat", f"{file_id}_binary_multilabel.mat"
        )
        label_save_path = os.path.join(labels_path, f"{file_id}.pkl")

        label = loadmat(label_path)["data"].astype(np.uint8)
        label = [
            Image.fromarray(label[:, :, i], mode="L").crop((0, 0, 320, 320))
            for i in range(label.shape[2])
        ]
        label = np.stack(label, axis=2)
        with open(label_save_path, "wb") as f:
            pickle.dump(label, f)

        fb.write(
            os.path.relpath(image_save_path, save_path)
            + ","
            + os.path.relpath(label_save_path, save_path)
            + "\n"
        )
    fb.close()

    fb = open(os.path.join(save_path, "val_affordance.txt"), "w")
    for line in tqdm(open(f"test_{split_mode}_split_id.txt", "r"), ncols=80):
        file_id = line.strip()

        image_path = os.path.join(cad120_path, "object_crop_images", f"{file_id}.png")
        image_save_path = os.path.join(images_path, f"{file_id}.png")

        image = Image.open(image_path).convert("RGB").crop((0, 0, 320, 320))
        image.save(image_save_path)

        label_path = os.path.join(
            cad120_path, "segmentation_mat", f"{file_id}_binary_multilabel.mat"
        )
        label_save_path = os.path.join(labels_path, f"{file_id}.pkl")

        label = loadmat(label_path)["data"].astype(np.uint8)
        label = [
            Image.fromarray(label[:, :, i], mode="L").crop((0, 0, 320, 320))
            for i in range(label.shape[2])
        ]
        label = np.stack(label, axis=2)
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
    source_path = "../../../dataset/CAD120"
    output_path = os.path.join("../../../dataset/cad120", split_mode)

    # if os.path.exists(output_path):
    #     shutil.rmtree(output_path)

    split_dataset(source_path, split_mode)
    # gen_dataset(source_path, output_path, split_mode)
    gen_keypoint_list(source_path, output_path, split_mode)

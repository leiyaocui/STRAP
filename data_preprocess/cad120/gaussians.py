import os
import numpy as np
from scipy.io import loadmat
import pickle
import shutil
from tqdm import tqdm


def get_llkh(allrc, btemp, crop_size):
    meanrc = np.asarray(btemp[0:2])
    covrc = np.asarray([[btemp[2], btemp[4]], [btemp[4], btemp[3]]])
    diff = allrc - meanrc
    expt = -0.5 * np.sum((diff @ np.linalg.pinv(covrc) * diff), axis=1)
    const = 1 / (np.sqrt(np.linalg.det(covrc)) * 2 * np.pi)
    llkh = (const * np.exp(expt)).reshape((crop_size, crop_size))
    return llkh


def gaussians(keypoints, file_path, save_path):
    crop_size = 321
    gaussian_size = 10

    file_name = os.path.basename(file_path)
    image_id = int(file_name[1:5])
    bb_id = int(file_name[6])

    input = loadmat(file_path)["data"]
    data = np.zeros((input.shape[2], crop_size, crop_size))

    crop_range = np.arange(1, crop_size + 1).reshape(1, -1)
    allr = np.tile(crop_range, (1, crop_size)).reshape(-1, 1)
    allc = np.tile(crop_range.T, (1, crop_size)).reshape(-1, 1)
    allrc = np.concatenate([allr, allc], axis=1)

    for i in range(input.shape[2]):
        if np.max(np.max(input[:, :, i])) > 0:
            y = keypoints[keypoints[:, 0].astype(np.int32) == image_id, :]
            y = y[y[:, 1].astype(np.int32) == bb_id, :]
            y = y[y[:, 2].astype(np.int32) == i + 1, :]

            for j in range(y.shape[0]):
                btemp = [y[j, 4], y[j, 3], gaussian_size ** 2, gaussian_size ** 2, 0]
                llkh = get_llkh(allrc, btemp, crop_size)
                llkh = llkh > (np.max(np.max(llkh)) / np.e)
                data[i, :, :] = np.maximum(data[i, :, :], llkh.astype(np.uint8))
    
    data = data.astype(np.uint8)
    with open(save_path, "wb") as f: 
        pickle.dump(data, f)


def train_set(cad120_path, save_path, split_mode):
    keypoints = np.loadtxt("keypoints.txt", delimiter=",")

    os.makedirs(save_path, exist_ok=True)

    images_path = os.path.join(save_path, "affordance", "images")
    labels_path = os.path.join(save_path, "affordance", "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    fb = open(os.path.join(save_path, "train_affordance.txt"), "w")
    for line in tqdm(open(f"train_{split_mode}_split_id.txt", "r"), ncols=80):
        line = line.strip()

        image_path = os.path.join(cad120_path, "object_crop_images", line + ".png")
        image_save_path = os.path.join(images_path, line + ".png")
        label_path = os.path.join(cad120_path, "segmentation_mat", line + "_binary_multilabel.mat")
        label_save_path = os.path.join(labels_path, line + ".pkl")
        
        shutil.copyfile(image_path, image_save_path)
        gaussians(keypoints, label_path, label_save_path)

        fb.write(os.path.relpath(image_save_path, save_path) + "," + os.path.relpath(label_save_path, save_path) + "\n")
    fb.close()


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
        label_path = os.path.join(cad120_path, "segmentation_mat", line + "_binary_multilabel.mat")
        label_save_path = os.path.join(labels_path, line + ".pkl")

        assert not os.path.exists(image_save_path) and not os.path.exists(label_save_path)
        
        shutil.copyfile(image_path, image_save_path)
        label = loadmat(label_path)["data"]
        label = (label > 0).astype(np.uint8).transpose((2, 0, 1))
        with open(label_save_path, "wb") as f: 
            pickle.dump(label, f)

        fb.write(os.path.relpath(image_save_path, save_path) + "," + os.path.relpath(label_save_path, save_path) + "\n")
    fb.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    split_mode = "object"

    source_path = "/home/DISCOVER_summer2022/cuily/dataset/CAD120"
    output_path = os.path.join("/home/DISCOVER_summer2022/cuily/dataset/cad120", split_mode)
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    train_set(source_path, output_path, split_mode)
    val_set(source_path, output_path, split_mode)


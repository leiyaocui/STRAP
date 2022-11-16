import os
from PIL import Image
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    dataset_path = "../dataset/umd/novel"
    fb = open(os.path.join(dataset_path, "train_affordance.txt"), "r")
    image_path_list = []
    for line in fb:
        line = line.strip()
        image_path = os.path.join(dataset_path, line.split(",")[0])
        image_path_list.append(image_path)

    c_sum = np.zeros(3)
    c_num = 0
    for path in tqdm(image_path_list, ncols=80):
        img = np.array(Image.open(path))
        h, w, c = img.shape
        c_sum += img.sum(axis=(0, 1))
        c_num += h * w

    c_mean = c_sum / c_num
    print(f"Mean(RGB order): [{c_mean[0]}, {c_mean[1]}, {c_mean[2]}]")

    c_mean = c_mean.reshape(1, 1, 3)
    c_sum = np.zeros(3)
    c_num = 0
    for path in tqdm(image_path_list, ncols=80):
        img = np.array(Image.open(path))
        h, w, c = img.shape
        c_sum += np.sum((img - c_mean) ** 2, axis=(0, 1))
        c_num += h * w

    c_std = np.sqrt(c_sum / c_num)
    print(f"Std(RGB order): [{c_std[0]}, {c_std[1]}, {c_std[2]}]")
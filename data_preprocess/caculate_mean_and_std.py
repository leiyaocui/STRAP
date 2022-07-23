import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    dataset_path = "../dataset/cad120/object/affordance/images"
    image_path_list = glob.glob(os.path.join(dataset_path, "*.png"))
    assert len(image_path_list) > 0

    c_sum = np.zeros(3)
    c_num = 0
    for path in tqdm(image_path_list, ncols=80):
        img = np.array(Image.open(path).convert("RGB"))
        h, w, c = img.shape
        c_sum += img.sum(axis=(0, 1))
        c_num += h * w

    c_mean = c_sum / c_num
    print(f"Mean(RGB order): [{c_mean[0]}, {c_mean[1]}, {c_mean[2]}]")

    c_mean = c_mean.reshape(1, 1, 3)
    c_sum = np.zeros(3)
    c_num = 0
    for path in tqdm(image_path_list, ncols=80):
        img = np.array(Image.open(path).convert("RGB"))
        h, w, c = img.shape
        c_sum += np.sum((img - c_mean) ** 2, axis=(0, 1))
        c_num += h * w

    c_std = np.sqrt(c_sum / (c_num - 1))
    print(f"Std(RGB order): [{c_std[0]}, {c_std[1]}, {c_std[2]}]")

# dataset_mean: [132.47758920907586, 106.32022472065732, 111.5704799224029]
# dataset_std: [67.45043019809088, 70.23484330785524, 72.19806953380163]

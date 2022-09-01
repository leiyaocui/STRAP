import os
import numpy as np
from tqdm import tqdm
import yaml

if __name__ == "__main__":
    dataset_path = "../dataset/cad120/object"
    fb = open(os.path.join(dataset_path, "train_affordance.txt"), "r")
    file_id_list = []
    for line in fb:
        line = line.strip()
        file_id = os.path.basename(line.split(",")[0]).split(".")[0]
        file_id_list.append(file_id)

    keypoint_path = "../dataset/cad120/object/train_affordance_keypoint.yaml"
    with open(keypoint_path, "r") as fb:
        keypoint_dict = yaml.safe_load(fb)

    class_list = [
        "openable",
        "cuttable",
        "pourable",
        "containable",
        "supportable",
        "holdable",
    ]
    num_class = len(class_list)

    class_hist = np.zeros((num_class), dtype=np.int64)
    for file_id in tqdm(file_id_list, ncols=80):
        keypoint = keypoint_dict[file_id]
        for k in keypoint.keys():
            class_hist[k] += 1
    print(f"Class Hist: {class_hist.tolist()}")

    class_hist_ratio = class_hist / len(file_id_list)
    print(f"Class Hist Ratio: {class_hist_ratio.tolist()}")

    class_weight = 1 / class_hist_ratio
    class_weight = class_weight / class_weight.sum()
    print(f"Class Weight: {np.round(class_weight, 6).tolist()}")

# class_weight: [0.170589, 0.44999, 0.121701, 0.090404, 0.094336, 0.07298]

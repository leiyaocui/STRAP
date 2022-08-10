import os
import yaml
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    task = [
        "openable",
        "cuttable",
        "pourable",
        "containable",
        "supportable",
        "holdable",
    ]

    data_dir = "../dataset/cad120/object"

    keypoint_path = os.path.join(data_dir, "train_affordance_keypoint.yaml")
    with open(keypoint_path, "r") as fb:
        keypoint_dict = yaml.safe_load(fb)

    pos_hist = np.zeros((len(task)))
    neg_hist = np.zeros((len(task)))

    for line in tqdm(
        open(os.path.join(data_dir, "train_affordance.txt"), "r"), ncols=80
    ):
        image_path, _ = line.strip().split(",")
        file_name = os.path.basename(image_path).split(".")[0]
        keypoint = keypoint_dict[file_name]
        
        for i in range(len(task)):
            if i in keypoint:
                pos_hist[i] += len(keypoint[i])
            else:
                neg_hist[i] += 320 * 320

    pos_neg_ratio = pos_hist / neg_hist

    print("pos_hist:")
    print(pos_hist.tolist())
    print("neg_hist:")
    print(neg_hist.tolist())

    print("pos_neg_ratio")
    print(pos_neg_ratio.tolist())

# pos_hist:
# [3504.0, 944.0, 4832.0, 8147.0, 6981.0, 13483.0]
# neg_hist:
# [478924800.0, 672972800.0, 353382400.0, 201728000.0, 226304000.0, 60928000.0]
# pos_neg_ratio
# [7.316388710711995e-06, 1.4027312842361533e-06, 1.3673572877426833e-05, 4.03860644035533e-05, 3.0847886029411764e-05, 0.00022129398634453782]
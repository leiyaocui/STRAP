import os
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F

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

    class_hist = [0.0] * len(task)

    for line in tqdm(
        open(os.path.join(data_dir, "train_affordance.txt"), "r"), ncols=80
    ):
        image_path, _ = line.strip().split(",")
        file_name = os.path.basename(image_path).split(".")[0]

        for cls_id, joints in keypoint_dict[file_name]:
            class_hist[int(cls_id)] += len(joints)

    print("class_hist:")
    print(class_hist)

    hist = torch.tensor(class_hist).double()
    hist = F.normalize(hist, p=1, dim=0)

    # classes_weight = 1.0 / hist
    # class_weight = 1.0 / torch.log(hist)
    class_weight = torch.exp(-hist)

    class_weight = F.normalize(class_weight, p=1, dim=0)
    class_weight = class_weight.float().numpy()

    print("class_weight:")
    print(class_weight)

# class_hist:   [1452.0,     600.0,      4410.0,     6154.0,     5549.0,     9439.0]
# class_weight: [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667] (reference)
# class_weight: [0.22723687, 0.54991320, 0.07481813, 0.05361520, 0.05946079, 0.03495581] (normal)
# class_weight: [0.23328905, 0.28841156, 0.14347772, 0.11838961, 0.12590544, 0.09052663] (log)
# class_weight: [0.18573777, 0.19155999, 0.16686374, 0.15664753, 0.16011870, 0.13907227] (softmax)

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

    classes_hist = [0.0] * len(task)

    for line in tqdm(
        open(os.path.join(data_dir, "train_affordance.txt"), "r"), ncols=80
    ):
        image_path, _ = line.strip().split(",")
        file_name = os.path.basename(image_path).split(".")[0]

        for cls_id, joints in keypoint_dict[file_name]:
            classes_hist[int(cls_id)] += len(joints)

    print("classes_hist:")
    print(classes_hist)

    hist = torch.tensor(classes_hist).double()
    hist = F.normalize(hist, p=1, dim=0)

    use_logp = False

    if use_logp:
        classes_weight = 1.0 / torch.log(hist)
    else:
        classes_weight = 1.0 / hist

    classes_weight = F.normalize(classes_weight, p=1, dim=0).float().numpy()

    print("classes_weight:")
    print(classes_weight)

# classes_hist: [1452.0, 600.0, 4410.0, 6154.0, 5549.0, 9439.0]
# classes_weight: [0.23328905 0.28841156 0.14347772 0.11838961 0.12590544 0.09052663] (use log)
# classes_weight: [0.22723687 0.5499132  0.07481813 0.0536152  0.05946079 0.03495581] (not use log)
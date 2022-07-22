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

    use_logp = True

    if use_logp:
        # https://arxiv.org/pdf/1809.09077v1.pdf formula (3)
        log_safety_const = 1.1
        classes_weight = 1.0 / torch.log(hist + log_safety_const)
    else:
        classes_weight = 1.0 / hist.clamp(min=1e-6)

    classes_weight = F.normalize(classes_weight, p=1, dim=0).float().numpy()

    print("classes_weight:")
    print(classes_weight)

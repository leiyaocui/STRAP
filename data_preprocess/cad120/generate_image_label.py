import os
import numpy as np
from tqdm import tqdm
import yaml

def gen_image_label(save_path):
    image_label_dict = {}

    fb = open("visible_affordance_info.txt", "r")
    for line in tqdm(fb, ncols=80):
        line = line.strip().split(" ")
        file_name = line[0].split(".")[0]
        label = np.array(line[1:], dtype=np.uint8).tolist()
        assert len(label) == 6

        image_label_dict[file_name] = label
    fb.close()

    with open(os.path.join(save_path, "train_affordance_image_label.yaml"), "w") as fb:
        yaml.safe_dump(image_label_dict, fb)

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    split_mode = "object"
    output_path = os.path.join("../../../dataset/cad120", split_mode)

    gen_image_label(output_path)

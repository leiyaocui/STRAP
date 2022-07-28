import os
import yaml
from PIL import Image
import pickle
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_dir, phase, transforms, label_level=["dense", "weak", "point"]):
        self.transforms = transforms
        self.label_level = label_level

        self.file_name_list = []
        self.image_list = []

        if "dense" in self.label_level:
            self.label_list = []

        if "weak" in self.label_level:
            self.weak_label_list = []
            weak_label_save_dir = os.path.join(data_dir, phase + "_weak_label")
            os.makedirs(weak_label_save_dir, exist_ok=True)

        if "point" in self.label_level:
            self.point_label_list = []
            point_label_path = os.path.join(data_dir, phase + "_keypoint.yaml")
            with open(point_label_path, "r") as fb:
                point_label_dict = yaml.safe_load(fb)

        for line in open(os.path.join(data_dir, phase + ".txt"), "r"):
            image_path, label_path = line.strip().split(",")

            file_name = os.path.basename(image_path).split(".")[0]
            self.file_name_list.append(file_name)

            self.image_list.append(os.path.join(data_dir, image_path))

            if "dense" in self.label_level:
                self.label_list.append(os.path.join(data_dir, label_path))

            if "weak" in self.label_level:
                self.weak_label_list.append(
                    os.path.join(weak_label_save_dir, file_name + ".pkl")
                )

            if "point" in self.label_level:
                self.point_label_list.append(point_label_dict[file_name])

    def __getitem__(self, index):
        data = dict()

        data["file_name"] = self.file_name_list[index]

        data["image"] = Image.open(self.image_list[index])

        if "dense" in self.label_level:
            with open(self.label_list[index], "rb") as fb:
                dense_label = pickle.load(fb)
                data["dense_label"] = [
                    Image.fromarray(dense_label[:, :, i], mode="L")
                    for i in range(dense_label.shape[2])
                ]

        if "weak" in self.label_level:
            with open(self.weak_label_list[index], "rb") as fb:
                weak_label = pickle.load(fb)
                data["weak_label"] = [
                    Image.fromarray(weak_label[:, :, i], mode="L")
                    for i in range(weak_label.shape[2])
                ]

        if "point" in self.label_level:
            data["point_label"] = self.point_label_list[index]

        data["validity"] = Image.new("L", data["image"].size, color=1)

        data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.image_list)


def make_dataloader(
    data_dir, phase, transforms, label_level, **kargs
):
    dataset = CustomDataset(data_dir, phase, transforms, label_level)

    return DataLoader(dataset, **kargs)

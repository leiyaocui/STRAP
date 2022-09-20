import numpy as np
from PIL import Image, ImageDraw
import torch


class Identity:
    def __call__(self, data):
        return data


class RandomHorizonalFlipPIL:
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, data):
        if self.rng.uniform(low=0, high=1) < 0.5:
            data["image"] = data["image"].transpose(Image.FLIP_LEFT_RIGHT)

            for k in ["dense_label", "pseudo_label", "weak_label"]:
                if k in data:
                    label = data[k]
                    data[k] = [it.transpose(Image.FLIP_LEFT_RIGHT) for it in label]

            if "point_label" in data:
                label = data["point_label"]
                new_label = {}
                w, h = data["image"].size
                for cls_id, joints in label.items():
                    new_joints = []
                    for it in joints:
                        new_joints.append([w - 1 - it[0], it[1]])
                    new_label[cls_id] = new_joints
                data["point_label"] = new_label

        return data


class RandomVerticalFlipPIL:
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, data):
        if self.rng.uniform(low=0, high=1) < 0.5:
            data["image"] = data["image"].transpose(Image.FLIP_TOP_BOTTOM)

            for k in ["dense_label", "pseudo_label", "weak_label"]:
                if k in data:
                    label = data[k]
                    data[k] = [it.transpose(Image.FLIP_TOP_BOTTOM) for it in label]

            if "point_label" in data:
                label = data["point_label"]
                new_label = {}
                w, h = data["image"].size
                for cls_id, joints in label.items():
                    new_joints = []
                    for it in joints:
                        new_joints.append([it[0], h - 1 - it[1]])
                    new_label[cls_id] = new_joints
                data["point_label"] = new_label

        return data


class ConvertPointLabel:
    def __init__(self, num_class, ignore_index=255):
        self.num_class = num_class
        self.ignore_index = ignore_index

    def __call__(self, data):
        image_size = data["image"].size

        weak_label = []
        visible_info = []
        for i in range(self.num_class):
            if i in data["point_label"]:
                visible_info.append(1)

                joints = data["point_label"][i]
                label = Image.new("L", image_size, color=self.ignore_index)
                draw = ImageDraw.Draw(label)
                for i in range(len(joints)):
                    draw.point([joints[i][0], joints[i][1]], fill=1)
            else:
                visible_info.append(0)

                label = Image.new("L", image_size, color=0)

            weak_label.append(label)

        data["weak_label"] = weak_label
        data["visible_info"] = visible_info

        return data


class PILToTensor:
    def __call__(self, data):
        for k in ["point_label"]:
            if k in data:
                del data[k]

        for k in data:
            if k in ["file_name"]:
                continue
            elif k == "image":
                data[k] = (
                    torch.from_numpy(np.array(data[k]))
                    .permute(2, 0, 1)
                    .contiguous()
                    .float()
                )
            elif k in [
                "dense_label",
                "weak_label",
                "pseudo_label",
                "visible_info",
                "mask_dst",
            ]:
                label = data[k]
                data[k] = [
                    torch.from_numpy(np.array(label[i])).long()
                    for i in range(len(label))
                ]
            else:
                raise ValueError("Not support data's key: {k}")

        return data


class ImageNormalizeTensor:
    def __init__(self, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, data):
        assert torch.is_tensor(data["image"])
        data["image"] = (data["image"] - self.mean) / self.std

        return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for tf in self.transforms:
            data = tf(data)

        return data

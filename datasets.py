import os
import torch
from PIL import Image
import numpy as np


class ConcatSegList(torch.utils.data.Dataset):
    def __init__(self, at, af, seg):
        self.at = at
        self.af = af
        self.seg = seg

        assert len(self.at) == len(self.af) and len(self.af) == len(self.seg)

    def __getitem__(self, index):
        return (self.at[index], self.af[index], self.seg[index])

    def __len__(self):
        return len(self.at)


class SegMultiHeadList(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, phase, transforms, ms_scale=None,
    ):
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.ms_scale = ms_scale

        self.image_list = None
        self.label_list = None

        image_path = os.path.join(self.data_dir, self.phase + "_images.txt")
        label_path = os.path.join(self.data_dir, self.phase + "_labels.txt")

        assert os.path.exists(image_path)
        self.image_list = [
            os.path.join(self.data_dir, line.strip()) for line in open(image_path, "r")
        ]
        if os.path.exists(label_path):
            self.label_list = [
                os.path.join(self.data_dir, line.strip())
                for line in open(label_path, "r")
            ]
            assert len(self.image_list) == len(self.label_list)

    def __getitem__(self, index):
        data = Image.open(self.image_list[index])
        if len(data.shape) == 2:
            data = np.stack([data, data, data], axis=2)
        data = [Image.fromarray(data)]

        if self.label_list is not None:
            label_data = [Image.open(i) for i in self.label_list[index].split(",")]
            data.append(label_data)

        data = list(self.transforms(*data))

        if self.ms_scale is not None:
            w, h = (640, 480)
            # 这里为什么是self.transforms()[0]需要检查一下
            ms_images = [
                self.transforms(
                    data[0].resize(
                        (round(int(w * s) / 32) * 32, round(int(h * s) / 32) * 32),
                        Image.BICUBIC,
                    )
                )[0]
                for s in self.ms_scale
            ]
            data.append(self.image_list[index])
            data.extend(ms_images)

        return tuple(data)

    def __len__(self):
        return len(self.image_list)

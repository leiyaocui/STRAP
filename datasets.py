import os
import torch
from PIL import Image
import numpy as np
import pickle


class ConcatMultiHeadDataset(torch.utils.data.Dataset):
    def __init__(self, at, af, seg):
        self.at = at
        self.af = af
        self.seg = seg

        assert len(self.at) == len(self.af) and len(self.af) == len(self.seg)

    def __getitem__(self, index):
        return self.at[index], self.af[index], self.seg[index]

    def __len__(self):
        return len(self.at)


class MultiHeadDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, ms_scale=None, out_name=False):
        self.transforms = transforms
        self.ms_scale = ms_scale
        self.out_name = out_name

        self.image_list = None
        self.label_list = None

        image_path = os.path.join(data_dir, phase + "_images.txt")
        label_path = os.path.join(data_dir, phase + "_labels.txt")

        assert os.path.exists(image_path)
        self.image_list = [
            os.path.join(data_dir, line.strip()) for line in open(image_path, "r")
        ]
        if os.path.exists(label_path):
            self.label_list = [
                [
                    os.path.join(data_dir, line_split.strip())
                    for line_split in line.strip().split(",")
                ]
                for line in open(label_path, "r")
            ]
            assert len(self.image_list) == len(self.label_list)

    def __getitem__(self, index):
        data = []

        image = Image.open(self.image_list[index])
        if len(image.getbands()) == 1:
            image = np.asarray(image)
            image = np.stack([image, image, image], axis=2)
            image = Image.fromarray(image)
        data.append(image)

        if self.label_list is not None:
            data.append([Image.open(it) for it in self.label_list[index]])

        if self.ms_scale is not None:
            w, h = (640, 480)
            ms_data = [
                self.transforms(
                    data[0].resize(
                        (round(int(w * s) / 32) * 32, round(int(h * s) / 32) * 32),
                        Image.BICUBIC,
                    )
                )[0]
                for s in self.ms_scale
            ]

        assert data[0].size == data[1][0].size
        data = list(self.transforms(*data))

        if self.ms_scale is not None:
            data.append(ms_data)

        if self.out_name:
            data.append(self.image_list[index])

        return tuple(data)

    def __len__(self):
        return len(self.image_list)


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, out_name=False):
        self.transforms = transforms
        self.out_name = out_name

        self.image_list = []
        self.label_list = []

        data_path = os.path.join(data_dir, phase + ".txt")

        for line in open(data_path, "r"):
            image_path, label_path = line.strip().split(",")
            self.image_list.append(os.path.join(data_dir, image_path))
            self.label_list.append(os.path.join(data_dir, label_path))

        assert len(self.image_list) == len(self.label_list)

    def __getitem__(self, index):
        data = []

        image = Image.open(self.image_list[index])
        if len(image.getbands()) == 1:
            image = np.asarray(image)
            image = np.stack([image, image, image], axis=2)
            image = Image.fromarray(image)
        data.append(image)

        with open(self.label_list[index], "rb") as fb:
            label_pkl = pickle.load(fb)

        data.append([Image.fromarray(label_pkl[:, :, i]) for i in range(label_pkl.shape[2])])

        data = list(self.transforms(*data))

        if self.out_name:
            data.append(self.image_list[index])

        return tuple(data)

    def __len__(self):
        return len(self.image_list)
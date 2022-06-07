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
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False, ms=False, scale=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.ms = ms
        self.scale = scale
        assert not(self.out_name and self.ms)
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(os.path.join(
            self.data_dir, self.image_list[index]))]
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data, data, data], axis=2)
        data = [Image.fromarray(data)]

        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):
                label_data.append(Image.open(os.path.join(self.data_dir, it)))
            data.append(label_data)
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        if self.ms:
            w, h = (640, 480)
            ms_images = [self.transforms(data[0].resize((round(int(w * s)/32) * 32, round(int(h * s)/32) * 32),
                                                        Image.BICUBIC))[0]
                         for s in self.scales]
            data.append(self.image_list[index])
            data.extend(ms_images)

        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    @torch.jit.ignore
    def read_lists(self):
        image_path = os.path.join(self.list_dir, self.phase + '_images.txt')
        label_path = os.path.join(self.list_dir, self.phase + '_labels.txt')
        assert os.path.exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if os.path.exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
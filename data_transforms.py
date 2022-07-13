import numbers
import random
import numpy as np
from PIL import Image
import torch


class RandomCropMultiHead:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image, label=None):
        w, h = image.size
        tw, th = self.size
        top = bottom = left = right = 0
        if w < tw:
            left = (tw - w) // 2
            right = tw - w - left
        if h < th:
            top = (th - h) // 2
            bottom = th - h - top
        if left > 0 or right > 0 or top > 0 or bottom > 0:
            image = pad_image("reflection", image, top, bottom, left, right)
            for i in range(len(label)):
                label[i] = pad_image(
                    "constant", label[i], top, bottom, left, right, value=255
                )

        w, h = image.size
        if w == tw and h == th:
            return image, label

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        image = image.crop((x1, y1, x1 + tw, y1 + th))
        if label is not None:
            for i in range(len(label)):
                label[i] = label[i].crop((x1, y1, x1 + tw, y1 + th))

        return image, label


class RandomScaleMultiHead:
    def __init__(self, scale):
        if isinstance(scale, numbers.Number):
            scale = [1 / scale, scale]
        self.scale = scale

    def __call__(self, image, label=None):
        ratio = random.uniform(self.scale[0], self.scale[1])
        w, h = image.size
        tw = int(ratio * w)
        th = int(ratio * h)
        if ratio == 1:
            return image, label
        elif ratio < 1:
            interpolation = Image.ANTIALIAS
        else:
            interpolation = Image.CUBIC

        image = image.resize((tw, th), interpolation)
        if label is not None:
            for i in range(len(label)):
                label[i] = label[i].resize((tw, th), Image.NEAREST)

        return image, label


class RandomRotateMultiHead:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label=None):
        w, h = image.size
        angle = random.randint(0, self.angle * 2) - self.angle

        image = pad_image("reflection", image, h, h, w, w)
        image = image.rotate(angle, resample=Image.BILINEAR)
        image = image.crop((w, h, w + w, h + h))
        if label is not None:
            for i in range(len(label)):
                label[i] = pad_image("constant", label[i], h, h, w, w, value=255)
                label[i] = label[i].rotate(angle, resample=Image.NEAREST)
                label[i] = label[i].crop((w, h, w + w, h + h))

        return image, label


class RandomHorizontalFlipMultiHead:
    def __call__(self, image, label=None):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if label is not None:
                for i in range(len(label)):
                    label[i] = label[i].transpose(Image.FLIP_LEFT_RIGHT)

        return image, label


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label=None):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)

        return image, label


def pad_reflection(image, top, bottom, left, right):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    next_top = next_bottom = next_left = next_right = 0
    if top > h - 1:
        next_top = top - h + 1
        top = h - 1
    if bottom > h - 1:
        next_bottom = bottom - h + 1
        bottom = h - 1
    if left > w - 1:
        next_left = left - w + 1
        left = w - 1
    if right > w - 1:
        next_right = right - w + 1
        right = w - 1
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image[top : top + h, left : left + w] = image
    new_image[:top, left : left + w] = image[top:0:-1, :]
    new_image[top + h :, left : left + w] = image[-1 : -bottom - 1 : -1, :]
    new_image[:, :left] = new_image[:, left * 2 : left : -1]
    new_image[:, left + w :] = new_image[:, -right - 1 : -right * 2 - 1 : -1]
    return pad_reflection(new_image, next_top, next_bottom, next_left, next_right)


def pad_constant(image, top, bottom, left, right, value):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top : top + h, left : left + w] = image
    return new_image


def pad_image(mode, image, top, bottom, left, right, value=0):
    if mode == "reflection":
        return Image.fromarray(
            pad_reflection(np.asarray(image), top, bottom, left, right)
        )
    elif mode == "constant":
        return Image.fromarray(
            pad_constant(np.asarray(image), top, bottom, left, right, value)
        )
    else:
        raise ValueError("Unknown mode {}".format(mode))


class ToTensorMultiHead:
    def __call__(self, image, label=None):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        else:
            image = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).contiguous()

        image = image.float().div(255)
        if label is not None:
            for i in range(len(label)):
                label[i] = torch.LongTensor(np.asarray(label[i], dtype=np.int32))

        return image, label


class MaskLabelMultiHead:
    def __init__(self, filled_value=255):
        self.filled_value = filled_value

    def __call__(self, image, label):
        assert label is not None
        mask = (sum(label) < 1).bool()
        for i in range(len(label)):
            label[i].masked_fill_(mask, self.filled_value)
        
        return image, label


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for tf in self.transforms:
            args = tf(*args)

        return args

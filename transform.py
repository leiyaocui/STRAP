import numpy as np
from PIL import Image, ImageDraw
import torch
import pickle
import pydensecrf.densecrf as dcrf


class Identity:
    def __call__(self, data):
        return data


class RandomScaledTiltedWarpedPIL:
    def __init__(
        self,
        random_crop_size=(0, 0),
        random_shrink_min=0.5,
        random_shrink_max=2.0,
        random_tilt_max_deg=10,
        random_wiggle_max_ratio=0,
        random_horizon_reflect=True,
        center_offset_instead_of_random=False,
        ignore_index=255,
    ):
        assert random_shrink_min > 0, "random_shrink_min must be positive"
        assert (
            random_shrink_max >= random_shrink_min
        ), "random_shrink_max > random_shrink_min"
        assert random_tilt_max_deg >= 0, "tilt must be non negative"
        assert (
            0 <= random_wiggle_max_ratio < 0.5
        ), "random_wiggle_max_ratio must be [0, 0.5)"

        self.dst_size = tuple(random_crop_size)
        self.random_shrink_min = random_shrink_min
        self.random_shrink_max = random_shrink_max
        self.random_tilt_max_deg = random_tilt_max_deg
        self.random_wiggle_max_ratio = random_wiggle_max_ratio
        self.random_horizon_reflect = random_horizon_reflect
        self.center_offset_instead_of_random = center_offset_instead_of_random
        self.ignore_index = ignore_index

    def __call__(self, data):
        dst_corners = [
            np.array([0, 0], dtype=np.float32),
            np.array([0, self.dst_size[1]], dtype=np.float32),
            np.array([self.dst_size[0], self.dst_size[1]], dtype=np.float32),
            np.array([self.dst_size[0], 0], dtype=np.float32),
        ]

        if self.random_horizon_reflect:
            if np.random.random() < 0.5:
                dst_corners = list(reversed(dst_corners))

        src_corners, src_scale = self.generate_corners(data["image"].size)

        warp_coef_inv = self.perspective_transform_from_corners(
            dst_corners, src_corners
        )
        warp_coef_fwd = self.perspective_transform_from_corners(
            src_corners, dst_corners
        )
        warp_coef_fwd = np.append(warp_coef_fwd, 1).reshape((3, 3))

        for k in data:
            if k == "file_name":
                continue
            elif k == "image":
                data[k] = data[k].transform(
                    self.dst_size,
                    Image.PERSPECTIVE,
                    warp_coef_inv,
                    Image.BILINEAR,
                    fillcolor=None,
                )
            elif k in ["dense_label", "weak_label"] and k in data:
                label = data[k]
                data[k] = [
                    label[i].transform(
                        self.dst_size,
                        Image.PERSPECTIVE,
                        warp_coef_inv,
                        Image.NEAREST,
                        fillcolor=self.ignore_index,
                    )
                    for i in range(len(label))
                ]
            elif k == "point_label":
                new_label = []
                for cls_id, joints in data[k]:
                    new_joints = []
                    for pt in joints:
                        pt = np.array([pt[0], pt[1], 1.0], np.float32)
                        pt_new = np.matmul(warp_coef_fwd, pt)
                        pt_x_new = pt_new[0] / pt_new[2]
                        pt_y_new = pt_new[1] / pt_new[2]
                        new_joints.append([pt_x_new, pt_y_new])
                    new_label.append((cls_id, new_joints))
                data[k] = new_label
            elif k == "validity":
                data[k] = data[k].transform(
                    self.dst_size,
                    Image.PERSPECTIVE,
                    warp_coef_inv,
                    Image.NEAREST,
                    fillcolor=0,
                )
            else:
                raise ValueError("Not support data's key: {k}")

        return data

    def perspective_transform_from_corners(self, corners_src, corners_dst):
        matrix = []
        for p_src, p_dst in zip(corners_src, corners_dst):
            matrix.append(
                [
                    p_src[0],
                    p_src[1],
                    1,
                    0,
                    0,
                    0,
                    -p_dst[0] * p_src[0],
                    -p_dst[0] * p_src[1],
                ]
            )
            matrix.append(
                [
                    0,
                    0,
                    0,
                    p_src[0],
                    p_src[1],
                    1,
                    -p_dst[1] * p_src[0],
                    -p_dst[1] * p_src[1],
                ]
            )
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(corners_dst).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)

        return np.array(res, dtype=np.float32).reshape(8)

    def dst_corners_bounding_box(self, corners):
        x_min, x_max = corners[0][0], corners[0][0]
        y_min, y_max = corners[0][1], corners[0][1]
        for corner in corners[1:]:
            x_min = min(x_min, corner[0])
            x_max = max(x_max, corner[0])
            y_min = min(y_min, corner[1])
            y_max = max(y_max, corner[1])

        return x_min, x_max, y_min, y_max

    def transform_scale_rotate_wiggle(self):
        corners = [
            np.array([-self.dst_size[0] / 2, -self.dst_size[1] / 2], dtype=np.float32),
            np.array([-self.dst_size[0] / 2, self.dst_size[1] / 2], dtype=np.float32),
            np.array([self.dst_size[0] / 2, self.dst_size[1] / 2], dtype=np.float32),
            np.array([self.dst_size[0] / 2, -self.dst_size[1] / 2], dtype=np.float32),
        ]

        max_wiggle_pix = (
            self.random_wiggle_max_ratio * min(self.dst_size[0], self.dst_size[1]) / 2
        )
        scale = np.random.uniform(self.random_shrink_min, self.random_shrink_max)
        angle_deg = (
            np.random.uniform(-self.random_tilt_max_deg, self.random_tilt_max_deg)
            if 0 < self.random_tilt_max_deg <= 45
            else 0
        )
        wiggle_factor = [
            np.array(
                [
                    np.random.uniform(-max_wiggle_pix, max_wiggle_pix),
                    np.random.uniform(-max_wiggle_pix, max_wiggle_pix),
                ],
                dtype=np.float32,
            )
            for _ in range(4)
        ]

        angle_rad = np.deg2rad(angle_deg)
        matrix_rot = np.array(
            [
                [np.cos(angle_rad), np.sin(-angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ],
            dtype=np.float32,
        )

        corners = [
            np.matmul(matrix_rot, scale * (c + w))
            for c, w in zip(corners, wiggle_factor)
        ]

        return corners, scale

    def generate_corners(self, src_size):
        corners, scale = self.transform_scale_rotate_wiggle()
        x_min, x_max, y_min, y_max = self.dst_corners_bounding_box(corners)

        range_x_min = -x_min
        range_x_max = src_size[0] - x_max
        range_y_min = -y_min
        range_y_max = src_size[1] - y_max

        if self.center_offset_instead_of_random or range_x_max <= range_x_min:
            offs_x = (range_x_min + range_x_max) * 0.5
        else:
            offs_x = np.random.uniform(range_x_min, range_x_max)

        if self.center_offset_instead_of_random or range_y_max <= range_y_min:
            offs_y = (range_y_min + range_y_max) * 0.5
        else:
            offs_y = np.random.uniform(range_y_min, range_y_max)

        corners = [c + np.array([offs_x, offs_y], dtype=np.float32) for c in corners]

        return corners, scale


class PILToTensor:
    def __call__(self, data):
        for k in data:
            if k in ["file_name", "point_label"]:
                continue
            elif k == "image":
                data[k] = (
                    torch.from_numpy(np.array(data[k]))
                    .permute(2, 0, 1)
                    .contiguous()
                    .float()
                )
            elif k in ["dense_label", "weak_label"] and k in data:
                label = data[k]
                data[k] = [
                    torch.from_numpy(np.array(label[i])).long()
                    for i in range(len(label))
                ]
            elif k == "validity" and k in data:
                data[k] = torch.from_numpy(np.array(data[k])).unsqueeze(0).float()
            else:
                raise ValueError("Not support data's key: {k}")

        return data


class ImageToTensorWithNumpy:
    def __call__(self, data):
        data["image_numpy"] = np.array(data["image"], dtype=np.uint8)
        data["image"] = (
            torch.from_numpy(data["image_numpy"]).permute(2, 0, 1).contiguous().float()
        )
        return data


class ImageNormalizeTensor:
    def __init__(self, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, data):
        assert torch.is_tensor(data["image"]), "data.image is not a tensor"
        data["image"] = (data["image"] - self.mean) / self.std

        return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for tf in self.transforms:
            data = tf(data)

        return data


def dense_crf(img, probs, mode):
    n_labels = probs.shape[0]

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    if mode == "softmax":
        U = -np.log(np.clip(probs, 1e-5, 1.0)).reshape(n_labels, -1)
    elif mode == "sigmoid":
        U = -np.log(np.concatenate([1 - probs, probs], axis=0)).reshape(2, -1)
    elif mode == "label":
        probs = probs.flatten()
        gt_prob = 0.8

        U = np.full((n_labels, len(probs)), -np.log(1 - gt_prob))
        U[probs, np.arange(U.shape[1])] = -np.log(gt_prob)

    d.setUnaryEnergy(U.astype(np.float32))

    d.addPairwiseGaussian(
        sxy=5, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    d.addPairwiseBilateral(
        sxy=60,
        srgb=10,
        rgbim=img,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(5)
    Q = np.argmax(Q, axis=0).reshape(img.shape[0], img.shape[1])
    return Q


def generate_weak_label(
    image, save_path, keypoint, stroke_width=20, pred=None, ignore_index=255,
):
    weak_label = []

    if pred is None:
        for _, joints in keypoint:
            if len(joints) > 0:
                label = Image.new(
                    "L", (image.shape[1], image.shape[0]), color=ignore_index
                )
                draw = ImageDraw.Draw(label)

                for i in range(len(joints)):
                    draw.ellipse(
                        (
                            joints[i][0] - stroke_width / 2,
                            joints[i][1] - stroke_width / 2,
                            joints[i][0] + stroke_width / 2,
                            joints[i][1] + stroke_width / 2,
                        ),
                        fill=1,
                    )

                # label = dense_crf(np.array(image), np.array(label), mode="label")
                # label[label==0] = ignore_index
            else:
                label = Image.new("L", (image.shape[1], image.shape[0]), color=0)

            weak_label.append(label)
    else:
        for i, _, joints in enumerate(keypoint):
            if len(joints) > 0:
                label = pred[i]
            else:
                label = Image.new("L", (image.shape[1], image.shape[0]), color=0)

            weak_label.append(label)

    weak_label = np.stack(weak_label, axis=2)
    with open(save_path, "wb") as fb:
        pickle.dump(weak_label, fb)

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

class RandomScaledTiltedWarpedPIL:
    def __init__(
        self,
        random_crop_size=(256, 256), # (width, height)
        random_scale_max=2.0,
        random_scale_min=0.5,
        random_tilt_max_deg=0,
        random_wiggle_max_ratio=0,
        random_horizon_reflect=True,
        center_offset_instead_of_random=False,
        ignore_index=255,
    ):
        assert random_scale_min > 0
        assert random_scale_max >= random_scale_min
        assert random_tilt_max_deg >= 0
        assert 0 <= random_wiggle_max_ratio < 0.5

        self.dst_size = random_crop_size
        self.random_shrink_min = 1.0 / random_scale_max
        self.random_shrink_max = 1.0 / random_scale_min
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
            if np.random.uniform() < 0.5:
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
            if k == "image":
                data[k] = data[k].transform(
                    self.dst_size,
                    Image.Transform.PERSPECTIVE,
                    warp_coef_inv,
                    Image.Resampling.BICUBIC,
                    fillcolor=None,
                )
            elif k in ["dense_label", "weak_label", "pseudo_label"]:
                label = data[k]
                data[k] = [
                    label[i].transform(
                        self.dst_size,
                        Image.Transform.PERSPECTIVE,
                        warp_coef_inv,
                        Image.Resampling.NEAREST,
                        fillcolor=self.ignore_index,
                    )
                    for i in range(len(label))
                ]
            elif k == "point_label":
                new_label = {}
                for cls_id, joints in data[k].items():
                    new_joints = []
                    for pt in joints:
                        pt = np.array([pt[0], pt[1], 1.0], np.float32)
                        pt_new = np.matmul(warp_coef_fwd, pt)
                        pt_x_new = pt_new[0] / pt_new[2]
                        pt_y_new = pt_new[1] / pt_new[2]
                        new_joints.append([pt_x_new, pt_y_new])
                    new_label[cls_id] = new_joints
                data[k] = new_label

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
        data["orig_image"] = data["image"].clone()
        data["image"] = (data["image"] - self.mean) / self.std

        return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for tf in self.transforms:
            data = tf(data)

        return data

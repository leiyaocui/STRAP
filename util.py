import os
from PIL import Image, ImageDraw
import numpy as np
import pickle
import pydensecrf.densecrf as dcrf


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


np.seterr(invalid="ignore")


def IoU(output, target, num_class, ignore_index=255):
    pred = output.int().cpu().numpy()
    target = target.int().cpu().numpy()
    mask = target != ignore_index

    pred = pred[mask]
    target = target[mask]

    hist = np.bincount(num_class * target + pred, minlength=num_class ** 2).reshape(
        num_class, num_class
    )
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    return ious[1] * 100


def save_colorful_image(data, file_name, save_dir, palettes):
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = data.cpu().numpy()
    img = Image.fromarray(palettes[data])
    img.save(save_path)


def dense_crf(img, probs, mode):
    n_labels = probs.shape[0]

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    if mode == "softmax":
        U = -np.log(np.clip(probs, 1e-5, 1.0)).reshape(n_labels, -1)
    elif mode == "sigmoid":
        U = -np.log(np.concatenate([1 - probs, probs], axis=0)).reshape(2, -1)
    elif mode == "label":
        probs = probs.flatten()
        gt_prob = 0.7

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
    image,
    save_path,
    keypoint,
    stroke_width=20,
    bgd_label=None,
    ignore_index=255,
):
    weak_label = []
    for idx, (_, joints) in enumerate(keypoint):
        if len(joints) > 0:
            if bgd_label is None:
                label = Image.new(
                    "L", (image.shape[1], image.shape[0]), color=ignore_index
                )
            else:
                label = Image.fromarray(bgd_label, mode="L")

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
            # else:
            #     label = Image.fromarray(pred[idx], mode="L")
        else:
            label = Image.new("L", (image.shape[1], image.shape[0]), color=0)

        weak_label.append(label)

    weak_label = np.stack(weak_label, axis=2)
    with open(save_path, "wb") as fb:
        pickle.dump(weak_label, fb)

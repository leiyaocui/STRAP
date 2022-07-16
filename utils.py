import os
from PIL import Image
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def fast_hist(pred, target, n):
    mask = (target >= 0) & (target < n)
    return torch.bincount(
        n * target[mask].int() + pred[mask], minlength=n ** 2
    ).reshape(n, n)


@torch.no_grad()
def per_class_iu(hist):
    return torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))


@torch.no_grad()
def IoU(output, target):
    num_classes = output.shape[1]
    pred = output.argmax(dim=1)
    hist = torch.zeros((num_classes, num_classes), device=output.device)
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return ious.cpu()


@torch.no_grad()
def save_colorful_image(data, file_name, save_dir, palettes):
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = data.cpu().numpy().squeeze()
    img = Image.fromarray(palettes[data])
    img.save(save_path)

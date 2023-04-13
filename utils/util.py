import numpy as np
import torch
import torch.distributed as dist


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self, value=0.0, count=0.0):
        self.value = value
        self.count = count

    def update(self, value, count):
        self.value += value * count
        self.count += count

    def get(self):
        return self.value / self.count


@torch.no_grad()
def reduce_dict(data_dict, local_rank):
    world_size = dist.get_world_size()
    if world_size == 1:
        return data_dict

    key_list = list(sorted(data_dict.keys()))
    value_list = [torch.as_tensor(data_dict[k]).cuda(local_rank) for k in key_list]

    value_list = torch.stack(value_list, dim=0)
    dist.all_reduce(value_list)
    value_list /= world_size

    reduced_dict = {k: v for k, v in zip(key_list, value_list)}

    return reduced_dict


np.seterr(invalid="ignore")


@torch.no_grad()
def IoU(output, target, num_class, ignore_index=255):
    pred = output.int().flatten().cpu().numpy()
    target = target.int().flatten().cpu().numpy()
    mask = target != ignore_index

    pred = pred[mask]
    target = target[mask]

    if target.max() == 0:
        return np.nan
    else:
        hist = np.bincount(num_class * target + pred, minlength=num_class**2).reshape(
            num_class, num_class
        )

        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

        return iou[1] * 100

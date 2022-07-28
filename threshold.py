import os
import yaml
from tqdm import tqdm
import torch
import numpy as np

from dataset import make_dataloader
import transform as TF
from model import CerberusAffordanceModel
from util import IoU, AverageMeter


class CerberusMain:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as fb:
            config = yaml.safe_load(fb)

        self.data_dir = config["data_dir"]

        self.class_list = config["affordance"]
        self.num_class = len(self.class_list)

        self.model = CerberusAffordanceModel(self.num_class).cuda()

        train_tf = TF.Compose(
            [
                TF.ConvertPointLabel(
                    stroke_width=config["stroke_diameter"], ignore_index=255
                ),
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(
                    mean=config["dataset_mean"], std=config["dataset_std"]
                ),
            ]
        )

        self.train_loader = make_dataloader(
            self.data_dir,
            "train_affordance",
            train_tf,
            label_level=["dense", "point"],
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        val_tf = TF.Compose(
            [
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(
                    mean=config["dataset_mean"], std=config["dataset_std"]
                ),
            ]
        )

        self.val_loader = make_dataloader(
            self.data_dir,
            "val_affordance",
            val_tf,
            label_level=["dense"],
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        if os.path.isfile(config["resume"]):
            checkpoint = torch.load(
                config["resume"], map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(
                {
                    k.replace("module.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }
            )

    @torch.no_grad()
    def threshold_eval(self):
        self.model.eval()

        threshold = [0.0] * self.num_class
        cnt = [0] * self.num_class

        for data in tqdm(self.train_loader, ncols=80):
            input = data["image"].cuda(non_blocking=True)
            target = data["weak_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)
            output = self.model(input)

            pred = []
            for i in range(self.num_class):
                pred.append(output[i].squeeze(1))

            for i in range(self.num_class):
                mask_fg = (target[i] == 1)
                mask_cnt = mask_fg.sum().item()
                if mask_cnt > 0:
                    threshold[i] += pred[i][mask_fg].sum().item()
                    cnt[i] += mask_cnt

        threshold_avg = [torch.sigmoid(torch.tensor(threshold[i] / cnt[i])).item() for i in range(self.num_class)]
        print(threshold_avg)

        score_meter = AverageMeter()
        score_per_class_meter = [AverageMeter() for _ in range(self.num_class)]

        for data in tqdm(self.val_loader, ncols=80):
            input = data["image"].cuda(non_blocking=True)
            target = data["dense_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)
            output = self.model(input)

            pred = []
            for i in range(self.num_class):
                pred.append((output[i].sigmoid().squeeze(1) > threshold_avg[i]).int())
                # pred.append((output[i].sigmoid().squeeze(1) > 0.9).int ())

            score = []
            for i in range(self.num_class):
                score_per_class = IoU(pred[i], target[i], num_class=2, ignore_index=255)
                score.append(score_per_class)
                if not np.isnan(score_per_class):
                    score_per_class_meter[i].update(score_per_class, input.shape[0])

            score = np.nanmean(score)
            if not np.isnan(score):
                score_meter.update(score, input.shape[0])

        print(f"miou_val: {score_meter.avg}")
        for i, it in enumerate(self.class_list):
            print(f"iou_{it}_val: {score_per_class_meter[i].avg}")

if __name__ == "__main__":
    cerberus = CerberusMain("train_cad120.yaml")
    cerberus.threshold_eval()

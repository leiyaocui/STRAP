import os
import numpy as np
import shutil
from tqdm import tqdm
from datetime import datetime
import pickle
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import make_dataloader
import transform as TF
from model import DPTAffordanceModel

from loss import bce_loss
from util import IoU, AverageMeter


class GenPseudoLabel:
    def __init__(self, config):
        self.save_dir = os.path.join(
            config["save_dir"], datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save Dir: {os.path.abspath(self.save_dir)}")

        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "log"))

        self.data_dir = config["data_dir"]
        self.ignore_index = 255

        self.dataset_mean = config["dataset_mean"]
        self.dataset_std = config["dataset_std"]

        self.class_list = config["affordance"]
        self.num_class = len(self.class_list)

        self.model = DPTAffordanceModel(self.num_class).cuda()

        self.epochs = config["epochs"]
        self.initial_lr = config["lr"]

        self.pseudo_label_dir = os.path.join(self.save_dir, "pseudo_label")
        os.makedirs(self.pseudo_label_dir, exist_ok=True)

        train_tf = TF.Compose(
            [
                TF.RandomHorizonalReflect(),
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

        self.train_loader = make_dataloader(
            self.data_dir,
            "train_affordance",
            train_tf,
            label_level=["dense", "pseudo"],
            pseudo_label_dir=self.pseudo_label_dir,
            batch_size=4,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )

        params = [
            {"params": self.model.pretrained.parameters()},
            {"params": self.model.scratch.parameters()},
        ]

        for i in range(self.num_class):
            params.append({"params": self.model.head_dict[str(i)].parameters()})

        self.optimizer = torch.optim.SGD(
            params,
            self.initial_lr,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        self.best_score = -1

    def exec(self):
        self.gen_pseudo(point_radius=0)
        for epoch in range(1, self.epochs + 1):
            if epoch % 20 == 0:
                checkpoint = torch.load(
                    os.path.join(self.save_dir, "model", "model_best.pth"),
                    map_location=lambda storage, loc: storage,
                )
                self.best_score = checkpoint["score"]
                self.model.load_state_dict(
                    {
                        k.replace("module.", ""): v
                        for k, v in checkpoint["state_dict"].items()
                    }
                )

                point_radius = (epoch * 320 / self.epochs) * 0.25
                self.gen_pseudo(point_radius)

            lr = self.initial_lr * (1 - (epoch % 20) / 100) ** 0.9
            for idx, param_group in enumerate(self.optimizer.param_groups):
                if idx == 0 or idx == 1:
                    param_group["lr"] = lr / self.num_class
                else:
                    param_group["lr"] = lr

            score = self.train(epoch)
            self.save_checkpoint(epoch, score)

    def train(self, epoch):
        self.model.train()

        loss_meter = AverageMeter()
        score_meter = AverageMeter()
        score_per_class_meter = [AverageMeter() for _ in range(self.num_class)]

        loop = tqdm(
            self.train_loader, desc=f"[Train] Epoch {epoch:03d}", leave=False, ncols=100
        )
        for data in loop:
            input = data["image"].cuda(non_blocking=True)
            target = data["pseudo_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)

            output = self.model(input)

            pred = []
            for i in range(self.num_class):
                output[i] = F.interpolate(
                    output[i], target[i].shape[-2:], mode="bilinear", align_corners=True
                )
                pred.append((output[i].detach() > 0.5).int())

            score = []
            for i in range(self.num_class):
                score_per_class = IoU(
                    pred[i],
                    data["dense_label"][i],
                    num_class=2,
                    ignore_index=self.ignore_index,
                )
                score.append(score_per_class)
                if not np.isnan(score_per_class):
                    score_per_class_meter[i].update(score_per_class, input.shape[0])

            score = np.nanmean(score)
            if not np.isnan(score):
                score_meter.update(score, input.shape[0])

            loss = []
            for i in range(self.num_class):
                l = bce_loss(output[i], target[i], ignore_index=self.ignore_index)
                loss.append(l)

            loss = sum(loss)
            if not torch.isnan(loss):
                loss_meter.update(loss.item(), input.shape[0])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loop.set_postfix(loss=loss.item(), score=score)

        self.writer.add_scalar(f"loss_train", loss_meter.avg, global_step=epoch)
        self.writer.add_scalar(f"miou_train", score_meter.avg, global_step=epoch)
        for i, it in enumerate(self.class_list):
            self.writer.add_scalar(
                f"iou_{it}_train", score_per_class_meter[i].avg, global_step=epoch
            )

        return score_meter.avg

    @torch.no_grad()
    def gen_pseudo(self, point_radius):
        assert point_radius >= 0
        self.model.eval()

        gen_pseudo_tf = TF.Compose(
            [
                TF.ConvertPointLabel(
                    self.num_class,
                    point_radius=point_radius,
                    ignore_index=self.ignore_index,
                ),
                TF.GenVisibleInfo(self.num_class),
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

        gen_pseudo_loader = make_dataloader(
            self.data_dir,
            "train_affordance",
            gen_pseudo_tf,
            label_level=["point"],
            pseudo_label_dir=self.pseudo_label_dir,
            batch_size=8,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )

        loop = tqdm(
            gen_pseudo_loader,
            desc=f"[Gen] Radius: {point_radius:.2f}",
            leave=False,
            ncols=100,
        )
        for data in loop:
            input = data["image"].cuda(non_blocking=True)
            visible_info = data["visible_info"]
            for i in range(self.num_class):
                visible_info[i] = visible_info[i].cuda(non_blocking=True)
            target = data["weak_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)

            if point_radius > 0:
                output = self.model(input)

            pred = []
            for i in range(self.num_class):
                if point_radius == 0:
                    p = target[i] * visible_info[i].reshape(-1, 1, 1)
                    pred.append(p.cpu().numpy())
                else:
                    out = F.interpolate(
                        output[i], input.shape[-2:], mode="bilinear", align_corners=True
                    ).squeeze(1)

                    p = torch.full_like(out, self.ignore_index)
                    p[(out > 0.99) & (target[i] == 1)] = 1
                    p[out < 0.01] = 0
                    p = p * visible_info[i].reshape(-1, 1, 1)
                    pred.append(p.cpu().numpy())

            pred = np.stack(pred, axis=3).astype(np.uint8)

            for i in range(pred.shape[0]):
                save_path = os.path.join(
                    self.pseudo_label_dir, data["file_name"][i] + ".pkl"
                )
                with open(save_path, "wb") as fb:
                    pickle.dump(pred[i], fb)

        del gen_pseudo_loader

    def save_checkpoint(self, epoch, score, backup_freq=10):
        save_dir = os.path.join(self.save_dir, "model")
        os.makedirs(save_dir, exist_ok=True)

        is_best = score > self.best_score
        self.best_score = max(score, self.best_score)

        state = {
            "state_dict": self.model.state_dict(),
            "score": score,
        }

        checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
        torch.save(state, checkpoint_path)

        if epoch % backup_freq == 0:
            history_path = os.path.join(save_dir, f"checkpoint_{epoch:03d}.pth")
            shutil.copyfile(checkpoint_path, history_path)

        if is_best:
            best_path = os.path.join(save_dir, "model_best.pth")
            shutil.copyfile(checkpoint_path, best_path)


if __name__ == "__main__":
    config = {
        "save_dir": "output",
        "data_dir": "../dataset/cad120/object",
        "affordance": [
            "openable",
            "cuttable",
            "pourable",
            "containable",
            "supportable",
            "holdable",
        ],
        "dataset_mean": [132.47758921, 106.32022472, 111.57047992],
        "dataset_std": [67.45043020, 70.23484331, 72.19806953],
        "epochs": 300,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    }
    main = GenPseudoLabel(config)
    main.exec()

import os
import numpy as np
import yaml
import shutil
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import make_dataloader
import transform as TF
from model import CerberusCAMModel
from loss import SigmoidCrossEntropyLoss
from util import AverageMeter


class CerberusCAMMain:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as fb:
            config = yaml.safe_load(fb)

        self.save_dir = os.path.join(
            config["save_dir"], datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save Dir: {os.path.abspath(self.save_dir)}")
        shutil.copyfile(yaml_path, os.path.join(self.save_dir, "archive_cam_config.yaml"))

        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "log"))

        self.mode = config["mode"]

        self.data_dir = config["data_dir"]
        self.dst_size = tuple(config["random_crop_size"])

        self.class_list = config["affordance"]
        self.num_class = len(self.class_list)
        self.initial_class_weight = config["class_weight"]
        assert len(self.initial_class_weight) == self.num_class

        self.model = CerberusCAMModel(self.num_class).cuda()

        self.epochs = config["epochs"]
        self.initial_lr = config["lr"]

        self.loss_ce = SigmoidCrossEntropyLoss(ignore_index=255)

        train_tf = TF.Compose(
            [
                TF.RandomScaledTiltedWarpedPIL(
                    random_crop_size=self.dst_size,
                    random_scale_min=config["random_scale_min"],
                    random_scale_max=config["random_scale_max"],
                    random_tilt_max_deg=config["random_tilt_max_deg"],
                    random_wiggle_max_ratio=config["random_wiggle_max_ratio"],
                    random_horizon_reflect=config["random_horizon_reflect"],
                    center_offset_instead_of_random=False,
                    ignore_index=255,
                )
                if config["random_crop_size"][0] > 0
                and config["random_crop_size"][1] > 0
                else TF.Identity(),
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
            label_level=["dense", "image"],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["workers"],
            pin_memory=True,
            drop_last=True,
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
        self.start_epoch = 0

        if os.path.isfile(config["resume"]):
            checkpoint = torch.load(
                config["resume"], map_location=lambda storage, loc: storage
            )
            self.start_epoch = checkpoint["epoch"]
            self.best_score = checkpoint["best_score"]
            self.model.load_state_dict(
                {
                    k.replace("module.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }
            )
            print(f"Resume Epoch: {self.start_epoch}")
            print(f"Score: {checkpoint['score']}")
            print(f"Best Score: {self.best_score}")

    def exec(self):
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.adjust_learning_rate(epoch - 1)
            score = self.train(epoch)
            self.save_checkpoint(epoch, score)

    def adjust_learning_rate(self, epoch):
        # epoch in [0, self.epochs)
        lr = self.initial_lr * (1 - epoch / self.epochs) ** 0.9

        self.class_weight = [1.0 / self.num_class] * self.num_class
        for i in range(self.num_class):
            self.class_weight[i] = self.initial_class_weight[i]

        for idx, param_group in enumerate(self.optimizer.param_groups):
            if idx == 0 or idx == 1:
                param_group["lr"] = lr
            else:
                param_group["lr"] = lr / self.class_weight[idx - 2]

    def train(self, epoch):
        self.model.train()

        loss_meter = AverageMeter()
        score_meter = AverageMeter()
        score_per_class_meter = [AverageMeter() for _ in range(self.num_class)]

        loop = tqdm(
            self.train_loader,
            desc=f"[Train] Epoch {epoch:03d}",
            leave=False,
            unit="batch",
            ncols=100,
        )
        for data in loop:
            input = data["image"].cuda(non_blocking=True)
            target = data["image_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)

            output = self.model(input)

            pred = []
            for i in range(self.num_class):
                pred.append((output[i].detach().sigmoid() > 0.5).long())

            score = []
            for i in range(self.num_class):
                score_per_class = (pred[i] == target[i]).float().mean().item() * 100
                score.append(score_per_class)
                score_per_class_meter[i].update(score_per_class, input.shape[0])

            score = np.mean(score)
            score_meter.update(score, input.shape[0])

            loss = []
            for i in range(self.num_class):
                l_ce = self.loss_ce(output[i], target[i])
                l = l_ce * self.class_weight[i]
                loss.append(l)

            loss = sum(loss)
            if not torch.isnan(loss):
                loss_meter.update(loss.item(), input.shape[0])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loop.set_postfix(loss=loss.item(), score=score)

        self.writer.add_scalar(f"loss_train", loss_meter.avg, global_step=epoch)
        self.writer.add_scalar(f"macc_train", score_meter.avg, global_step=epoch)
        for i, it in enumerate(self.class_list):
            self.writer.add_scalar(
                f"acc_{it}_train", score_per_class_meter[i].avg, global_step=epoch
            )

        return score_meter.avg

    def save_checkpoint(self, epoch, score, backup_freq=10):
        save_dir = os.path.join(self.save_dir, "model")
        os.makedirs(save_dir, exist_ok=True)

        is_best = score > self.best_score
        self.best_score = max(score, self.best_score)

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "score": score,
            "best_score": self.best_score,
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
    cerberus = CerberusCAMMain("train_cam_cad120.yaml")
    cerberus.exec()

import os
import numpy as np
import yaml
import argparse
import shutil
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import make_dataloader
import utils.transform as TF
from models.model import DPTAffordanceModel

from utils.loss import bce_loss, gated_crf_loss
from utils.util import IoU, AverageMeter


class STRAP_SECOND:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as fb:
            config = yaml.safe_load(fb)

        self.save_dir = config["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save Dir: {os.path.abspath(self.save_dir)}")

        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "log"))

        self.data_dir = config["data_dir"]
        self.ignore_index = 255

        self.dataset_mean = config["dataset_mean"]
        self.dataset_std = config["dataset_std"]

        self.class_list = config["affordance"]
        self.num_class = len(self.class_list)

        self.model = DPTAffordanceModel(config["num_objects"], self.num_class, use_hf=True)

        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.initial_lr = config["lr"]

        self.crf_config = config["crf"]

        self.model = self.model.cuda()

        train_tf = TF.Compose(
            [
                TF.RandomHorizonalFlipPIL(),
                TF.ConvertPointLabel(
                    self.num_class, ignore_index=self.ignore_index
                ),
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(
                    mean=self.dataset_mean, std=self.dataset_std
                ),
            ]
        )

        self.train_loader = make_dataloader(
            self.data_dir,
            "train_affordance",
            train_tf,
            label_level=["dense", "point"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config["workers"],
            pin_memory=True,
            drop_last=True,
        )

        val_tf = TF.Compose(
            [
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

        self.val_loader = make_dataloader(
            self.data_dir,
            "val_affordance",
            val_tf,
            label_level=["dense"],
            batch_size=1,
            shuffle=False,
            num_workers=config["workers"],
            pin_memory=True,
            drop_last=False,
        )

        params = [
            {"params": self.model.pretrained.parameters()},
            {"params": self.model.scratch.parameters()},
        ]

        for i in range(len(self.model.head_dict)):
            params.append({"params": self.model.head_dict[str(i)].parameters()})
        params.append({"params": self.model.hierarchical_head.parameters()})

        self.optimizer = torch.optim.SGD(
            params,
            self.initial_lr,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

        self.best_score = -1
        self.start_epoch = 0

        if os.path.isfile(config["resume"]):
            checkpoint = torch.load(
                config["resume"], map_location=lambda storage, loc: storage
            )

            self.model.load_state_dict(
                {
                    k.replace("module.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                },
                strict=False,
            )

            if not config["restart"]:
                self.start_epoch = checkpoint["epoch"]
                self.best_score = checkpoint["best_score"]

            print(f"Resume Epoch: {self.start_epoch}")
            print(f"Score: {checkpoint['score']}")
            print(f"Best Score: {self.best_score}")

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    def exec(self):
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.adjust_learning_rate(epoch - 1)
            self.train(epoch)
            score = self.validate(epoch)
            self.save_checkpoint(epoch, score)

    def adjust_learning_rate(self, epoch):
        # epoch in [0, self.epochs)
        lr = self.initial_lr * (1 - epoch / self.epochs) ** 0.9

        for idx, param_group in enumerate(self.optimizer.param_groups):
            if idx < 2:
                param_group["lr"] = lr / self.num_class
            else:
                param_group["lr"] = lr

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
            target = data["weak_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)
            dense_target = data["dense_label"]
            for i in range(self.num_class):
                dense_target[i] = dense_target[i].cuda(non_blocking=True)
            visible_info = (
                torch.stack(data["visible_info"], dim=1).cuda(non_blocking=True).float()
            )

            output, output_h = self.model(input, with_hc=True)
            output_h = output_h[-1]

            pred = []
            for i in range(self.num_class):
                pred.append((output[i].detach() > 0).int())

            score = []
            for i in range(self.num_class):
                score_per_class = IoU(
                    pred[i],
                    dense_target[i],
                    num_class=2,
                    ignore_index=self.ignore_index,
                )
                if not np.isnan(score_per_class):
                    score_per_class_meter[i].update(score_per_class, input.shape[0])
                    score.append(score_per_class)

            if len(score) > 0:
                score = np.mean(score)
                score_meter.update(score, input.shape[0])

            loss = []
            for i in range(self.num_class):
                l_ce = bce_loss(output[i], target[i], ignore_index=self.ignore_index)
                l_crf = self.crf_config["weight"] * gated_crf_loss(
                    input,
                    output[i],
                    kernels_desc=self.crf_config["kernels_desc"],
                    kernels_radius=self.crf_config["kernels_radius"],
                )
                l = l_ce + l_crf
                loss.append(l)
            loss = sum(loss)
            loss += F.binary_cross_entropy_with_logits(output_h, visible_info)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), input.shape[0])
            loop.set_postfix(loss=loss.item(), score=score)

        self.writer.add_scalar(f"loss_train", loss_meter.get(), global_step=epoch)
        self.writer.add_scalar(f"miou_train", score_meter.get(), global_step=epoch)
        for i, it in enumerate(self.class_list):
            self.writer.add_scalar(
                f"iou_{it}_train", score_per_class_meter[i].get(), global_step=epoch
            )

        return score_meter.get()

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        score_meter = AverageMeter()
        score_per_class_meter = [AverageMeter() for _ in range(self.num_class)]

        loop = tqdm(
            self.val_loader, desc=f"[Val] Epoch {epoch:03d}", leave=False, ncols=100
        )
        for data in loop:
            input = data["image"].cuda(non_blocking=True)
            target = data["dense_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)

            output = self.model(input)

            pred = []
            for i in range(self.num_class):
                pred.append((output[i].detach() > 0).int())

            score = []
            for i in range(self.num_class):
                score_per_class = IoU(
                    pred[i],
                    target[i],
                    num_class=2,
                    ignore_index=self.ignore_index,
                )
                if not np.isnan(score_per_class):
                    score_per_class_meter[i].update(score_per_class, input.shape[0])
                    score.append(score_per_class)

            if len(score) > 0:
                score = np.mean(score)
                score_meter.update(score, input.shape[0])

            loop.set_postfix(score=score)

        self.writer.add_scalar(f"miou_val", score_meter.get(), global_step=epoch)
        for i, it in enumerate(self.class_list):
            self.writer.add_scalar(
                f"iou_{it}_val", score_per_class_meter[i].get(), global_step=epoch
            )

        return score_meter.get()

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
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        metavar="PATH",
        help="YAML Config Path",
    )
    args = parser.parse_args()

    yaml_path = args.config
    print("Config: " + yaml_path)
    if os.path.exists(yaml_path):
        main = STRAP_SECOND(yaml_path)
        main.exec()

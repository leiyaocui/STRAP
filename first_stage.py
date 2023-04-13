import os
import numpy as np
import yaml
import argparse
import shutil
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel

from datasets.dataset import make_dataset
import utils.transform as TF
from models.model import DPTAffordanceModel

from utils.loss import bce_loss, gated_crf_loss
from utils.util import IoU, AverageMeter, reduce_dict


class STRAP_FIRST:
    def __init__(self, yaml_path):
        dist.init_process_group("nccl")
        self.local_rank = dist.get_rank()

        with open(yaml_path, "r") as fb:
            config = yaml.safe_load(fb)

        self.save_dir = config["save_dir"]
        if self.local_rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Config: {yaml_path}")
            print(f"Save Dir: {os.path.abspath(self.save_dir)}")

        if self.local_rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "log"))

        self.data_dir = config["data_dir"]
        self.ignore_index = 255

        self.dataset_mean = config["dataset_mean"]
        self.dataset_std = config["dataset_std"]

        self.class_list = config["affordance"]
        self.num_class = len(self.class_list)

        model = DPTAffordanceModel(config["num_objects"], self.num_class).cuda(
            self.local_rank
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = DistributedDataParallel(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )

        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.initial_lr = config["lr"]

        self.crf_config = config["crf"]

        train_tf = TF.Compose(
            [
                TF.ConvertPointLabel(self.num_class, ignore_index=self.ignore_index),
                TF.RandomScaledTiltedWarpedPIL(
                    random_crop_size=(320, 320),
                    random_scale_max=2.0,
                    random_scale_min=0.5,
                    random_horizon_reflect=True,
                    ignore_index=self.ignore_index,
                ),
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

        train_dataset = make_dataset(
            self.data_dir,
            "train_affordance",
            train_tf,
            label_level=["dense", "point"],
        )
        self.train_sampler = DistributedSampler(
            train_dataset, shuffle=False, drop_last=False
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=config["workers"],
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            sampler=self.train_sampler,
        )

        val_tf = TF.Compose(
            [
                TF.ConvertPointLabel(self.num_class, ignore_index=self.ignore_index),
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

        val_dataset = make_dataset(
            self.data_dir,
            "val_affordance",
            val_tf,
            label_level=["dense", "point"],
        )
        self.val_sampler = DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=config["workers"],
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            sampler=self.val_sampler,
        )

        params = [
            {"params": self.model.module.pretrained.parameters()},
            {"params": self.model.module.scratch.parameters()},
        ]

        for i in range(len(self.model.module.head_dict)):
            params.append({"params": self.model.module.head_dict[str(i)].parameters()})

        self.optimizer = torch.optim.SGD(
            params,
            self.initial_lr,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

        self.best_score = -1
        self.start_epoch = 0

        if os.path.isfile(config["resume"]) and self.local_rank:
            checkpoint = torch.load(
                config["resume"], map_location=lambda storage, loc: storage
            )

            self.model.load_state_dict(checkpoint["state_dict"])

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

            self.train_sampler.set_epoch(epoch)

            self.model.train()
            self.run_one_epoch(epoch, self.train_loader)

            self.val_sampler.set_epoch(epoch)
            self.model.eval()
            with torch.no_grad():
                score = self.run_one_epoch(epoch, self.val_loader)

            if self.local_rank == 0:
                self.save_checkpoint(epoch, score)

    def adjust_learning_rate(self, epoch):
        # epoch in [0, self.epochs)
        lr = self.initial_lr * (1 - epoch / self.epochs) ** 0.9

        for idx, param_group in enumerate(self.optimizer.param_groups):
            if idx < 2:
                param_group["lr"] = lr / self.num_class
            else:
                param_group["lr"] = lr

    def run_one_epoch(self, epoch, dataloader):
        grad_enabled = torch.is_grad_enabled()
        if grad_enabled:
            mode = "train"
        else:
            mode = "val"

        loss_meter = AverageMeter()
        score_meter = AverageMeter()
        score_per_class_meter = [AverageMeter() for _ in range(self.num_class)]

        if self.local_rank == 0:
            loop = tqdm(
                dataloader,
                desc=f"[{mode}] Epoch {epoch:03d}",
                leave=False,
                ncols=100,
            )
        else:
            loop = dataloader

        for data in loop:
            input = data["image"].cuda(self.local_rank, non_blocking=True)
            target = data["weak_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(self.local_rank, non_blocking=True)
            dense_target = data["dense_label"]
            for i in range(self.num_class):
                dense_target[i] = dense_target[i].cuda(self.local_rank, non_blocking=True)

            output = self.model(input)

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

            if grad_enabled:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_meter.update(loss.item(), input.shape[0])

            if self.local_rank == 0:
                loop.set_postfix(loss=loss.item(), score=score)

        if self.local_rank == 0:
            reduced_dict = {"loss": loss_meter.get(), "miou": score_meter.get()}
            for i, it in enumerate(self.class_list):
                reduced_dict[f"{it}_iou"] = score_per_class_meter[i].get()
            reduced_dict = reduce_dict(reduced_dict, self.local_rank)

            loss_meter.reset(reduced_dict["loss"], 1)
            score_meter.reset(reduced_dict["miou"], 1)
            for i, it in enumerate(self.class_list):
                score_per_class_meter[i].reset(reduced_dict[f"{it}_iou"], 1)

            self.writer.add_scalar(f"loss_{mode}", loss_meter.get(), global_step=epoch)
            self.writer.add_scalar(f"miou_{mode}", score_meter.get(), global_step=epoch)
            for i, it in enumerate(self.class_list):
                self.writer.add_scalar(
                    f"iou_{it}_{mode}",
                    score_per_class_meter[i].get(),
                    global_step=epoch,
                )

        return score_meter.get()

    def save_checkpoint(self, epoch, score, backup_freq=10):
        save_dir = os.path.join(self.save_dir, "model")
        os.makedirs(save_dir, exist_ok=True)

        is_best = score > self.best_score
        self.best_score = max(score, self.best_score)

        state = {
            "epoch": epoch,
            "state_dict": self.model.module.state_dict(),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="YAML Config Path",
    )
    args = parser.parse_args()

    yaml_path = args.config
    if os.path.exists(yaml_path):
        strap = STRAP_FIRST(yaml_path)
        strap.exec()


if __name__ == "__main__":
    main()

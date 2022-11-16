import os
import numpy as np
import yaml
import argparse
import shutil
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import skimage.segmentation
import skimage.draw
import skimage.morphology

from datasets.dataset import make_dataloader
import utils.transform as TF
from models.model import DPTAffordanceModel

from utils.loss import bce_loss, gated_crf_loss
from utils.util import IoU, AverageMeter


class STRAP_EM:
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

        self.model = DPTAffordanceModel(config["num_objects"], self.num_class, use_hf=True).cuda()
        self.model_dir = os.path.join(self.save_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        self.epochs = config["epochs"]
        self.initial_lr = config["lr"]

        self.crf_config = config["crf"]

        self.pseudo_label_dir = os.path.join(self.save_dir, "pseudo_label")
        os.makedirs(self.pseudo_label_dir, exist_ok=True)

        self.rng = np.random.default_rng()
        gen_pseudo_tf = TF.Compose(
            [
                TF.ConvertPointLabel(self.num_class, ignore_index=self.ignore_index),
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

        self.gen_pseudo_loader = make_dataloader(
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

        train_tf = TF.Compose(
            [
                TF.RandomHorizonalFlipPIL(),
                TF.ConvertPointLabel(self.num_class, ignore_index=self.ignore_index),
                TF.PILToTensor(),
                TF.ImageNormalizeTensor(mean=self.dataset_mean, std=self.dataset_std),
            ]
        )

        self.train_loader = make_dataloader(
            self.data_dir,
            "train_affordance",
            train_tf,
            label_level=["dense", "pseudo", "point"],
            pseudo_label_dir=self.pseudo_label_dir,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["workers"],
            pin_memory=True,
            drop_last=False,
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

        for i in range(self.num_class):
            params.append({"params": self.model.head_dict[str(i)].parameters()})
        params.append({"params": self.model.hierarchical_head.parameters()})

        self.optimizer = torch.optim.SGD(
            params,
            self.initial_lr,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

        assert os.path.isfile(config["resume"])
        checkpoint = torch.load(
            config["resume"], map_location=lambda storage, loc: storage
        )

        self.model.load_state_dict(
            {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        )

        self.best_score = -1

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    def exec(self):
        P = 10
        for epoch in range(1, self.epochs + 1):
            self.adjust_learning_rate((epoch - 1) % P)
            if epoch % P == 1:
                self.gen_pseudo(epoch, use_dilation=True, use_disk=True)

            self.train(epoch)
            score = self.validate(epoch)
            self.save_checkpoint(epoch, score)

    def adjust_learning_rate(self, epoch):
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
            target = data["pseudo_label"]
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

    @torch.no_grad()
    def gen_pseudo(self, epoch, use_dilation=False, use_disk=False, use_rnd=True):
        self.model.eval()

        if use_disk:
            radius = np.ceil(epoch / self.epochs * 100)
        threshold = 7

        if use_dilation:
            footprint = skimage.morphology.disk(10)

        loop = tqdm(
            self.gen_pseudo_loader,
            desc=f"[Gen] Epoch: {epoch:03d}",
            leave=False,
            ncols=100,
        )
        for data in loop:
            input = data["image"].cuda(non_blocking=True)
            weak_label = data["weak_label"]
            visible_info = data["visible_info"]
            for i in range(self.num_class):
                weak_label[i] = weak_label[i].cpu().numpy()
                visible_info[i] = visible_info[i].reshape(-1, 1, 1).cpu().numpy()

            output = self.model(input)

            pred_list = []
            for i in range(self.num_class):
                label = weak_label[i]
                out = output[i].squeeze(1).cpu().numpy()
                pred = (out > 0).astype(np.uint8)

                fg_mask_list = []
                if use_disk:
                    disk_mask_list = []
                bg_mask_list = []
                for b in range(label.shape[0]):
                    keypoints = np.argwhere(label[b] == 1)
                    if len(keypoints) > 0:
                        fg_mask = np.zeros(label[b].shape, dtype=np.uint8)
                        disk_mask = np.zeros(label[b].shape, dtype=np.uint8)
                        for it in keypoints:
                            mask = skimage.segmentation.flood(
                                pred[b], tuple(it), connectivity=1
                            )
                            fg_mask[mask] = 1
                            if use_disk:
                                disk_rr, disk_cc = skimage.draw.disk(
                                    tuple(it), radius, shape=disk_mask.shape
                                )
                                disk_mask[disk_rr, disk_cc] = 1

                        if use_dilation:
                            bg_mask = skimage.morphology.binary_dilation(
                                fg_mask, footprint
                            ).astype(np.uint8)
                        else:
                            bg_mask = fg_mask.copy()

                        if use_rnd:
                            rnd_p = epoch / self.epochs
                            rnd_mask = self.rng.choice(
                                [False, True], size=fg_mask.shape, p=[rnd_p, 1 - rnd_p]
                            )
                            fg_mask[rnd_mask] = self.ignore_index
                            bg_mask[rnd_mask] = self.ignore_index
                    else:
                        fg_mask = np.ones(label[b].shape, dtype=np.uint8)
                        if use_disk:
                            disk_mask = np.ones(label[b].shape, dtype=np.uint8)
                        bg_mask = np.zeros(label[b].shape, dtype=np.uint8)

                    fg_mask_list.append(fg_mask)
                    if use_disk:
                        disk_mask_list.append(disk_mask)
                    bg_mask_list.append(bg_mask)
                fg_mask = np.stack(fg_mask_list, axis=0)
                if use_disk:
                    disk_mask = np.stack(disk_mask_list, axis=0)
                bg_mask = np.stack(bg_mask_list, axis=0)

                if use_disk:
                    fg_mask = fg_mask * disk_mask

                p = np.full(label.shape, self.ignore_index)
                p[(bg_mask == 0) & (out < -threshold)] = 0
                p[(fg_mask == 1) & (out > threshold)] = 1
                p[label == 1] = 1
                p = p * visible_info[i]

                pred_list.append(p)

            pred = np.stack(pred_list, axis=3).astype(np.uint8)

            for i in range(pred.shape[0]):
                save_path = os.path.join(
                    self.pseudo_label_dir, data["file_name"][i] + ".pkl"
                )
                with open(save_path, "wb") as fb:
                    pickle.dump(pred[i], fb)

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
        main = STRAP_EM(yaml_path)
        main.exec()

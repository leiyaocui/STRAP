import os
import numpy as np
import yaml
import shutil
from tqdm import tqdm
from datetime import datetime
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
from skimage.segmentation import flood

from dataset import make_dataloader
import transform as TF
from model import DPTAffordanceModel

from loss import bce_loss
from util import IoU, AverageMeter


class GenPseudoLabel:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as fb:
            config = yaml.safe_load(fb)

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
        self.model_dir = os.path.join(self.save_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        self.epochs = config["epochs"]
        self.initial_lr = config["lr"]

        self.pseudo_label_dir = os.path.join(self.save_dir, "pseudo_label")
        os.makedirs(self.pseudo_label_dir, exist_ok=True)

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
            batch_size=config["batch_size"],
            shuffle=True,
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
        self.best_score = checkpoint["best_score"]
        self.model.load_state_dict(
            {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    def exec(self):
        for epoch in range(1, self.epochs + 1):
            self.adjust_learning_rate(epoch - 1)
            if epoch % 20 == 1:
                if epoch < 20:
                    self.gen_pseudo(epoch, mode=1)
                else:
                    self.gen_pseudo(epoch, mode=2)

            self.train(epoch)
            score = self.validate(epoch)
            self.save_checkpoint(epoch, score)

    def adjust_learning_rate(self, epoch):
        # epoch in [0, self.epochs)
        lr = self.initial_lr * (1 - epoch / self.epochs) ** 0.9

        for idx, param_group in enumerate(self.optimizer.param_groups):
            if idx < 2:
                param_group["lr"] = lr / self.num_class
            elif idx:
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

            output = self.model(input)

            pred = []
            for i in range(self.num_class):
                pred.append((output[i].detach() > 0).int())

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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), input.shape[0])
            loop.set_postfix(loss=loss.item(), score=score)

        self.writer.add_scalar(f"loss_train", loss_meter.avg, global_step=epoch)
        self.writer.add_scalar(f"miou_train", score_meter.avg, global_step=epoch)
        for i, it in enumerate(self.class_list):
            self.writer.add_scalar(
                f"iou_{it}_train", score_per_class_meter[i].avg, global_step=epoch
            )

        return score_meter.avg

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
                    pred[i], target[i], num_class=2, ignore_index=self.ignore_index
                )
                score.append(score_per_class)
                if not np.isnan(score_per_class):
                    score_per_class_meter[i].update(score_per_class, input.shape[0])

            score = np.nanmean(score)
            if not np.isnan(score):
                score_meter.update(score, input.shape[0])

            loop.set_postfix(score=score)

        self.writer.add_scalar(f"miou_val", score_meter.avg, global_step=epoch)
        for i, it in enumerate(self.class_list):
            self.writer.add_scalar(
                f"iou_{it}_val", score_per_class_meter[i].avg, global_step=epoch
            )

        return score_meter.avg

    @torch.no_grad()
    def gen_pseudo(self, epoch, mode=0):
        self.model.eval()

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

            if mode == 0:
                pred = np.stack(data["weak_label"], axis=1)
                pred = pred.transpose(1, 0, 2, 3)
            elif mode == 1:
                pred_list = []
                for i in range(self.num_class):
                    label = weak_label[i]
                    out = output[i]
                    pred = (out > -8).int().squeeze(1).cpu().numpy()

                    fg_mask_list = []
                    for b in range(label.shape[0]):
                        keypoints = np.argwhere(label[b] == 1)
                        if len(keypoints) > 0:
                            fg_mask = np.zeros(label[b].shape, dtype=np.uint8)
                            for it in keypoints:
                                mask = flood(pred[b], tuple(it), connectivity=1)
                                fg_mask[mask] = 1
                        else:
                            fg_mask = np.ones(label[b].shape, dtype=np.uint8)
                        fg_mask_list.append(fg_mask)
                    fg_mask = np.stack(fg_mask_list, axis=0)

                    p = np.zeros(label.shape)
                    p[fg_mask == 1] = self.ignore_index
                    p[label == 1] = 1
                    p = p * visible_info[i]

                    pred_list.append(p)

                pred = np.stack(pred_list, axis=3).astype(np.uint8)
            elif mode == 2:
                pred_list = []
                for i in range(self.num_class):
                    label = weak_label[i]
                    out = output[i]
                    pred = (out > 0).int().squeeze(1).cpu().numpy()

                    fg_mask_list = []
                    for b in range(label.shape[0]):
                        keypoints = np.argwhere(label[b] == 1)
                        if len(keypoints) > 0:
                            fg_mask = np.zeros(label[b].shape, dtype=np.uint8)
                            for it in keypoints:
                                mask = flood(pred[b], tuple(it), connectivity=1)
                                fg_mask[mask] = 1
                        else:
                            fg_mask = np.ones(label[b].shape, dtype=np.uint8)
                        fg_mask_list.append(fg_mask)
                    fg_mask = np.stack(fg_mask_list, axis=0)

                    p = np.full(label.shape, self.ignore_index)
                    p[out < -8] = 0
                    p[(fg_mask == 1) & (out > 8)] = 1
                    p[label == 1] = 1
                    p = p * visible_info[i]

                    pred_list.append(p)

                pred = np.stack(pred_list, axis=3).astype(np.uint8)
            else:
                raise ValueError(f"mode: {mode} is not supported")

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
    main = GenPseudoLabel("train_cad120.yaml")
    main.exec()

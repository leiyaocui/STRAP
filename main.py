import os
import numpy as np
import yaml
import shutil
from tqdm import tqdm
from datetime import datetime
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import make_dataloader
import transform as TF
from model import CerberusAffordanceModel
from loss import SigmoidCrossEntropyLoss
from util import IoU, AverageMeter, save_colorful_image


class CerberusMain:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as fb:
            config = yaml.safe_load(fb)

        self.save_dir = os.path.join(
            config["save_dir"], datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save Dir: {os.path.abspath(self.save_dir)}")
        shutil.copyfile(yaml_path, os.path.join(self.save_dir, "archive_config.yaml"))

        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "log"))

        self.mode = config["mode"]

        self.data_dir = config["data_dir"]
        self.dst_size = tuple(config["random_crop_size"])

        self.class_list = config["affordance"]
        self.num_class = len(self.class_list)
        # self.class_weight = np.array(config["class_weight"])
        # self.initial_class_weight = 1.0 / self.class_hist
        # assert len(self.initial_class_weight) == self.num_class

        self.model = CerberusAffordanceModel(self.num_class)

        if self.mode == "train":
            self.epochs = config["epochs"]
            self.initial_lr = config["lr"]
            self.train_level = config["train_level"]
            assert self.train_level in ["dense", "weak_offline", "weak_online"]
            self.update_dataset = config["update_dataset"]

            keypoint_path = os.path.join(
                self.data_dir, "train_affordance_keypoint.yaml"
            )
            with open(keypoint_path, "r") as fb:
                self.keypoint_dict = yaml.safe_load(fb)

            self.model = self.model.cuda()

            self.loss_neg_weight = config["neg_weight"]
            assert len(self.loss_neg_weight) == self.num_class
            self.loss_ce_list = []
            for i in range(self.num_class):
                # self.loss_ce_list.append(
                #     torch.nn.CrossEntropyLoss(
                #         weight=torch.tensor(
                #             [self.loss_neg_weight[i], 1], dtype=torch.float32
                #         ),
                #         ignore_index=255,
                #     ).cuda()
                # )
                self.loss_ce_list.append(
                    torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
                )
            # self.loss_ce = SigmoidCrossEntropyLoss(ignore_index=255)
            # self.loss_crf = SigmoidGatedCRFLoss(
            #     kernels_desc=[{"weight": 1, "xy": 6, "image": 0.1}], kernels_radius=5,
            # )

            if self.train_level == "weak_offline" and self.update_dataset:
                updata_tf = TF.Compose(
                    [
                        TF.ConvertPointLabel(
                            self.num_class, config["point_diameter"], ignore_index=255
                        ),
                        TF.PILToTensor(),
                    ]
                )

                self.update_loader = make_dataloader(
                    self.data_dir,
                    "train_affordance",
                    updata_tf,
                    label_level=["point"],
                    batch_size=1,
                    shuffle=False,
                    num_workers=config["workers"],
                    pin_memory=True,
                    drop_last=False,
                )

            train_tf = TF.Compose(
                [
                    TF.ConvertPointLabel(
                        self.num_class, config["point_diameter"], ignore_index=255
                    )
                    if self.train_level == "weak_online"
                    and config["point_diameter"] > 0
                    else TF.Identity(),
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
                    TF.ConvertPointLabel(
                        self.num_class, config["point_diameter"], ignore_index=255
                    )
                    if self.train_level == "weak_online"
                    and config["point_diameter"] == 0
                    else TF.Identity(),
                    TF.PILToTensor(),
                    TF.ImageNormalizeTensor(
                        mean=config["dataset_mean"], std=config["dataset_std"]
                    ),
                ]
            )

            if self.train_level == "weak_offline":
                train_label_level = ["dense", "weak"]
            elif self.train_level == "weak_online":
                train_label_level = ["dense", "point"]
            else:
                train_label_level = ["dense"]

            self.train_loader = make_dataloader(
                self.data_dir,
                "train_affordance",
                train_tf,
                label_level=train_label_level,
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
        elif self.mode == "test":
            self.save_vis = config["save_vis"]

        val_tf = TF.Compose(
            [
                TF.ResizePIL(self.dst_size),
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
            num_workers=config["workers"],
            pin_memory=(self.mode == "train"),
            drop_last=False,
        )

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
        if self.mode == "train":
            if self.train_level == "weak_offline" and self.update_dataset:
                self.update()
            for epoch in range(self.start_epoch + 1, self.epochs + 1):
                self.adjust_learning_rate(epoch - 1)
                self.train(epoch)
                score = self.validate(epoch)

                self.save_checkpoint(epoch, score)

        elif self.mode == "test":
            self.validate(epoch, save_vis=self.save_vis)

    def adjust_learning_rate(self, epoch):
        # epoch in [0, self.epochs)
        lr = self.initial_lr * (1 - epoch / self.epochs) ** 0.9

        for idx, param_group in enumerate(self.optimizer.param_groups):
            if idx == 0 or idx == 1:
                param_group["lr"] = lr / self.num_class
            else:
                param_group["lr"] = lr

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
        step_idx = 0
        for data in loop:
            step_idx += 1
            input = data["image"].cuda(non_blocking=True)
            # mask = data["validity"].cuda(non_blocking=True)
            if self.train_level == "dense":
                target = data["dense_label"]
            else:
                target = data["weak_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)

            output = self.model(input)

            pred = []
            for i in range(self.num_class):
                pred.append(output[i].detach().argmax(dim=1))
                # pred.append((output[i].detach().sigmoid() > 0.5).long())

            score = []
            for i in range(self.num_class):
                score_per_class = IoU(
                    pred[i], data["dense_label"][i], num_class=2, ignore_index=255
                )
                score.append(score_per_class)
                if not np.isnan(score_per_class):
                    score_per_class_meter[i].update(score_per_class, input.shape[0])

            score = np.nanmean(score)
            if not np.isnan(score):
                score_meter.update(score, input.shape[0])

            loss = []
            for i in range(self.num_class):
                l_ce = self.loss_ce_list[i](output[i], target[i])
                # l_crf = self.loss_crf(input, output[i], mask_src=mask)
                # l = l_ce + 0.1 * l_crf
                l = l_ce
                loss.append(l)

                # self.writer.add_scalar(
                #     f"per_step_loss_{self.class_list[i]}_train",
                #     l.item(),
                #     global_step=(epoch - 1) * loop.total + step_idx,
                # )

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
    def validate(self, epoch, save_vis=False):
        self.model.eval()

        score_meter = AverageMeter()
        score_per_class_meter = [AverageMeter() for _ in range(self.num_class)]

        loop = tqdm(
            self.val_loader,
            desc=f"[Val] Epoch {epoch:03d}",
            leave=False,
            unit="batch",
            ncols=100,
        )
        step_idx = 0
        for data in loop:
            step_idx += 1
            input = data["image"].cuda(non_blocking=True)
            target = data["dense_label"]
            for i in range(self.num_class):
                target[i] = target[i].cuda(non_blocking=True)
            output = self.model(input)

            pred = []
            for i in range(self.num_class):
                pred.append(output[i].detach().argmax(dim=1))
                # pred.append((output[i].detach().sigmoid() > 0.5).long())

            score = []
            for i in range(self.num_class):
                score_per_class = IoU(pred[i], target[i], num_class=2, ignore_index=255)
                score.append(score_per_class)
                if not np.isnan(score_per_class):
                    score_per_class_meter[i].update(score_per_class, input.shape[0])

            score = np.nanmean(score)
            if not np.isnan(score):
                score_meter.update(score, input.shape[0])

            loop.set_postfix(score=score)

            if save_vis:
                palette = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
                save_dir = os.path.join(self.save_dir, "test_result")

                for i in range(self.num_class):
                    file_name = f"{data['file_name']}_{self.class_list[i]}_pred.png"
                    save_colorful_image(pred[i], file_name, save_dir, palette)

        self.writer.add_scalar(f"miou_val", score_meter.avg, global_step=epoch)
        for i, it in enumerate(self.class_list):
            self.writer.add_scalar(
                f"iou_{it}_val", score_per_class_meter[i].avg, global_step=epoch
            )

        return score_meter.avg

    @torch.no_grad()
    def update(self):
        self.model.eval()

        loop = tqdm(
            self.update_loader, desc=f"[Update]", leave=False, unit="batch", ncols=100
        )
        for data in loop:
            file_name = data["file_name"][0]
            save_path = os.path.join(
                self.data_dir, "train_affordance_weak_label", file_name + ".pkl"
            )

            weak_label = data["weak_label"]
            for i in range(self.num_class):
                weak_label[i] = weak_label[i].squeeze(0).cpu().numpy().astype(np.uint8)
            weak_label = np.stack(weak_label, axis=2)
            with open(save_path, "wb") as fb:
                pickle.dump(weak_label, fb)

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
    # CUDA_VISIBLE_DEVICES=1

    cerberus = CerberusMain("train_cad120.yaml")
    cerberus.exec()

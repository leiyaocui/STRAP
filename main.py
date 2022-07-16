import os
from tabnanny import check
import numpy as np
import yaml
import shutil
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import make_dataloader
import transforms as TF
from models import CerberusSegmentationModel
from losses import GatedCRFLoss
from utils import (
    IoU,
    AverageMeter,
    save_colorful_image,
)


class CerberusMain:
    def __init__(self, yaml_path):
        config = yaml.safe_load(open(yaml_path, "r"))

        self.mode = config["mode"]
        self.save_dir = os.path.join(
            config["save_dir"], self.mode, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "log"))

        self.task_root_list = list(config["task"].keys())
        assert len(self.task_root_list) == 1
        self.task_list = [v["category"] for v in config["task"].values()]

        self.model = CerberusSegmentationModel(config["task"])

        if self.mode == "train":
            self.epochs = config["epochs"]
            self.validate_freq = config["validate_freq"]

            self.model = self.model.cuda()

            self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
            self.loss_crf = GatedCRFLoss(
                kernels_desc=[{"weight": 1, "xy": 6, "image": 0.1}], kernels_radius=5,
            ).cuda()

            random_crop_size = config["random_crop_size"]
            random_crop_size = (
                random_crop_size
                if isinstance(random_crop_size, tuple)
                else (random_crop_size, random_crop_size)
            )

            train_tf = TF.Compose(
                [
                    TF.RandomScaledTiltedWarpedPIL(
                        random_crop_size=random_crop_size,
                        random_scale_min=config["random_scale_min"],
                        random_scale_max=config["random_scale_max"],
                        random_tilt_max_deg=config["random_tilt_max_deg"],
                        random_wiggle_max_ratio=config["random_wiggle_max_ratio"],
                        random_horizon_reflect=config["random_horizon_reflect"],
                        ignore_index=255,
                        center_offset_instead_of_random=False,
                    )
                    if random_crop_size[0] > 0 and random_crop_size[1] > 0
                    else TF.Identity(),
                    TF.PointCoordinatesToPIL(ignore_index=255, stroke_width=10),
                    TF.PILToTensor(),
                    TF.ImageNormalizeTensor(
                        mean=config["dataset_mean"], std=config["dataset_std"]
                    ),
                ]
            )

            self.train_loader = make_dataloader(
                config["data_dir"],
                f"train_{self.task_root_list[0]}",
                train_tf,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["workers"],
                pin_memory=True,
                drop_last=True,
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
                config["data_dir"],
                f"val_{self.task_root_list[0]}",
                val_tf,
                batch_size=1,
                shuffle=False,
                num_workers=config["workers"],
                pin_memory=True,
            )

            self.optimizer = torch.optim.SGD(
                [
                    {"params": self.model.pretrained.parameters()},
                    {"params": self.model.scratch.parameters()},
                ],
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )

            if config["lr_mode"] == "step":
                lambda_func = lambda e: 0.1 ** (e // config["step"])

            elif config["lr_mode"] == "poly":
                lambda_func = lambda e: (1 - e / self.epochs) ** 0.9
            else:
                raise ValueError(f'Unknown lr mode {config["lr_mode"]}')

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda_func
            )

            torch.backends.cudnn.benchmark = True
        elif self.mode == "test":
            self.save_vis = config["save_vis"]

            val_tf = TF.Compose(
                [
                    TF.PILToTensor(),
                    TF.ImageNormalizeTensor(
                        mean=config["dataset_mean"], std=config["dataset_std"]
                    ),
                ]
            )

            self.val_loader = make_dataloader(
                config["data_dir"],
                f"val_{self.task_root_list[0]}",
                val_tf,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["workers"],
                pin_memory=True,
            )
        else:
            raise ValueError(f"Unknown exec mode {self.mode}")

        self.best_score = 0
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
            print(
                f"Loaded checkpoint:"
                f"Epoch: {checkpoint['epoch']} Score: {checkpoint['best_score']}"
            )
        else:
            print(f"No checkpoint is found")

    def exec(self):
        if self.mode == "train":
            for epoch in range(self.start_epoch + 1, self.epochs + 1):
                self.train(epoch)

                self.scheduler.step()

                if epoch % self.validate_freq == 0:
                    score = self.validate(epoch)

                    is_best = score > self.best_score
                    self.best_score = max(score, self.best_score)

                    self.save_checkpoint(epoch, is_best)
        elif self.mode == "test":
            self.validate(epoch, save_vis=self.save_vis)

    def train(self, epoch):
        self.model.train()

        loss_list = AverageMeter()
        # loss_per_task_list = [AverageMeter() for _ in range(len(self.task_list[0]))]
        score_list = AverageMeter()

        for data in tqdm(
            self.train_loader, desc=f"[Train] Epoch {epoch:03d}", ncols=80
        ):
            input = data["image"].cuda(non_blocking=True)

            target = data["weak_label"]
            for i in range(len(target)):
                target[i] = target[i].cuda(non_blocking=True)

            mask = data["validity"].cuda(non_blocking=True)

            output = self.model(input)

            score = []
            for i in range(len(output)):
                iou = IoU(output[i], target[i])[1]
                if not np.isnan(iou):
                    score.append(iou)

            if len(score) > 0:
                score_list.update(np.mean(score), input.shape[0])

            loss = []
            for idx in range(len(output)):
                loss_ce_single = self.loss_ce(output[idx], target[idx])
                loss_crf_single = self.loss_crf(
                    input, output[i].softmax(dim=1), mask_src=mask,
                )

                if torch.isnan(loss_ce_single):
                    loss_single = 0.1 * loss_crf_single
                else:
                    loss_single = loss_ce_single + 0.1 * loss_crf_single

                loss.append(loss_single)

                # loss_per_task_list[idx].update(
                #     loss_single.item(), input.shape[0]
                # )

            loss = sum(loss) / len(output)

            loss_list.update(loss.item(), input.shape[0])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.writer.add_scalar(
            f"train_{self.task_root_list[0]}_loss_avg",
            loss_list.avg,
            global_step=epoch,
        )
        self.writer.add_scalar(
            f"train_{self.task_root_list[0]}_score_avg",
            score_list.avg,
            global_step=epoch,
        )
        # for i, it in enumerate(self.task_list[0]):
        #     self.writer.add_scalar(
        #         f"train_class_{it}_loss_avg",
        #         loss_per_task_list[i].avg,
        #         global_step=epoch,
        #     )

    @torch.no_grad()
    def validate(self, epoch, save_vis=False):
        self.model.eval()

        score_list = AverageMeter()
        score_per_task_list = [AverageMeter() for _ in range(len(self.task_list[0]))]

        for data in tqdm(self.val_loader, desc=f"[Val] Epoch {epoch:03d}", ncols=80):
            input = data["image"].cuda(non_blocking=True)

            target = data["dense_label"]
            for i in range(len(target)):
                target[i] = target[i].cuda(non_blocking=True)

            output = self.model(input)

            score = []
            for i in range(len(output)):
                iou = IoU(output[i], target[i])[1]
                if not np.isnan(iou):
                    score.append(iou)
                    score_per_task_list[i].update(iou, input.shape[0])

            if len(score) > 0:
                score_list.update(np.mean(score), input.shape[0])

            if save_vis:
                palette = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
                save_dir = os.path.join(self.save_dir, "test_result")

                for idx in range(len(output)):
                    pred = output[idx].argmax(dim=1)
                    file_name = f"{data['file_name']}_{self.task_list[0][idx]}_pred.png"
                    save_colorful_image(pred, file_name, save_dir, palette)

        self.writer.add_scalar(
            f"val_{self.task_root_list[0]}_miou_avg", score_list.avg, global_step=epoch,
        )
        for i, it in enumerate(self.task_list[0]):
            self.writer.add_scalar(
                f"val_class_{it}_iou_avg",
                score_per_task_list[i].avg,
                global_step=epoch,
            )

        return score_list.avg

    def save_checkpoint(self, epoch, is_best, backup_freq=10):
        save_dir = os.path.join(self.save_dir, "model")
        os.makedirs(save_dir, exist_ok=True)

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    cerberus = CerberusMain("train_weak_cad120.yaml")
    cerberus.exec()

    # cerberus = CerberusMain("train_cad120.yaml")
    # cerberus.exec()

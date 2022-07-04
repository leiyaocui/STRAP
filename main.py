import os
import numpy as np
import yaml
import shutil
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import EMDataset
import data_transforms as transforms
from models import CerberusSegmentationModelSingle
from utils import (
    IoU,
    MinNormSolver,
    AverageMeter,
    AFFORDANCE_PALETTE,
    update_label,
    save_colorful_image,
)


class CerberusSingleTrain:
    def __init__(self, yaml_path):
        config = yaml.safe_load(open(yaml_path, "r"))

        self.mode = config["mode"]
        self.save_dir = os.path.join(
            config["save_dir"], self.mode, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.task_root_list = list(config["task"].keys())
        assert len(self.task_root_list) == 1
        self.task_list = [v["category"] for v in config["task"].values()]

        self.model = CerberusSegmentationModelSingle(config["task"])

        if self.mode == "train":
            self.epochs = config["epochs"]

            keypoint_list = np.loadtxt(config["keypoint_file"], delimiter=",")

            self.keypoint_dict = {}
            for i in range(keypoint_list.shape[0]):
                key = f"1{int(keypoint_list[i, 0]):04d}_{int(keypoint_list[i, 1])}_{int(keypoint_list[i, 2])}"
                self.keypoint_dict[key] = [keypoint_list[i, 4], keypoint_list[i, 3]]

            self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "log"))

            self.model = self.model.cuda()
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()

            train_tf = []

            if config["random_rotate"] > 0:
                train_tf.append(
                    transforms.RandomRotateMultiHead(config["random_rotate"])
                )
            if config["random_scale"] > 0:
                train_tf.append(transforms.RandomScaleMultiHead(config["random_scale"]))

            train_tf.extend(
                [
                    transforms.RandomCropMultiHead(config["random_crop"]),
                    transforms.RandomHorizontalFlipMultiHead(),
                    transforms.ToTensorMultiHead(),
                    transforms.Normalize(
                        mean=config["data_mean"], std=config["data_std"]
                    ),
                ]
            )

            train_tf = transforms.Compose(train_tf)

            dataset_train = EMDataset(
                config["data_dir"], f"train_{self.task_root_list[0]}", train_tf
            )

            self.train_loader = DataLoader(
                dataset_train,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["workers"],
                pin_memory=True,
                drop_last=True,
            )

            update_train_tf = transforms.Compose(
                [
                    transforms.ToTensorMultiHead(),
                    transforms.Normalize(
                        mean=config["data_mean"], std=config["data_std"]
                    ),
                ]
            )

            dataset_update_train = EMDataset(
                config["data_dir"],
                f"train_{self.task_root_list[0]}",
                update_train_tf,
                out_name=True,
            )

            self.update_train_loader = DataLoader(
                dataset_update_train,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            )

            val_tf = transforms.Compose(
                [
                    transforms.ToTensorMultiHead(),
                    transforms.Normalize(
                        mean=config["data_mean"], std=config["data_std"]
                    ),
                ]
            )

            dataset_val = EMDataset(
                config["data_dir"], f"val_{self.task_root_list[0]}", val_tf
            )

            self.val_loader = DataLoader(
                dataset_val,
                batch_size=1,
                shuffle=False,
                num_workers=config["workers"],
                pin_memory=True,
                drop_last=True,
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
            self.have_gt = config["have_gt"]

            test_tf = transforms.Compose(
                [
                    transforms.ToTensorMultiHead(),
                    transforms.Normalize(
                        mean=config["data_mean"], std=config["data_std"]
                    ),
                ]
            )

            dataset_test = EMDataset(
                config["data_dir"], "val_affordance", test_tf, out_name=True,
            )

            self.test_loader = DataLoader(
                dataset_test,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["workers"],
            )
        else:
            raise ValueError(f"Unknown exec mode {self.mode}")

        self.best_score = 0
        self.start_epoch = 0

        if os.path.isfile(config["resume"]):
            checkpoint = torch.load(config["resume"])
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
            print(f"No checkpoint found")

    def exec(self):
        if self.mode == "train":
            for epoch in range(self.start_epoch, self.epochs):
                self.train(epoch)
                score = self.validate(epoch)
                self.scheduler.step()

                is_best = score > self.best_score
                self.best_score = max(score, self.best_score)
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_score": self.best_score,
                }
                save_dir = os.path.join(self.save_dir, "model")
                self.save_checkpoint(state, is_best, save_dir)
        elif self.mode == "test":
            save_dir = os.path.join(self.save_dir, "test_pred_img")
            self.test(save_dir, save_vis=self.save_vis, have_gt=self.have_gt)
        else:
            raise ValueError(f"Unknown exec mode {self.mode}")

    def train(self, epoch):
        self.model.train()

        loss_list = AverageMeter()
        loss_per_task_list = [AverageMeter() for _ in range(len(self.task_list[0]))]
        score_list = AverageMeter()

        for input, target in tqdm(
            self.train_loader, desc=f"[Train] Epoch {epoch+1:04d}", ncols=80
        ):
            input = input.cuda(non_blocking=True)
            for i in range(len(target)):
                target[i] = target[i].cuda(non_blocking=True)
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
                loss_single = self.criterion(output[idx], target[idx])
                loss.append(self.criterion(output[idx], target[idx]))
                loss_per_task_list[idx].update(
                    loss_single.item(), input.shape[0]
                )
            loss = sum(loss)
            loss_list.update(loss.item(), input.shape[0])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.writer.add_scalar(
            f"train_epoch_{self.task_root_list[0]}_loss_avg", loss_list.avg, global_step=epoch
        )
        self.writer.add_scalar(
            f"train_epoch_{self.task_root_list[0]}_score_avg", score_list.avg, global_step=epoch
        )
        for i, it in enumerate(self.task_list[0]):
            self.writer.add_scalar(
                f"train_epoch_task_{it}_loss_avg",
                loss_per_task_list[i].avg,
                global_step=epoch,
            )

        with torch.no_grad():
            for input, target, file_path in tqdm(
                self.update_train_loader,
                desc=f"[UpdateTrain] Epoch {epoch+1:04d}",
                ncols=80,
            ):
                input = input.cuda(non_blocking=True)
                file_path = file_path[0]
                id = os.path.basename(file_path).split(".")[0]

                output = self.model(input)

                for i in range(len(output)):
                    output[i] = output[i].argmax(dim=1).cpu()

                for i, it in enumerate(output):
                    key = f"{id}_{i+1}"
                    if key not in self.keypoint_dict.keys():
                        continue

                    keypoint = self.keypoint_dict[f"{id}_{i+1}"]
                    keypoint = np.asarray(
                        [
                            [int(keypoint[0]), int(keypoint[0]) + 1],
                            [int(keypoint[1]), int(keypoint[1]) + 1],
                        ]
                    )

                    # only support batch_size = 1.
                    if (
                        it[0, keypoint[0, 0], keypoint[1, 0]] == 1
                        and it[0, keypoint[0, 0], keypoint[1, 1]] == 1
                        and it[0, keypoint[0, 1], keypoint[1, 0]] == 1
                        and it[0, keypoint[0, 1], keypoint[1, 1]] == 1
                    ):
                        output[i] = np.sign(target[i] + it).squeeze(axis=0)
                    else:
                        output[i] = np.sign(target[i]).squeeze(axis=0)

                data = np.stack(output, axis=0).astype(np.uint8)

                update_label(data, file_path)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        loss_list = AverageMeter()
        # loss_per_task_list = [AverageMeter() for _ in range(len(self.task_list[0]))]
        score_list = AverageMeter()
        score_per_task_list = [AverageMeter() for _ in range(len(self.task_list[0]))]

        for input, target in tqdm(
            self.val_loader, desc=f"[Val] Epoch {epoch+1:04d}", ncols=80
        ):
            input = input.cuda(non_blocking=True)
            for i in range(len(target)):
                target[i] = target[i].cuda(non_blocking=True)
            output = self.model(input)

            loss = []
            for idx in range(len(output)):
                loss_single = self.criterion(output[idx], target[idx])
                loss.append(loss_single)
                # loss_per_task_list[idx].update(
                #     loss_single.item(), input.shape[0]
                # )
            loss = sum(loss)
            loss_list.update(loss.item(), input.shape[0])

            score = []
            for i in range(len(output)):
                iou = IoU(output[i], target[i])[1]
                if not np.isnan(iou):
                    score.append(iou)
                    score_per_task_list[i].update(iou, input.shape[0])

            if len(score) > 0:
                score_list.update(np.mean(score), input.shape[0])

        self.writer.add_scalar(
            f"val_epoch_{self.task_root_list[0]}_loss_avg",
            loss_list.avg,
            global_step=epoch,
        )
        self.writer.add_scalar(
            f"val_epoch_{self.task_root_list[0]}_score_avg",
            score_list.avg,
            global_step=epoch,
        )
        for i, it in enumerate(self.task_list[0]):
            # self.writer.add_scalar(
            #     f"val_epoch_task_{it}_loss_avg",
            #     loss_per_task_list[i].avg,
            #     global_step=epoch,
            # )
            self.writer.add_scalar(
                f"val_epoch_task_{it}_score_avg",
                score_per_task_list[i].avg,
                global_step=epoch,
            )

        return score_list.avg

    @torch.no_grad()
    def test(self, save_dir, save_vis, have_gt):
        self.model.eval()

        for data_i, data in enumerate(self.test_loader):
            PALETTE = AFFORDANCE_PALETTE

            name = data[-1][0]
            print(f"File name: {name}")
            name = os.path.basename(name)

            input = data[0]
            if have_gt:
                target = data[1]

            output = self.model(input)

            if save_vis:
                assert ".png" in name or ".jpg" in name or ".bmp" in name
                for idx in range(len(output)):
                    file_name = f"{data_i}/{self.task_list[0][idx]}/{name[:-4]}.png"
                    pred = output[idx].argmax(dim=1)
                    # save_image(pred, file_name, save_dir)
                    save_colorful_image(pred, file_name, f"{save_dir}_color", PALETTE)

            if have_gt:
                for idx in range(len(target)):
                    file_name = f"{data_i}/{self.task_list[0][idx]}/{name[:-4]}_gt.png"
                    pred = target[idx].argmax(dim=0)
                    # save_image(pred, file_name, save_dir)
                    save_colorful_image(pred, file_name, f"{save_dir}_color", PALETTE)

                score = []
                for i in range(len(output)):
                    iou = IoU(output[i], target[i])[1]
                    if not np.isnan(iou):
                        score.append(iou)
                score = np.mean(score) / input.shape[0]

                print(f"Task {self.task_root_list[0]} score: {score:.2f}")

    def save_checkpoint(self, state, is_best, save_dir, backup_freq=10):
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
        torch.save(state, checkpoint_path)

        if is_best:
            best_path = os.path.join(save_dir, "model_best.pth")
            shutil.copyfile(checkpoint_path, best_path)

        if state["epoch"] % backup_freq == 0:
            history_path = os.path.join(
                save_dir, f"checkpoint_{state['epoch']:04d}.pth"
            )
            shutil.copyfile(checkpoint_path, history_path)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    cerberus = CerberusSingleTrain("train_weak_cad120.yaml")
    cerberus.exec()
    # cerberus = CerberusSingleTrain("test_weak_cad120.yaml")
    # cerberus.exec()

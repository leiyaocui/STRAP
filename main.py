import os
import numpy as np
import yaml
import shutil
import threading
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import SegMultiHeadList, ConcatSegList
import data_transforms as transforms
from models import CerberusSegmentationModelMultiHead
from utils import (
    mIoU,
    MinNormSolver,
    AverageMeter,
    AFFORDANCE_PALETTE,
    NYU40_PALETTE,
    save_image,
    save_colorful_image,
)


class CerberusTrain:
    def __init__(self, yaml_path):
        config = yaml.safe_load(open(yaml_path, "r"))

        self.mode = config["mode"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.task_root_list = ["At", "Af", "Seg"]
        self.task_list = [
            [
                "Wood",
                "Painted",
                "Paper",
                "Glass",
                "Brick",
                "Metal",
                "Flat",
                "Plastic",
                "Textured",
                "Glossy",
                "Shiny",
            ],
            ["L", "M", "R", "S", "W"],
            ["Segmentation"],
        ]

        self.model = CerberusSegmentationModelMultiHead()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.criterion = self.criterion.to(self.device)

        if self.mode == "train":
            self.epochs = config["epochs"]

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

            dataset_at_train = SegMultiHeadList(
                config["data_dir"], "train_attribute", train_tf
            )
            dataset_af_train = SegMultiHeadList(
                config["data_dir"], "train_affordance", train_tf
            )
            dataset_seg_train = SegMultiHeadList(config["data_dir"], "train", train_tf)

            self.train_loader = DataLoader(
                ConcatSegList(dataset_at_train, dataset_af_train, dataset_seg_train),
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["workers"],
                pin_memory=True,
                drop_last=True,
            )

            val_tf = transforms.Compose(
                [
                    transforms.RandomCropMultiHead(config["random_crop"]),
                    transforms.ToTensorMultiHead(),
                    transforms.Normalize(
                        mean=config["data_mean"], std=config["data_std"]
                    ),
                ]
            )

            dataset_at_val = SegMultiHeadList(
                config["data_dir"], "val_attribute", val_tf
            )
            dataset_af_val = SegMultiHeadList(
                config["data_dir"], "val_affordance", val_tf
            )
            dataset_seg_val = SegMultiHeadList(config["data_dir"], "val", val_tf)

            self.val_loader = DataLoader(
                ConcatSegList(dataset_at_val, dataset_af_val, dataset_seg_val),
                batch_size=1,
                shuffle=False,
                num_workers=config["workers"],
                pin_memory=True,
                drop_last=True,
            )

            # 这里学习率是这么设置吗？
            self.optimizer = torch.optim.SGD(
                [
                    {"params": self.model.pretrained.parameters(), "lr": config["lr"]},
                    {"params": self.model.scratch.parameters(), "lr": config["lr"],},
                    {
                        "params": self.model.sigma.parameters(),
                        "lr": config["lr"] * 0.01,
                    },
                ],
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

        elif self.mode == "test":
            self.ms_scales = config["ms_scales"]
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

            dataset_at_test = SegMultiHeadList(
                config["data_dir"],
                "val_attribute",
                test_tf,
                ms_scale=self.ms_scales,
                out_name=True,
            )
            dataset_af_test = SegMultiHeadList(
                config["data_dir"],
                "val_affordance",
                test_tf,
                ms_scale=self.ms_scales,
                out_name=True,
            )
            dataset_seg_test = SegMultiHeadList(
                config["data_dir"],
                "val",
                test_tf,
                ms_scale=self.ms_scales,
                out_name=True,
            )

            self.test_loader = DataLoader(
                ConcatSegList(dataset_at_test, dataset_af_test, dataset_seg_test),
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["workers"],
                pin_memory=False,
            )
        else:
            raise ValueError(f"Unknown exec mode {self.mode}")

        torch.backends.cudnn.benchmark = True

        self.best_prec1 = 0
        self.start_epoch = 0

        if os.path.isfile(config["resume"]):
            print(f"loading checkpoint '{config['resume']}'")
            checkpoint = torch.load(config["resume"])
            self.start_epoch = checkpoint["epoch"]
            self.best_prec1 = checkpoint["best_prec1"]
            self.model.load_state_dict(
                {
                    k.replace("module.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }
            )
            print(
                f"loaded checkpoint '{config['resume']}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"no checkpoint found at '{config['resume']}'")

        self.writer = SummaryWriter()

    def exec(self):
        if self.mode == "train":
            for epoch in range(self.start_epoch, self.epochs):
                self.writer.add_scalars(
                    f"train_lr",
                    {
                        "pretrained": self.scheduler.get_last_lr()[0],
                        "scratch": self.scheduler.get_last_lr()[1],
                    },
                    global_step=epoch,
                )

                self.train(epoch)
                prec1 = self.validate(epoch)

                is_best = prec1 > self.best_prec1
                self.best_prec1 = max(prec1, self.best_prec1)
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_prec1": self.best_prec1,
                }
                save_dir = "output/model"
                self.save_checkpoint(state, is_best, save_dir)
        elif self.mode == "test":
            self.test(save_vis=self.save_vis, have_gt=self.have_gt)
        else:
            raise ValueError(f"Unknown exec mode {self.mode}")

    def train(self, epoch):
        self.model.train()

        loss_list = []
        loss_per_task_list = []
        score_list = []

        for i in range(len(self.task_root_list)):
            loss_list.append(AverageMeter())
            loss_per_task_list.append(
                [AverageMeter() for _ in range(len(self.task_list[i]))]
            )
            score_list.append(AverageMeter())

        for task_data_pair in self.train_loader:
            grads = {}
            task_loss = []
            for task_i, (input, target) in enumerate(task_data_pair):
                input = input.to(self.device)
                target = [target[i].to(self.device) for i in range(len(target))]
                output = self.model(input, task_i)
                assert len(output) == len(target) and len(
                    self.task_list[task_i]
                ) == len(target)

                self.optimizer.zero_grad(set_to_none=True)

                loss = []
                for idx in range(len(output)):
                    loss_single = self.criterion(output[idx], target[idx])
                    loss.append(loss_single)
                    loss_per_task_list[task_i][idx].update(
                        loss_single.item(), input.shape[0]
                    )

                loss = sum(loss)
                loss.backward(retain_graph=True)

                task_loss.append(loss)
                loss_list[task_i].update(loss.item(), input.shape[0])

                # 为什么只需要记录这些gradient，这里再看一下论文怎么说的？
                grads[self.task_root_list[task_i]] = []
                for cnt in self.model.pretrained.parameters():
                    if cnt.grad is not None:
                        grads[self.task_root_list[task_i]].append(cnt.grad.data.clone())
                grads[self.task_root_list[task_i]].append(
                    self.model.scratch.layer1_rn.weight.grad.data.clone()
                )
                grads[self.task_root_list[task_i]].append(
                    self.model.scratch.layer2_rn.weight.grad.data.clone()
                )
                grads[self.task_root_list[task_i]].append(
                    self.model.scratch.layer3_rn.weight.grad.data.clone()
                )
                grads[self.task_root_list[task_i]].append(
                    self.model.scratch.layer4_rn.weight.grad.data.clone()
                )

                score = []
                # 计算IoU时，为什么不一样？或许是每个task要求不一样，前两个只看对的，最后那个看类别
                if task_i < 2:
                    for idx in range(len(output)):
                        ious = mIoU(output[idx], target[idx])
                        score.append(ious[1])
                elif task_i == 2:
                    for idx in range(len(output)):
                        ious = mIoU(output[idx], target[idx])
                        score.append(np.mean(ious))
                else:
                    raise ValueError(f"Not support task_i: {task_i}")

                score_list[task_i].update(np.mean(score), input.shape[0])

            sol, min_norm = MinNormSolver.find_min_norm_element(
                [grads[i] for i in self.task_root_list]
            )

            # self.writer.add_scalars(
            #     f"train_min_norm_scale",
            #     {'sol_0': sol[0], 'sol_1': sol[1], 'sol_2': sol[2]},
            #     global_step=epoch*len(self.train_loader) + data_i
            # )

            self.optimizer.zero_grad(set_to_none=True)

            loss = [sol[i] * task_loss[i] for i in range(len(self.task_root_list))]
            loss = sum(loss)
            loss.backward()

            self.optimizer.step()

        for i, it in enumerate(self.task_root_list):
            self.writer.add_scalar(
                f"train_epoch_{it}_loss_avg", loss_list[i].avg, global_step=epoch
            )
            self.writer.add_scalar(
                f"train_epoch_{it}_score_avg", score_list[i].avg, global_step=epoch
            )
            for j, it in enumerate(self.task_list[i]):
                self.writer.add_scalar(
                    f"train_epoch_task_{it}_loss_avg",
                    loss_per_task_list[i][j].avg,
                    global_step=epoch,
                )

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        loss_list = []
        loss_per_task_list = []
        score_list = []

        for i in range(len(self.task_root_list)):
            loss_list.append(AverageMeter())
            loss_per_task_list.append(
                [AverageMeter() for _ in range(len(self.task_list[i]))]
            )
            score_list.append(AverageMeter())

        for task_data_pair in self.val_loader:
            for task_i, (input, target) in enumerate(task_data_pair):
                input = input.to(self.device)
                target = [
                    target[i].to(self.device, non_blocking=True)
                    for i in range(len(target))
                ]

                output = self.model(input, task_i)

                loss = []
                for idx in range(len(output)):
                    loss_single = self.criterion(output[idx], target[idx])
                    loss.append(loss_single)
                    loss_per_task_list[task_i][idx].update(
                        loss_single.item(), input.shape[0]
                    )
                loss = sum(loss)
                loss_list[task_i].update(loss.item(), input.shape[0])

                score = []
                if task_i < 2:
                    for idx in range(len(output)):
                        ious = mIoU(output[idx], target[idx])
                        score.append(ious[1])
                elif task_i == 2:
                    for idx in range(len(output)):
                        ious = mIoU(output[idx], target[idx])
                        score.append(np.mean(ious))
                else:
                    raise ValueError(f"Not support task_i: {task_i}")

                score_list[task_i].update(np.mean(score), input.shape[0])

        for i, it in enumerate(self.task_root_list):
            self.writer.add_scalar(
                f"val_epoch_{it}_loss_avg", loss_list[i].avg, global_step=epoch
            )
            self.writer.add_scalar(
                f"val_epoch_{it}_score_avg", score_list[i].avg, global_step=epoch
            )
            for j, it in enumerate(self.task_list[i]):
                self.writer.add_scalar(
                    f"val_epoch_task_{it}_loss_avg",
                    loss_per_task_list[i][j].avg,
                    global_step=epoch,
                )

        score = np.mean([score_list[i].avg for i in range(len(self.task_root_list))])
        self.writer.add_text(
            f"[Val] Average Score", f"Epoch_{epoch}: {score:.3f}", global_step=epoch
        )
        self.writer.add_scalar("val_score_avg", score, global_step=epoch)

        return score

    @torch.no_grad()
    def test(self, save_vis=False, have_gt=False):
        self.model.eval()

        score_list = [AverageMeter() for _ in range(len(self.task_root_list))]

        for task_data_pair in self.test_loader:
            for task_i, data in enumerate(task_data_pair):
                if task_i < 2:
                    PALETTE = AFFORDANCE_PALETTE
                elif task_i == 2:
                    PALETTE = NYU40_PALETTE
                else:
                    raise ValueError(f"Not support task_i: {task_i}")

                task_list = self.task_list[task_i]
                iou_compute_cmd = "per_class_iu(hist_list[task_i][idx])"
                if task_i < 2:
                    iou_compute_cmd = "[" + iou_compute_cmd + "[1]]"

                name = data[-1]
                print(f"File name: {name}")

                base_input = data[0]
                ms_input = data[-2]

                if have_gt:
                    target = data[1]

                ms_output = []

                for input in ms_input:
                    input = input.to(self.device)
                    output = self.model(input, task_i)
                    output = F.interpolate(
                        output,
                        base_input.shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                    ms_output.append(output)

                ms_output = torch.sum(torch.as_tensor(ms_output), dim=0)
                output = ms_output.argmax(dim=1)

                if save_vis:
                    assert ".png" in name or ".jpg" in name or ".bmp" in name
                    for idx in range(len(output)):
                        file_name = (
                            os.path.basename(name)[:-4] + self.task_list[idx] + ".png"
                        )
                        save_dir = "output/test_pred_img"
                        save_image(output[idx], file_name, save_dir)
                        save_colorful_image(
                            output[idx], file_name, save_dir + "_color", PALETTE
                        )

                        if task_i == 2:
                            gt_name = (
                                os.path.basename(name)[:-4] + task_list[idx] + "_gt.png"
                            )
                            label_mask = target[idx] == 255
                            save_colorful_image(
                                (target[idx] - label_mask * 255).numpy(),
                                gt_name,
                                save_dir + "_color",
                                PALETTE,
                            )

                if have_gt:
                    score = []
                    if task_i < 2:
                        for idx in range(len(output)):
                            ious = mIoU(output[idx], target[idx])
                            score.append(ious[1])
                    elif task_i == 2:
                        for idx in range(len(output)):
                            ious = mIoU(output[idx], target[idx])
                            score.append(np.mean(ious))
                    else:
                        raise ValueError(f"Not support task_i: {task_i}")

                    score = np.mean(score)
                    score_list[task_i].update(score, input[0].shape[0])

                    print(f"Task {self.task_root_list[task_i]:.2f} score: {score}")

        if have_gt:
            score = np.mean(
                [score_list[i].avg for i in range(len(self.task_root_list))]
            )
            return score

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

    # def resize_4d_tensor(self, tensor, width, height):
    #     tensor_cpu = tensor.cpu().numpy()
    #     if tensor.size(2) == height and tensor.size(3) == width:
    #         return tensor_cpu
    #     out_size = (tensor.size(0), tensor.size(1), height, width)
    #     out = np.empty(out_size, dtype=np.float32)

    #     def resize_one(i, j):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (width, height), Image.BILINEAR
    #             )
    #         )

    #     def resize_channel(j):
    #         for i in range(tensor.size(0)):
    #             out[i, j] = np.array(
    #                 Image.fromarray(tensor_cpu[i, j]).resize(
    #                     (width, height), Image.BILINEAR
    #                 )
    #             )

    #     workers = [
    #         threading.Thread(target=resize_channel, args=(j,))
    #         for j in range(tensor.size(1))
    #     ]
    #     for w in workers:
    #         w.start()
    #     for w in workers:
    #         w.join()
    #     return out


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cerberus = CerberusTrain("test.yaml")
    cerberus.exec()

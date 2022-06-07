import os
import numpy as np
from PIL import Image
import yaml
import shutil
import threading
from timm.utils import AverageMeter
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import SegMultiHeadList, ConcatSegList
import data_transforms as transforms
from min_norm_solvers import MinNormSolver
from models import CerberusSegmentationModelMultiHead


def mIoU(output, target):
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)
    pred = pred.cpu().data.numpy().flatten()
    target = target.cpu().data.numpy().flatten()
    k = (target >= 0) & (target < num_classes)
    hist += np.bincount(
        num_classes * target[k].astype(int) + pred[k], minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)) * 100
    return np.round(np.nanmean(ious), 2)


class CerberusTrain:
    CITYSCAPE_PALETTE = np.asarray(
        [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    NYU40_PALETTE = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 80],
            [0, 0, 160],
            [0, 0, 240],
            [0, 80, 0],
            [0, 80, 80],
            [0, 80, 160],
            [0, 80, 240],
            [0, 160, 0],
            [0, 160, 80],
            [0, 160, 160],
            [0, 160, 240],
            [0, 240, 0],
            [0, 240, 80],
            [0, 240, 160],
            [0, 240, 240],
            [80, 0, 0],
            [80, 0, 80],
            [80, 0, 160],
            [80, 0, 240],
            [80, 80, 0],
            [80, 80, 80],
            [80, 80, 160],
            [80, 80, 240],
            [80, 160, 0],
            [80, 160, 80],
            [80, 160, 160],
            [80, 160, 240],
            [80, 240, 0],
            [80, 240, 80],
            [80, 240, 160],
            [80, 240, 240],
            [160, 0, 0],
            [160, 0, 80],
            [160, 0, 160],
            [160, 0, 240],
            [160, 80, 0],
            [160, 80, 80],
            [160, 80, 160],
            [160, 80, 240],
        ],
        dtype=np.uint8,
    )

    AFFORDANCE_PALETTE = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

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
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)
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
                shuffle=True,
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

            self.optimizer = torch.optim.SGD(
                [
                    {"params": self.model.pretrained.parameters(), "lr": config["lr"]},
                    {
                        "params": self.model.scratch.parameters(),
                        "lr": config["lr"] * 0.01,
                    },
                ],
                # {'params': sel.model.sigma.parameters(), 'lr': config['lr'] * 0.01}],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )

            if config["lr_mode"] == "step":

                def lambda_func(e):
                    return 0.1 ** (e // config["step"])

            elif config["lr_mode"] == "poly":

                def lambda_func(e):
                    return (1 - e / self.epochs) ** 0.9

            else:
                raise ValueError(f'Unknown lr mode {config["lr_mode"]}')

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda_func
            )

        elif self.mode == "test":
            self.ms_scales = config["ms_scales"]

            test_tf = transforms.Compose(
                [
                    transforms.ToTensorMultiHead(),
                    transforms.Normalize(
                        mean=config["data_mean"], std=config["data_std"]
                    ),
                ]
            )

            dataset_at_test = SegMultiHeadList(
                config["data_dir"], "var_attribute", test_tf, ms_scale=self.ms_scales
            )
            dataset_af_test = SegMultiHeadList(
                config["data_dir"],
                "var_affordance",
                test_tf,
                scale=self.scales,
            )
            dataset_seg_test = SegMultiHeadList(
                config["data_dir"], "var", test_tf, ms_scale=self.ms_scales
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
                self.writer.add_text(
                    f"Epoch: {epoch}",
                    f"[Train] pretrained_lr: {self.scheduler.get_last_lr()[0]:.06f}",
                )
                self.writer.add_text(
                    f"Epoch: {epoch}",
                    f"[Train] scratch_lr: {self.scheduler.get_last_lr()[1]:.06f}",
                )

                self.train(epoch, print_freq=1)
                prec1 = self.validate(epoch, print_freq=10)

                is_best = prec1 > self.best_prec1
                self.best_prec1 = max(prec1, self.best_prec1)
                checkpoint_path = "checkpoint_latest.pth"
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_prec1": self.best_prec1,
                }
                self.save_checkpoint(state, is_best, checkpoint_path)
        elif self.mode == "test":
            self.test(save_vis=True, has_gt=False, output_dir="output")
        else:
            raise ValueError(f"Unknown exec mode {self.mode}")

    def train(self, epoch, print_freq):
        self.model.train()

        losses_list = []
        losses_array_list = []
        scores_list = []

        for i in len(self.task_root_list):
            losses_list.append(AverageMeter())
            losses_array_list.append([AverageMeter() for _ in len(self.task_list[i])])
            scores_list.append(AverageMeter())

        for i, in_tar_name_pair in enumerate(self.train_loader):
            grads = {}
            task_loss_list = []
            for index, (input, target) in enumerate(in_tar_name_pair):

                input = input.to(self.device)
                # 可以尝试把target从list变为torch
                target = [target[idx].to(self.device) for idx in range(len(target))]

                output = self.model(input, index)

                assert len(output) == len(target)

                loss_array = [
                    self.criterion(output[idx], target[idx])
                    for idx in range(len(output))
                ]
                loss = sum(loss_array)
                task_loss_list.append(loss)

                losses_list[index].update(loss.item(), input.shape[0])

                for idx in range(len(self.task_list[index])):
                    losses_array_list[index][idx].update(
                        loss_array[idx].item(), input.shape[0]
                    )

                scores_array = []
                if index < 2:
                    for idx in range(len(output)):
                        scores_array.append(mIoU(output[idx], target[idx])[1])
                elif index == 2:
                    for idx in range(len(output)):
                        scores_array.append(mIoU(output[idx], target[idx]))
                else:
                    raise ValueError(f"Unkwon task index")

                scores_list[index].update(np.mean(scores_array), input.shape[0])

                # 清除gradient，或许可以使用self.optimizer.zero_grad()或者用torch.detach()或者require_grad = False
                for cnt in self.model.pretrained.parameters():
                    cnt.grad = None
                self.model.scratch.layer1_rn.weight.grad = None
                self.model.scratch.layer2_rn.weight.grad = None
                self.model.scratch.layer3_rn.weight.grad = None
                self.model.scratch.layer4_rn.weight.grad = None

                loss.backward()

                grads[self.task_root_list[index]] = []
                for cnt in self.model.pretrained.parameters():
                    if cnt.grad is not None:
                        grads[self.task_root_list[index]].append(
                            cnt.grad.clone().detach()
                        )
                grads[self.task_root_list[index]].append(
                    self.model.scratch.layer1_rn.weight.grad.clone().detach()
                )
                grads[self.task_root_list[index]].append(
                    self.model.scratch.layer2_rn.weight.grad.clone().detach()
                )
                grads[self.task_root_list[index]].append(
                    self.model.scratch.layer3_rn.weight.grad.clone().detach()
                )
                grads[self.task_root_list[index]].append(
                    self.model.scratch.layer4_rn.weight.grad.clone().detach()
                )

            if index == 2:
                del input, target
                task_loss_array_new = []
                for index_new, (input_new, target_new, _) in enumerate(
                    in_tar_name_pair
                ):
                    input_new = input_new.to(self.device)
                    target_var_new = [
                        target_new[idx].to(self.device)
                        for idx in range(len(target_new))
                    ]
                    output_new, _, _ = self.model(input_new, index_new)
                    loss_array_new = [
                        self.criterion(output_new[idx], target_var_new[idx])
                        for idx in range(len(output_new))
                    ]
                    local_loss_new = sum(loss_array_new)
                    task_loss_array_new.append(local_loss_new)
                assert len(task_loss_array_new) == 3

                sol, min_norm = MinNormSolver.find_min_norm_element(
                    [grads[cnt] for cnt in self.root_task_list_array]
                )

                self.logger.info(f"scale is: |{sol[0]}|\t|{sol[1]}|\t|{sol[2]}|\t")

                loss_new = 0
                loss_new += sol[0] * task_loss_array_new[0]
                loss_new += sol[1] * task_loss_array_new[1]
                loss_new += sol[2] * task_loss_array_new[2]

                self.optimizer.zero_grad()
                loss_new.backward()
                self.optimizer.step()

                if i % print_freq == 0:
                    losses_info = ""
                    for idx, it in enumerate(self.task_list[index]):
                        loss = losses_array_list[index][idx]
                        losses_info += f"Loss_{it} {loss.val:.4f} ({loss.avg:.4f}) \t"
                        self.writer.add_scalar(
                            "train_task_" + it + "_loss_val",
                            losses_array_list[index][idx].val,
                            global_step=epoch * len(self.train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "train_task_" + it + "_loss_avg",
                            losses_array_list[index][idx].avg,
                            global_step=epoch * len(self.train_loader) + i,
                        )

                    loss = losses_list[index]
                    loss_info = losses_info
                    top1 = scores_list[index]
                    self.logger.info(
                        f"Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t"
                        f"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        f"{loss_info}"
                        f"Score {top1.val:.3f} ({top1.avg:.3f})"
                    )
                    self.writer.add_scalar(
                        "train_" + str(index) + "_losses_val",
                        losses_list[index].val,
                        global_step=epoch * len(self.train_loader) + i,
                    )
                    self.writer.add_scalar(
                        "train_" + str(index) + "_losses_avg",
                        losses_list[index].avg,
                        global_step=epoch * len(self.train_loader) + i,
                    )
                    self.writer.add_scalar(
                        "train_" + str(index) + "_score_val",
                        scores_list[index].val,
                        global_step=epoch * len(self.train_loader) + i,
                    )
                    self.writer.add_scalar(
                        "train_" + str(index) + "_score_avg",
                        scores_list[index].avg,
                        global_step=epoch * len(self.train_loader) + i,
                    )
        for i in range(3):
            self.writer.add_scalar(
                "train_epoch_loss_average", losses_list[index].avg, global_step=epoch
            )
            self.writer.add_scalar(
                "train_epoch_scores_val", scores_list[index].avg, global_step=epoch
            )

    @torch.no_grad()
    def validate(self, epoch, print_freq):
        self.model.eval()

        losses_list = []
        losses_array_list = []
        score_list = []
        score = AverageMeter()

        for i in range(3):
            losses_list.append(AverageMeter())
            losses_array_list.append(
                [AverageMeter() for _ in len(self.task_list_array[i])]
            )
            score_list.append(AverageMeter())

        for i, pairs in enumerate(self.val_loader):
            for index, (input, target) in enumerate(pairs):
                input = input.to(self.device)
                target = [
                    target[idx].to(self.device, non_blocking=True)
                    for idx in range(len(target))
                ]

                output = self.model(input, index)

                loss_array = [
                    self.criterion(output[idx], target[idx])
                    for idx in range(len(output))
                ]
                loss = sum(loss_array)

                losses_list[index].update(loss.item(), input.shape[0])

                for idx in range(len(self.task_list_array[index])):
                    losses_array_list[index][idx].update(
                        loss_array[idx].item(), input.shape[0]
                    )

                scores_array = []

                if index < 2:
                    scores_array = [
                        mIoU(output[idx], target[idx])[1] for idx in range(len(output))
                    ]
                elif index == 2:
                    scores_array = [
                        mIoU(output[idx], target[idx]) for idx in range(len(output))
                    ]
                else:
                    raise ValueError(f"Unkwon task index")

                tmp = np.mean(scores_array)
                if not np.isnan(tmp):
                    score_list[index].update(tmp, input.size(0))

                if i % print_freq == 0:
                    loss = losses_list[index]
                    score = score_list[index]
                    self.writer.add_text(
                        f"Test: [{i}/{len(self.val_loader)}]\t",
                        f"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        f"Score {score.val:.3f} ({score.avg:.3f})",
                    )
            score.update(
                np.nanmean([score_list[0].val, score_list[1].val, score_list[2].val])
            )
            if i % print_freq == 0:
                self.writer.add_text(
                    f"Test: [{i}/{len(self.val_loader)}]\t",
                    f"total score is:{score.val:.3f} ({score.avg:.3f})",
                )

        for idx, item in enumerate(["attribute", "affordance", "segmentation"]):
            self.writer.add_scalar(
                "val_" + item + "_loss_average", losses_list[idx].avg, global_step=epoch
            )
            self.writer.add_scalar(
                "val_" + item + "_score_average", score_list[idx].avg, global_step=epoch
            )

        top1 = score
        self.logger.info(f" * Score {top1.avg:.3f}")
        self.writer.add_scalar("val_score_average", score.avg, global_step=epoch)

        return score.avg

    @torch.no_grad()
    def test(self, save_vis=False, has_gt=False, output_dir=None):
        self.model.eval()

        task_name = ["Attribute", "Affordance", "Segmentation"]

        hist_array_array = []
        hist_array_array_acc = []
        for i in range(len(self.task_root_list)):
            if i < 2:
                num_classes = 2
            elif i == 2:
                num_classes = 40
            else:
                raise ValueError(f"Wrong nunm_classes: {num_classes}")
            hist_array_array.append(
                [
                    np.zeros((num_classes, num_classes))
                    for _ in range(len(self.task_list[i]))
                ]
            )
            hist_array_array_acc.append(
                [
                    np.zeros((num_classes, num_classes))
                    for _ in range(len(self.task_list[i]))
                ]
            )

        for i, in_tar_pair in enumerate(self.test_loader):
            for index, input in enumerate(in_tar_pair):
                if index < 2:
                    num_classes = 2
                    PALETTE = CerberusTrain.AFFORDANCE_PALETTE
                elif index == 2:
                    num_classes = 40
                    PALETTE = CerberusTrain.NYU40_PALETTE
                else:
                    raise ValueError(f"Wrong index: {index}")

                task_list = self.task_list[index]
                iou_compute_cmd = "per_class_iu(hist_array_array[index][idx])"
                if num_classes == 2:
                    iou_compute_cmd = "[" + iou_compute_cmd + "[1]]"

                num_scales = len(self.scales)

                if has_gt:
                    name = input[2]
                    label = input[1]
                else:
                    name = input[1]

                self.writer.add_text("Test", f"File name is {name}")

                h, w = input[0].shape[2:4]
                images = input[-num_scales:]
                outputs = []

                for image in images:
                    image = image.to(self.device)
                    final = self.model(image, index)
                    outputs.append([entity.data for entity in final])

                final = []
                for label_idx in range(len(outputs[0])):
                    tmp_tensor_list = [
                        self.resize_4d_tensor(out[label_idx], w, h) for out in outputs
                    ]
                    final.append(sum(tmp_tensor_list))

                pred = [label_entity.argmax(axis=1) for label_entity in final]

                if save_vis:
                    for idx in range(len(pred)):
                        assert len(name) == 1
                        file_name = (name[0][:-4] + task_list[idx] + ".png",)
                        self.save_output_images(pred[idx], file_name, output_dir)
                        self.save_colorful_images(
                            pred[idx], file_name, output_dir + "_color", PALETTE
                        )
                        if index == 2:
                            gt_name = (name[0][:-4] + task_list[idx] + "_gt.png",)
                            label_mask = label[idx] == 255

                            self.save_colorful_images(
                                (label[idx] - label_mask * 255).numpy(),
                                gt_name,
                                output_dir + "_color",
                                PALETTE,
                            )

                if has_gt:
                    map_score_array = []
                    for idx in range(len(label)):
                        pred[idx] = pred[idx].flatten()
                        label[idx] = label[idx].numpy().flatten()
                        hist_array_array[index][idx] = np.bincount(
                            num_classes * label[idx].astype(int) + pred[idx],
                            minlength=num_classes ** 2,
                        ).reshape(num_classes, num_classes)
                        hist_array_array_acc[index][idx] += hist_array_array[index][idx]

                        map_score_array.append(
                            round(
                                np.nanmean(
                                    [it * 100.0 for it in eval(iou_compute_cmd)]
                                ),
                                2,
                            )
                        )

                        self.writer.add_text(
                            "Test",
                            "===> task${}$ mAP {mAP:.3f}".format(
                                task_list[idx], mAP=map_score_array[idx]
                            ),
                        )

                    if len(map_score_array) > 1:
                        assert len(map_score_array) == len(label)
                        self.writer.add_text(
                            "Test",
                            "===> task${}$ mAP {mAP:.3f}".format(
                                task_name[index],
                                mAP=round(np.nanmean(map_score_array), 2),
                            ),
                        )

        if has_gt:
            ious_array = []
            for index, iter in enumerate(hist_array_array_acc):
                ious = []
                for idx, _ in enumerate(iter):
                    iou_compute_cmd = "per_class_iu(hist_array_array_acc[index][idx])"
                    if index < 2:
                        iou_compute_cmd = "[" + iou_compute_cmd + "[1]]"
                    tmp_result = [i * 100.0 for i in eval(iou_compute_cmd)]
                    ious.append(tmp_result)
                ious_array.append(ious)
            for num, ious in enumerate(ious_array):
                for idx, i in enumerate(ious):
                    self.logger.info(f"task {self.task_list_array[num][idx]}")
                    self.logger.info(" ".join(f"{ii:.3f}" for ii in i))
            for num, ious in enumerate(ious_array):
                self.logger.info(
                    f"task {task_name[num]} : {[np.nanmean(i) for i in ious_array][num]:.2d}"
                )

            return round(np.nanmean([np.nanmean(i) for i in ious_array]), 2)

    def save_checkpoint(self, state, is_best, filename):
        torch.save(state, filename)

        if is_best:
            shutil.copyfile(filename, "model_best.pth")

        if state["epoch"] % 10 == 0:
            history_path = f'checkpoint_{state["epoch"]:03d}.pth'
            shutil.copyfile(filename, history_path)

    def save_output_images(self, predictions, filenames, output_dir):
        """
        Saves a given (B x C x H x W) into an image file.
        If given a mini-batch tensor, will save the tensor as a grid of images.
        """
        for ind in range(len(filenames)):
            im = Image.fromarray(predictions[ind].astype(np.uint8))
            fn = os.path.join(output_dir, filenames[ind][:-4] + ".png")
            out_dir = os.path.split(fn)[0]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            im.save(fn)

    def save_colorful_images(self, predictions, filenames, output_dir, palettes):
        """
        Saves a given (B x C x H x W) into an image file.
        If given a mini-batch tensor, will save the tensor as a grid of images.
        """
        for ind in range(len(filenames)):
            im = Image.fromarray(palettes[predictions[ind].squeeze()])
            fn = os.path.join(output_dir, filenames[ind][:-4] + ".png")
            out_dir = os.path.split(fn)[0]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            im.save(fn)

    def resize_4d_tensor(self, tensor, width, height):
        tensor_cpu = tensor.cpu().numpy()
        if tensor.size(2) == height and tensor.size(3) == width:
            return tensor_cpu
        out_size = (tensor.size(0), tensor.size(1), height, width)
        out = np.empty(out_size, dtype=np.float32)

        def resize_one(i, j):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR
                )
            )

        def resize_channel(j):
            for i in range(tensor.size(0)):
                out[i, j] = np.array(
                    Image.fromarray(tensor_cpu[i, j]).resize(
                        (width, height), Image.BILINEAR
                    )
                )

        workers = [
            threading.Thread(target=resize_channel, args=(j,))
            for j in range(tensor.size(1))
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        return out


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cerberus = CerberusTrain("train.yaml")
    cerberus.exec()

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
from models import CerberusSegmentationModelMultiHead
from utils import mIoU, MinNormSolver


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
                config["data_dir"], "var_affordance", test_tf, scale=self.scales,
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
                self.writer.add_scalars(
                    f"train_lr",
                    {'pretrained': self.scheduler.get_last_lr()[0], 'scratch': self.scheduler.get_last_lr()[1]},
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
                self.save_checkpoint(state, is_best)
        elif self.mode == "test":
            self.test(save_vis=True, have_gt=False, output_dir="output")
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

        for i, task_data_pair in enumerate(self.train_loader):
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
                if task_i < len(self.task_root_list) - 1:
                    for idx in range(len(output)):
                        ious = mIoU(output[idx], target[idx])
                        score.append(ious[1])
                elif task_i == len(self.task_root_list) - 1:
                    for idx in range(len(output)):
                        ious = mIoU(output[idx], target[idx])
                        score.append(np.mean(ious))
                else:
                    raise ValueError(f"Not support task_i: {task_i}")

                score_list[task_i].update(np.mean(score), input.shape[0])

            sol, min_norm = MinNormSolver.find_min_norm_element(
                [grads[i] for i in self.task_root_list]
            )

            self.writer.add_scalars(
                f"train_min_norm_scale",
                {'sol_0': sol[0], 'sol_1': sol[1], 'sol_2': sol[2]},
                global_step=epoch*len(self.train_loader) + i
            )

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
            loss_per_task_list.append([AverageMeter() for _ in range(len(self.task_list[i]))])
            score_list.append(AverageMeter())

        for i, task_data_pair in enumerate(self.val_loader):
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
                # 计算IoU时，为什么不一样？
                if task_i < len(self.task_root_list) - 1:
                    for idx in range(len(output)):
                        ious = mIoU(output[idx], target[idx])
                        score.append(ious[1])
                elif task_i == len(self.task_root_list) - 1:
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
    def test(self, save_vis=False, have_gt=False, output_dir=None):
        self.model.eval()

        hist_array_array = []
        hist_array_array_acc = []
        # need to adjust for specific task group or mimic another code which I have lost their location.
        for i in range(3):
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

        for i, task_data_pair in enumerate(self.test_loader):
            for task_i, input in enumerate(task_data_pair):
                if task_i < 2:
                    num_classes = 2
                    PALETTE = CerberusTrain.AFFORDANCE_PALETTE
                elif task_i == 2:
                    num_classes = 40
                    PALETTE = CerberusTrain.NYU40_PALETTE
                else:
                    raise ValueError(f"Wrong task_i: {task_i}")

                task_list = self.task_list[task_i]
                iou_compute_cmd = "per_class_iu(hist_array_array[task_i][idx])"
                if num_classes == 2:
                    iou_compute_cmd = "[" + iou_compute_cmd + "[1]]"

                num_scales = len(self.scales)

                if have_gt:
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
                    final = self.model(image, task_i)
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
                        if task_i == 2:
                            gt_name = (name[0][:-4] + task_list[idx] + "_gt.png",)
                            label_mask = label[idx] == 255

                            self.save_colorful_images(
                                (label[idx] - label_mask * 255).numpy(),
                                gt_name,
                                output_dir + "_color",
                                PALETTE,
                            )

                if have_gt:
                    map_score_array = []
                    for idx in range(len(label)):
                        pred[idx] = pred[idx].flatten()
                        label[idx] = label[idx].numpy().flatten()
                        hist_array_array[task_i][idx] = np.bincount(
                            num_classes * label[idx].astype(int) + pred[idx],
                            minlength=num_classes ** 2,
                        ).reshape(num_classes, num_classes)
                        hist_array_array_acc[task_i][idx] += hist_array_array[task_i][idx]

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
                                task_name[task_i],
                                mAP=round(np.nanmean(map_score_array), 2),
                            ),
                        )

        if have_gt:
            ious_array = []
            for task_i, iter in enumerate(hist_array_array_acc):
                ious = []
                for idx, _ in enumerate(iter):
                    iou_compute_cmd = "per_class_iu(hist_array_array_acc[task_i][idx])"
                    if task_i < 2:
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

    def save_checkpoint(self, state, is_best, backup_freq=10):
        os.makedirs('model', exist_ok=True)

        file_path = "model/checkpoint_latest.pth"
        torch.save(state, file_path)

        if is_best:
            shutil.copyfile(file_path, "model/model_best.pth")

        if state["epoch"] % backup_freq == 0:
            history_path = f'model/checkpoint_{state["epoch"]:03d}.pth'
            shutil.copyfile(file_path, history_path)

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

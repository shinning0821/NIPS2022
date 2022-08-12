#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
join = os.path.join

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
)
from dataset import data_interface
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from utils import ramps,config
from models.unetr2d import UNETR2D
from losses import kl_loss
print("Successfully imported all requirements!")


def main():
    args = config.return_args()
    monai.config.print_config()

    #%% set training/validation split
    np.random.seed(args.seed)
    model_path = join(args.work_dir, args.model_name + "_3class")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )

    img_path = join(args.data_path, "images")
    gt_path = join(args.data_path, "labels")
    unlable_img_path = join(args.ssl_data_path, "images")
    unlable_gt_path = join(args.ssl_data_path, "labels")

    img_names = sorted(os.listdir(img_path))
    unlable_img_names = sorted(os.listdir(unlable_img_path))
    gt_names = [img_name.split(".")[0] + "_label.png" for img_name in img_names]

    img_num = len(img_names)
    unlable_img_num = len(unlable_img_names)

    val_frac = 0.5
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[0:504]
    val_indices = indices[504:]

    train_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in train_indices
    ]
    val_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in val_indices
    ]
    unlable_files = [
        {"img": join(unlable_img_path, unlable_img_names[i]), "label": join(unlable_gt_path, unlable_img_names[i])}
        for i in np.arange(unlable_img_num)
    ]

    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)},unlable training image num:{len(unlable_files)}"
    )

    #%% create data loaders
    train_loader = data_interface.return_trainloader(args=args,train_files=train_files)
    val_loader = data_interface.return_trainloader(args,val_files)
    unlable_loader = data_interface.return_unlableloader(args=args,unlable_files=unlable_files)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )

    post_pred = Compose(
        [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
    )
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model1 = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=args.num_class,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
        
    model2 = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=args.num_class,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)


    loss_function = monai.losses.DiceCELoss(softmax=True)
    initial_lr = args.initial_lr

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=initial_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = torch.optim.SGD(model1.parameters(), lr=initial_lr,
                           momentum=0.9, weight_decay=0.0001)
                           
    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(model_path)


    iter_num = 0
    for epoch in range(1, max_epochs):
        model1.train()
        model2.train()
        epoch_loss = 0
        for step, (batch_data,unlable_batch) in enumerate(zip(train_loader,unlable_loader), 1):
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)

            # 首先进行有监督部分的损失计算
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            
            labels_onehot = monai.networks.one_hot(
                labels, args.num_class
            )  # (b,cls,256,256)
            loss1 = loss_function(outputs1, labels_onehot)
            loss2 = loss_function(outputs2, labels_onehot)

            unlable_inputs = unlable_batch["img"].to(device)
            # 然后计算无标签数据的一致性损失
            unlable_outputs1 = model1(unlable_inputs)
            unlable_outputs_soft1 = torch.softmax(unlable_outputs1, dim=1)

            unlable_outputs2 = model2(unlable_inputs)
            unlable_outputs_soft2 = torch.softmax(unlable_outputs2, dim=1)

            consistency_weight = 0.1 * ramps.sigmoid_rampup( iter_num // 150, 200)
            r_drop_loss = kl_loss.compute_kl_loss(unlable_outputs_soft1, unlable_outputs_soft2)
            loss = loss1 + loss2 + consistency_weight * r_drop_loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            lr_ = initial_lr * (1.0 - iter_num / 30000) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            epoch_loss += loss.item()
            epoch_len = len(train_files) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        model1.eval()
        model2.eval()
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model1.state_dict(),
            "optimizer_state_dict": optimizer1.state_dict(),
            "loss": epoch_loss_values,
        }
        
        if epoch > 15 and epoch % val_interval == 0:
            model1.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data[
                        "label"
                    ].to(device)
                    val_labels_onehot = monai.networks.one_hot(
                        val_labels, args.num_class
                    )
                    roi_size = (256, 256)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model1
                    )
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_onehot = [
                        post_gt(i) for i in decollate_batch(val_labels_onehot)
                    ]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels_onehot)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(checkpoint, join(model_path, "best_Dice_model_{:.4f}.pth".format(best_metric)))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")
            if (epoch - best_metric_epoch) > epoch_tolerance:
                print(
                    f"validation metric does not improve" #for {epoch_tolerance} epochs! current {epoch= }, {best_metric_epoch=}"
                )
                break

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()
    torch.save(checkpoint, join(model_path, "final_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()

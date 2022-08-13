#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
join = os.path.join

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from dataset import data_interface
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from utils import ramps
from configs import config_dan
from models import valid,discriminator
from models.discriminator import FCDiscriminator
import torch.nn.functional as F
print("Successfully imported all requirements!")


def main():
    args = config_dan.return_args()
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

    val_frac = 0.3
    indices = np.arange(img_num)
    unlable_indices = np.arange(unlable_img_num)
    np.random.shuffle(indices)
    np.random.shuffle(unlable_indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[296:]
    val_indices = indices[:296]

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
        for i in unlable_indices
    ]

    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)},unlable training image num:{len(unlable_files)}"
    )

    #%% create data loaders
    train_loader = data_interface.return_trainloader(args=args,train_files=train_files)
    val_loader = data_interface.return_trainloader(args,val_files)
    unlable_loader = data_interface.return_unlableloader(args=args,unlable_files=unlable_files)

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
        
    DAN = FCDiscriminator(num_classes=args.num_class)
    DAN = DAN.cuda()
    # checkpoint1 = torch.load(join(args.model_path, 'best_Dice_model_0.5865.pth'), map_location=torch.device(device))
    # model1.load_state_dict(checkpoint1['model_state_dict'])

    loss_function = monai.losses.DiceCELoss(softmax=True)
    initial_lr = args.initial_lr / 2

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=initial_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = torch.optim.SGD(DAN.parameters(), lr=initial_lr/100,
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
        epoch_loss = 0
        epoch_dan = 0
        for step, (batch_data,unlable_batch) in enumerate(zip(train_loader,unlable_loader), 1):
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            unlable_inputs = unlable_batch["img"].to(device)

            # 初始化鉴别器的向量
            DAN_Utarget = torch.tensor([0] * args.batch_size).cuda()
            DAN_Ltarget = torch.tensor([1] * args.batch_size).cuda()

            model1.train()
            DAN.eval()
            outputs1 = model1(inputs)
            unlable_outputs1 = model1(unlable_inputs)
            unlable_outputs_soft1 = torch.softmax(unlable_outputs1, dim=1)            
            # 首先进行有监督部分的损失计算
            labels_onehot = monai.networks.one_hot(
                labels, args.num_class
            )  # (b,cls,256,256)
            supervised_loss = loss_function(outputs1, labels_onehot)
            DAN_outputs = DAN(
                unlable_outputs_soft1, unlable_inputs)

            consistency_weight = 0.1 * ramps.sigmoid_rampup( iter_num // 150, 200)
            # print(DAN_outputs.shape)
            # print(DAN_Ltarget.shape)
            consistency_loss = F.cross_entropy(
                DAN_outputs, DAN_Ltarget)
            loss = supervised_loss + consistency_weight * consistency_loss            
        
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            model1.eval()
            DAN.train()

            with torch.no_grad():
                outputs = model1(inputs)
                outputs_soft = torch.softmax(outputs, dim=1)
                unlable_outputs1 = model1(unlable_inputs)
                unlable_outputs_soft1 = torch.softmax(unlable_outputs1, dim=1)     

            DAN_outputs1 = DAN(outputs_soft, inputs)
            DAN_outputs2 = DAN(unlable_outputs_soft1, unlable_inputs)
            # print(DAN_outputs1.shape)
            # print(DAN_outputs2.shape)
            DAN_loss1 = F.cross_entropy(DAN_outputs1, DAN_Ltarget)
            DAN_loss2 = F.cross_entropy(DAN_outputs2, DAN_Utarget)

            DAN_loss = DAN_loss1 + DAN_loss2
            optimizer2.zero_grad()
            DAN_loss.backward()
            optimizer2.step()

            iter_num += 1
            lr_ = initial_lr * (1.0 - iter_num / 30000) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            epoch_loss += loss.item()
            epoch_dan += DAN_loss.item()
            epoch_len = len(train_files) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        DAN_loss /= step
        epoch_loss_values.append(epoch_loss)
        model1.eval()
        print(f"epoch {epoch} average loss: {epoch_loss:.4f},dan loss:{DAN_loss:.4f}")
        checkpoint_model1 = {
            "epoch": epoch,
            "model_state_dict": model1.state_dict(),
            "optimizer_state_dict": optimizer1.state_dict(),
            "loss": epoch_loss_values,
        }

        
        if epoch > args.start_epoch and epoch % val_interval == 0:
            model1.eval()
            with torch.no_grad():
                metric,val_images,val_labels,val_outputs = valid.compute_DiceMetric(args,device,val_loader,model1)
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(checkpoint_model1, join(model_path, "best_Dice_model1_{:.4f}.pth".format(best_metric)))
                   
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
    torch.save(checkpoint_model1, join(model_path, "final_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()

from __future__ import division

from models import *
# from utils.utils import *
# from utils.coco import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from utils.str2bool import str2bool

from config.load_classes import load_classes

from config.parse_config import *
from models.yolo3 import weights_init_normal
from models.yolo3 import Darknet

import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser(description='YOLO3 Detector Training With Pytorch')
    # train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset_type', default='VOC', choices=['VOC', 'COCO', 'KAIST'],type=str, help='VOC, COCO or KAIST (requires image_set )')
    parser.add_argument('--dataset_root', default=None, help='Dataset root directory path') #TODO VPY: should be in kaist.data
    parser.add_argument('--image_set', default=None, help='[KAIST] Imageset')
    parser.add_argument('--basenet', default=None, help='Pretrained base model') #TODO VPY: should be in kaist.data
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training') #TODO VPY: should be in kaist.data
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
    parser.add_argument('--save_frequency', default=5000, type=int, help='Frequency to save model [default: 5000 iters]')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading') #TODO 4
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
    # parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    # parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    # parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss visualization')
    parser.add_argument('--save_folder', default='checkpoints', help='Directory for saving checkpoint models')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: lwir inverted][...]') #TODO VPY update when required
    # parser.add_argument('--show_dataset', default=False, type=str2bool, help='Show every image used ?')
    parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    args = parser.parse_args()
    return args


def main(args):
    cuda = torch.cuda.is_available() and args.cuda

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    classes = load_classes(args.class_path)

    # Get data configuration
    data_config = parse_data_config(args.data_config_path)
    train_path = data_config["train"]
    iters = int(data_config["iters"])

    # Get hyper parameters
    hyperparams = parse_model_config(args.model_config_path)[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])
    model = Darknet(args.model_config_path)

    # Initiate model
    if args.basenet is not None:
        model.load_weights(args.basenet)
    else:
        model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    model.train()

    # Get dataloader
    if args.dataset_type == "COCO":
        from data.coco_list import ListDataset
        dataset = ListDataset(train_path)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)
    elif args.dataset_type == "KAIST":
        from data.kaist import KAISTDetection, KAISTAnnotationTransform
        from augmentations.YOLOaugmentations import YOLOaugmentation
        # dataset = KAISTDetection(root=data_config["dataset_root"], image_set=data_config["train"], transform=SSDAugmentation(args.img_size), image_fusion=int(data_config["image_fusion"]))

        kaist_root = data_config["dataset_root"]
        image_set = data_config["train"]
        image_fusion = int(data_config["image_fusion"])

        dataset = KAISTDetection(root=kaist_root, image_set=image_set, transform=YOLOaugmentation(args.img_size), image_fusion=image_fusion, output_format="YOLO", target_transform=KAISTAnnotationTransform(output_format="YOLO"))
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False, #True,
            num_workers=args.num_workers
            # collate_fn=torch.utils.data.dataloader.default_collate,
            # pin_memory=True
            )
        print("VPY: we might want to change collate_fn ?") #TODO VPY
    else:
        raise NotImplementedError

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(iters):
        for batch_i, (_, imgs, targets, _, _) in enumerate(dataloader):
            #print("In learning loop:\nbatch_i: {}\ntmp: {}\n imgs.shape: {}\n targets.shape: {}\n\n".format(batch_i, tmp, imgs.shape, targets.shape))
            imgs = Variable(imgs.type(Tensor))

            import cv2
            test_img = np.transpose(imgs[0].cpu().numpy(), (1, 2, 0))
            test_img = test_img[:,:,(2,1,0)]
            cv2.imshow('image', test_img)
            cv2.waitKey(500)

            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()

            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    iters,
                    batch_i,
                    len(dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )

            model.seen += imgs.size(0)

        if epoch % args.checkpoint_interval == 0:
            model.save_weights("%s/%d.weights" % (args.checkpoint_dir, epoch))

if __name__ == "__main__":
    args = arg_parser()
    main(args)
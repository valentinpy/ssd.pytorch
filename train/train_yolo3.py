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
from utils.visualization import vis_plot

def arg_parser():
    parser = argparse.ArgumentParser(description='YOLO3 Detector Training With Pytorch')
    parser.add_argument('--image_set', default=None, help='[KAIST] Imageset')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
    parser.add_argument('--save_folder', default='checkpoints', help='Directory for saving checkpoint models')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: lwir inverted][...]') #TODO VPY update when required
    parser.add_argument('--show_dataset', default=False, type=str2bool, help='Show every image used ?')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    args = parser.parse_args()
    return args


def main(args):
    args = vars(arg_parser())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}
    del config

    if args['visdom']:
        viz = vis_plot()
    else:
        viz = None

    cuda = torch.cuda.is_available() and args['cuda']

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    classes = load_classes(args['names'])

    # Get data configuration
    # data_config = parse_data_config(args.data_config_path)
    iters = args['yolo_max_iters']

    # Get hyper parameters
    hyperparams = parse_model_config(args['yolo_model_config_path'])[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])
    model = Darknet(args['yolo_model_config_path'])

    # Initiate model
    if args['yolo_initial_weights'] is not None:
        model.load_weights(args['yolo_initial_weights'])
    else:
        model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    model.train()

    # Get dataloader
    if args['name'] == "COCO":
        from data.coco_list import ListDataset, detection_collate_COCO_YOLO
        train_path = args["train_set"]
        dataset = ListDataset(train_path)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(hyperparams['batch']),
            shuffle=False,
            num_workers=args['num_workers'],
            collate_fn=detection_collate_COCO_YOLO)
    elif args['name'] == "KAIST":
        from data.kaist import KAISTDetection, KAISTAnnotationTransform, detection_collate_KAIST_YOLO
        from augmentations.YOLOaugmentations import YOLOaugmentation
        # dataset = KAISTDetection(root=data_config["dataset_root"], image_set=data_config["train"], transform=SSDAugmentation(args.img_size), image_fusion=int(data_config["image_fusion"]))

        kaist_root = args["dataset_root"]
        image_set = args["image_set"]
        image_fusion = args["image_fusion"]

        dataset = KAISTDetection(root=kaist_root, image_set=image_set, transform=YOLOaugmentation(args['yolo_img_size']), image_fusion=image_fusion, output_format="YOLO", target_transform=KAISTAnnotationTransform(output_format="YOLO"))
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(hyperparams['batch']),
            shuffle=False, #True,
            num_workers=args['num_workers'],
            collate_fn=detection_collate_KAIST_YOLO
            # collate_fn=torch.utils.data.dataloader.default_collate,
            # pin_memory=True
            )
    else:
        raise NotImplementedError

    if args['visdom']:
        vis_title = 'YOLO3.PyTorch on ' + dataset.name + ' | lr: ' + "n/d"
        vis_legend = ['Model loss', 'n/d', 'Total Loss']
        iter_plot = viz.create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        # epoch_plot = viz.create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)


    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    iteration_total = 0
    for epoch in range(iters):
        for batch_i, (imgs, targets) in enumerate(dataloader):
            #print("In learning loop:\nbatch_i: {}\ntmp: {}\n imgs.shape: {}\n targets.shape: {}\n\n".format(batch_i, tmp, imgs.shape, targets.shape))
            imgs = Variable(imgs.type(Tensor))

            import cv2
            test_img = np.transpose(imgs[0].cpu().numpy(), (1, 2, 0))
            test_img = test_img[:,:,(2,1,0)]
            # cv2.imshow('image', test_img)
            # cv2.waitKey(500)

            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()

            if iteration_total % 100 == 0:
                print(
                    "[%s] [Epoch %d/%d, Batch %d/%d, total %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                    % (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch,
                        iters,
                        batch_i,
                        len(dataloader),
                        epoch * len(dataloader) + batch_i,
                        iters*len(dataloader),
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

            if ((iteration_total) % (args['save_frequency']) == 0 ) and (epoch*len(dataloader)+ batch_i) > 0:
                model.save_weights("%s/YOLO3_%d.weights" % (args['save_folder'], epoch*len(dataloader)+batch_i))
                print("Weights saved for epoch {}/{}, batch {}/{}".format(epoch, iters, batch_i, len(dataloader)))

            if args['visdom']:
                # viz.update_vis_plot(iteration_total, loss_l.data.item(), loss_c.data.item(), iter_plot, epoch_plot, 'append')
                viz.update_vis_plot(iteration_total, loss.item(), 0, iter_plot, None, 'append')
            iteration_total += 1

if __name__ == "__main__":
    args = arg_parser()
    main(args)
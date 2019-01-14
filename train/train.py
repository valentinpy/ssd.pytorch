import argparse
from utils.str2bool import str2bool
from config.parse_config import *
from config.load_classes import load_classes
import torch
import os
import sys
import glob
import time
import datetime

from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.visualization import vis_plot
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.nn as nn


def arg_parser():
    parser = argparse.ArgumentParser(description='MobileNet2_SSD/YOLO3 Detector Training With Pytorch')
    parser.add_argument('--image_set', default=None, help='[KAIST] Imageset')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
    parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss visualization')
    parser.add_argument('--save_folder', default='checkpoints', help='Directory for saving checkpoint models')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: lwir inverted][...]') #TODO VPY update when required
    parser.add_argument('--show_dataset', default=False, type=str2bool, help='Show every image used ?')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    parser.add_argument("--model", type=str, default=None, help="Model to train, either 'VGG_SSD','MOBILENET2_SSD' or 'YOLO'")

    args = vars(parser.parse_args())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}

    if args['model'] == "VGG_SSD":
        args = {key.replace("vgg_", ""): value for key, value in args.items() if (not key.startswith("mobilenet")) and (not key.startswith("yolo"))}
    elif args['model'] == "MOBILENET2_SSD":
        args = {key.replace("mobilenet_", ""): value for key, value in args.items() if (not key.startswith("vgg")) and (not key.startswith("yolo"))}
    elif args["model"] == "YOLO":
        args = {key: value for key, value in args.items() if (not key.startswith("vgg")) and (not key.startswith("mobilenet"))}


    return args

def train(args, viz = None):
    model_name = args["model"]
    dataset_name = args["name"]
    imageset_name = args["image_set"].split("/")[-1].split(".")[0]
    cuda = torch.cuda.is_available() and args['cuda']

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # get hyperparameters
    if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
        epochs = args['ssd_iters']
        learning_rate = args['ssd_lr']
        gamma = args['ssd_gamma']
        batch_size = args["ssd_batch_size"]
        ssd_min_dim = args["ssd_min_dim"]
    elif model_name == "YOLO":
        epochs = args['yolo_max_iters']
        hyperparams = parse_model_config(args['yolo_model_config_path'])[0]
        learning_rate = float(hyperparams["learning_rate"])
        batch_size = int(hyperparams["batch"])

    # get augmentation function
    if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
        from data.kaist import detection_collate_KAIST_SSD
        from augmentations.SSDaugmentations import SSDAugmentation
    elif model_name == "YOLO":
        from data.kaist import detection_collate_KAIST_YOLO
        from augmentations.YOLOaugmentations import YOLOaugmentation

    # load dataset
    print('Preparing the dataset...')
    if dataset_name == 'VOC':
        from data.voc0712 import VOCDetection, detection_collate_VOC
        dataset = VOCDetection(root=args['dataset_root'], transform=SSDAugmentation(ssd_min_dim))
        data_loader = DataLoader(dataset, batch_size, num_workers=args['num_workers'], shuffle=True, collate_fn=detection_collate_VOC, pin_memory=True)
    elif dataset_name == 'KAIST':
        from data.kaist import KAISTDetection, KAISTAnnotationTransform
        kaist_root = args["dataset_root"]
        image_set = args["image_set"]
        image_fusion = args["image_fusion"]

        if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
            dataset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=SSDAugmentation(ssd_min_dim), image_fusion=args['image_fusion'], target_transform=KAISTAnnotationTransform(output_format="SSD"))
            data_loader = DataLoader(dataset, batch_size, num_workers=args['num_workers'], shuffle=True, collate_fn=detection_collate_KAIST_SSD, pin_memory=True)
        elif model_name == "YOLO":
            dataset = KAISTDetection(root=kaist_root, image_set=image_set, transform=YOLOaugmentation(args['yolo_img_size']), image_fusion=image_fusion, target_transform=KAISTAnnotationTransform(output_format="YOLO"))
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,  # True,
                num_workers=args['num_workers'],
                collate_fn=detection_collate_KAIST_YOLO)

    elif dataset_name == "COCO":
        from data.coco_list import ListDataset, detection_collate_COCO_YOLO
        dataset = ListDataset(args["train_set"])
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args['num_workers'],
            collate_fn=detection_collate_COCO_YOLO)

    # build net
    if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
        from models.vgg16_ssd import build_vgg_ssd
        from models.mobilenet2_ssd import build_mobilenet_ssd
        from layers.modules import MultiBoxLoss
        if model_name == "VGG_SSD":
            ssd_net = build_vgg_ssd('train', args['ssd_min_dim'], args['classes'], cfg=args)
        else:
            ssd_net = build_mobilenet_ssd('train', args['ssd_min_dim'], args['classes'], cfg=args)
        model = ssd_net

        if cuda:
            # model = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        if args['resume']:
            print('Resuming training, loading {}...'.format(args['resume']))
            ssd_net.load_weights(args['resume'])
            raise NotImplementedError
        else:
            weights = torch.load(args['ssd_initial_weights'])

            if image_fusion > 2: # if image is fused
                weights['0.weight'] = torch.stack((weights['0.weight'][:, 0, :, :], weights['0.weight'][:, 1, :, :], weights['0.weight'][:, 2, :, :], weights['0.weight'][:, 0, :, :]), dim=1)

            print('Loading base network...')
            ssd_net.basenet.load_state_dict(weights)

            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)

        if cuda:
            model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args['ssd_momentum'], weight_decay=args['ssd_weight_decay'])
        criterion = MultiBoxLoss(args['classes'], 0.5, 3, args['cuda'], args['ssd_variance'])


    elif model_name == "YOLO":
        from models.yolo3 import Darknet
        from models.yolo3 import weights_init_normal

        model = Darknet(args['yolo_model_config_path'])

        if args['yolo_initial_weights'] is not None:
            model.load_weights(args['yolo_initial_weights'])
        else:
            model.apply(weights_init_normal)

        if cuda:
            model = model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # set train state
    model.train()


    # create plots
    if args['visdom']:
        vis_title = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S - ")
        vis_title += str(model_name + ".PyTorch on '" + args["name"] + ' | lr: ' + str(learning_rate))
        vis_legend = ['Model loss', 'n/d', 'Total Loss'] if model_name == "YOLO" else ['loc loss', 'conf loss', 'Total Loss']
        iter_plot = viz.create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)

    epoch_size = len(dataset) // batch_size

    print("Training model: '{}' on dataset: '{}'".format(model_name, dataset_name))
    print('Dataset length: {}'.format(len(dataset)))
    print('Epoch size: {}, batch size: {}'.format(epoch_size, batch_size))

    step_index = 0
    loc_loss = 0
    conf_loss = 0

    iteration_total = 0
    for epoch in range(epochs):
        for batch_i, (imgs, targets) in enumerate(data_loader):
            t0 = time.time()

            # load train data
            if model_name == "YOLO":
                imgs = Variable(imgs.type(Tensor))
                targets = Variable(targets.type(Tensor), requires_grad=False)
                optimizer.zero_grad()
                loss = model(imgs, targets)
                loss.backward()
                optimizer.step()

            elif model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
                lr_steps = args['ssd_lr_steps']
                if iteration_total in lr_steps:
                    step_index += 1
                    adjust_learning_rate(optimizer, gamma, step_index, learning_rate)
                if cuda:
                    images = imgs.cuda()
                    targets = [ann.cuda() for ann in targets]
                else:
                    raise NotImplementedError

                # forward
                out = model(images)

                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                # loc_loss += loss_l.data.item()
                # conf_loss += loss_c.data.item()
                loc_loss = loss_l.data.item()
                conf_loss = loss_c.data.item()

            batch_time = time.time() - t0

            # log progress
            if iteration_total % 50 == 0:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
                    loc_loss = loss_l
                    conf_loss = loss_c
                    tot_loss = loss_l + loss_c
                    print(
                        "[%s] [Iteration: %d/%d] [Batch %d/%d, Epoch %d/%d, total images %d/%d] [Losses: loc: %f, conf %f, total %f] [batch: %.3fms] " %
                        (now,
                         iteration_total, epoch_size * epochs,
                         batch_i, len(data_loader),
                         epoch, epochs,
                         (iteration_total * batch_size), epochs * len(data_loader) * batch_size,
                         loc_loss, conf_loss, tot_loss,
                         batch_time * 1000)
                        )

                elif model_name == "YOLO":
                    conf_loss = model.losses["conf"]
                    cls_loss = model.losses["cls"]
                    tot_loss = loss.item()
                    recall = model.losses["recall"]
                    precision = model.losses["precision"]

                    print("[%s] [Iteration: %d/%d] [Batch %d/%d, Epoch %d/%d, total images %d/%d] [Losses: conf %f, cls %f, total %f, recall: %.5f, precision: %.5f] [batch: %.3fms] "%
                          ( now,
                            iteration_total, epoch_size * epochs,
                            batch_i, len(data_loader),
                            epoch, epochs,
                            (iteration_total * batch_size), epochs * len(data_loader) * batch_size,
                            conf_loss, cls_loss, tot_loss, recall, precision,
                            batch_time * 1000)
                    )

            model.seen += imgs.size(0)

            # save progress
            if ((iteration_total) % (args['save_frequency']) == 0) and iteration_total > 0:
                saved_model_name = "%s/%s__%s__%s__fusion-%d__iter-%d.weights" % (args['save_folder'], model_name, dataset_name, imageset_name, image_fusion, epoch * len(data_loader) + batch_i)
                model.save_weights(saved_model_name)
                # torch.save(ssd_net.state_dict(), model_name)
                print("Weights saved for epoch {}/{}, batch {}/{}".format(epoch, epochs, batch_i, len(data_loader)))

            plot_loss1 = loc_loss if model_name in{"VGG_SSD", "MOBILENET2_SSD"} else loss.item()
            plot_loss2 = conf_loss if model_name in{"VGG_SSD", "MOBILENET2_SSD"} else 0

            if args['visdom']:
                viz.update_vis_plot(iteration_total, plot_loss1, plot_loss2, iter_plot, None, 'append')
            iteration_total += 1

    print("Finished training at {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

def adjust_learning_rate(optimizer, gamma, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    #prepare for CUDA
    cuda = torch.cuda.is_available() and args['cuda']
    if torch.cuda.is_available():
        if cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("WARNING: It looks like you have a CUDA device, but aren't using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # prepare visualization
    if args['visdom']:
        viz = vis_plot()
    else:
        viz = None

    # train
    train(args, viz)


def check_args(args):
    #prepare output folder
    if not os.path.exists(args['save_folder']):
        os.makedirs(args['save_folder'])
    if len(os.listdir(args['save_folder'])) != 0:
        print("Save directory is not empty! : {}".format(args['save_folder']))
        if "tmp" in args['save_folder']:
            print("but as the save folder contains 'tmp', we remove old data:")
            files = glob.glob(args['save_folder'] + '/*')
            for f in files:
                if f.endswith('.weights') or f.endswith('.pth'):
                    print("rm {}".format(f))
                    os.remove(f)
        else:
            print("not a tmp folder, you must fix it yourself!")
            sys.exit(-1)

    if args['save_frequency'] < 0:
        print("save frequency must be > 0")
        sys.exit(-1)

    if args['name'] == "KAIST":
        if args['image_fusion'] == -1:
            print("image fusion must be specified")
            sys.exit(-1)
        print("Image fusion value: {}".format(args['image_fusion']))

    if args['dataset_root'] == None:
        print('Must specify dataset_root')
        sys.exit(-1)

    if not os.path.exists(args['dataset_root']):
        print('Must specify *existing* dataset_root')
        sys.exit(-1)

    if args["model"] in {"VGG_SSD", "MOBILENET2_SSD"}:
        if args["name"] not in {"VOC", "KAIST"}:
            print("Dataset {} not supported with model {}".format(args["name"], args["model"]))
            sys.exit(-1)

    elif args["model"] == "YOLO":
        if args["name"] not in {"COCO", "KAIST"}:
            print("Dataset {} not supported with model {}".format(args["name"], args["model"]))
            sys.exit(-1)
    else:
        print("Model {} is not supported".format(args["model"]))
        sys.exit(-1)


if __name__ == "__main__":
    args = arg_parser()
    check_args(args)
    main(args)

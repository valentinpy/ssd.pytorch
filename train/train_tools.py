import datetime
from config.parse_config import *
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from models.yolo3 import Darknet
from models.yolo3 import weights_init_normal

from models.vgg16_ssd import build_vgg_ssd
from models.mobilenet2_ssd import build_mobilenet_ssd
from layers.modules import MultiBoxLoss


def build_training_net(args):
    # get parameters from args
    model_name = args["model"]
    cuda = torch.cuda.is_available() and args['cuda']
    image_fusion = args["image_fusion"]

    # init return variables
    model = None
    criterion = None
    optimizer = None

    # get learning rate
    if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
        learning_rate = args['ssd_lr']
    elif model_name == "YOLO":
        learning_rate = args['yolo_lr']
    else:
        raise NotImplementedError

    # build net
    if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
        if model_name == "VGG_SSD":
            ssd_net = build_vgg_ssd('train', args['ssd_min_dim'], args['classes'], cfg=args)
        else:
            ssd_net = build_mobilenet_ssd('train', args['ssd_min_dim'], args['classes'], cfg=args)
        model = ssd_net

        if cuda:
            cudnn.benchmark = True

        if "resume" in args:
            print('Resuming training, loading {}...'.format(args['resume']))
            ssd_net.load_weights(args['resume'])
            raise NotImplementedError
        else:
            weights = torch.load(args['ssd_initial_weights'])

            if image_fusion > 2 and model_name == "VGG_SSD":  # if image is fused
                weights['0.weight'] = torch.stack((weights['0.weight'][:, 0, :, :], weights['0.weight'][:, 1, :, :], weights['0.weight'][:, 2, :, :],
                                                   weights['0.weight'][:, 0, :, :]), dim=1)

            if image_fusion > 2 and model_name == "MOBILENET2_SSD":
                weights['0.0.weight'] = torch.stack((weights['0.0.weight'][:, 0, :, :], weights['0.0.weight'][:, 1, :, :], weights['0.0.weight'][:, 2, :, :],
                                                   weights['0.0.weight'][:, 0, :, :]), dim=1)

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
        model = Darknet(args['yolo_model_config_path'])

        if args['yolo_initial_weights'] is not None:
            model.load_weights(args['yolo_initial_weights'])
        else:
            model.apply(weights_init_normal)

        if cuda:
            model = model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    return model, criterion, optimizer


def print_learning_progress_xxx_YOLO(batch_i, batch_size, batch_time, data_loader, epoch, epoch_size, epochs, iteration_total, model, loss):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Ex: [2019-01-14 16:22:39]
    conf_loss = model.losses["conf"]
    cls_loss = model.losses["cls"]
    tot_loss = loss.item()
    recall = model.losses["recall"]
    precision = model.losses["precision"]
    print(
        "[%s] [Iteration: %d/%d] [Batch %d/%d, Epoch %d/%d, total images %d/%d] [Losses: conf %f, cls %f, total %f, recall: %.5f, precision: %.5f] [batch: %.3fms] " %
        (now,
         iteration_total, epoch_size * epochs,
         batch_i, len(data_loader),
         epoch, epochs,
         (iteration_total * batch_size), epochs * len(data_loader) * batch_size,
         conf_loss, cls_loss, tot_loss, recall, precision,
         batch_time * 1000)
        )


def print_learning_progress_xxx_SSD(batch_i, batch_size, batch_time, conf_loss, data_loader, epoch, epoch_size, epochs, iteration_total, loc_loss):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Ex: [2019-01-14 16:22:39]
    tot_loss = loc_loss + conf_loss
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

def get_hyperparams(args):
    model_name = args["model"]
    if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
        epochs = args['ssd_iters']
        learning_rate = args['ssd_lr']
        gamma = args['ssd_gamma']
        batch_size = args["ssd_batch_size"]
    elif model_name == "YOLO":
        epochs = args['yolo_max_iters']
        hyperparams = parse_yolo_model_config(args['yolo_model_config_path'])[0]
        learning_rate = float(hyperparams["learning_rate"])
        batch_size = int(hyperparams["batch"])
        gamma = None
    else:
        raise NotImplementedError
    return batch_size, epochs, gamma, learning_rate


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

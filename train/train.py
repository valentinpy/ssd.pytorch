from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from models.ssd import build_ssd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
import datetime
import cv2
from utils.misc import str2bool

from data.kaist import compute_KAIST_dataset_mean


def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    # train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'KAIST'],
                        type=str, help='VOC, COCO or KAIST (requires image_set )')
    parser.add_argument('--dataset_root', default=VOC_ROOT,
                        help='Dataset root directory path')
    parser.add_argument('--image_set', default=None,
                        help='[KAIST] Imageset')
    parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--save_frequency', default=5000, type=int,
                        help='Frequency to save model [default: 5000 iters]')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--visdom', default=False, type=str2bool,
                        help='Use visdom for loss visualization')
    parser.add_argument('--save_folder', default='checkpoints',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--image_fusion', default=-1, type=int,
                        help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [...]') #TODO VPY update when required
    parser.add_argument('--show_dataset', default=False, type=str2bool,
                        help='Show every image used ?')
    args = parser.parse_args()

    return args


def show_dataset(dataset_root, image_set, image_fusion):
    print("showing dataset")

    fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
    writer = cv2.VideoWriter('dataset.avi', fourcc, 30, (2048, 1024), isColor=True)

    dataset = KAISTDetection(root=dataset_root, image_set=image_set, transform=None, image_fusion=image_fusion)
    data_loader = data.DataLoader(dataset, 1, num_workers=1, shuffle=True, collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    i = 0
    for i in range(len(dataset)):  # while True:# iteration in range(args.start_iter, cfg['max_iter']):
        try:
            # load train data
            image, annotations = next(batch_iterator)
            image = image[0].permute(1, 2, 0).numpy().astype(np.uint8).copy()
            width = image.shape[1]
            height = image.shape[0]

            for annotation in annotations:
                annotation = (annotation.numpy())[0]
                cv2.rectangle(image,
                              (int(annotation[0] * width), int(annotation[1] * height)),
                              (int(annotation[2] * width), int(annotation[3] * height)),
                              (0, 255, 0),
                              1
                              )
            cv2.imshow('decoded image', image)
            cv2.waitKey(1)
            writer.write(image)
            print(image.mean(axis=(0, 1)))
            time.sleep(1)
        except StopIteration:
            break
    writer.release()

def compute_VOC_dataset_mean(dataset_root, image_set):
    print("compute images mean")
    images_mean = np.zeros((3), dtype=np.float64)  # [0,0,0]
    #
    # # create batch iterator
    dataset_mean = VOCDetection(root=dataset_root, transform=None)
    data_loader_mean = data.DataLoader(dataset_mean, 1, num_workers=1, shuffle=False, collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader_mean)
    i = 0
    for i in range(len(dataset_mean)):  # while True:# iteration in range(args.start_iter, cfg['max_iter']):
        # for i in range(100):
        #     print("Debug: not all data!!!!!")
        try:
            # load train data
            image, _ = next(batch_iterator)
            images_mean += image[0].permute(1, 2, 0).numpy().mean(axis=(0, 1))
        except StopIteration:
            break
    #         batch_iterator = iter(data_loader)
    #         images, targets = next(batch_iterator)
    # print(i)
    # print("pre image mean is: {}".format(image_mean))
    images_mean = images_mean / i
    print("image mean is: {}".format(images_mean))
    return images_mean

def train(args):
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                print('Must specify dataset_root if specifying dataset')
                sys.exit(-1)
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], VOC_MEANS)) # TODO VPY VOC MEANS ?!
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            print('Must specify dataset if specifying dataset_root')
            sys.exit(-1)
        cfg = voc
        dataset_mean = compute_VOC_dataset_mean(args.dataset_root, args.image_set)

        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], VOC_MEANS))

    elif args.dataset == 'KAIST':
        if args.dataset_root == COCO_ROOT:
            print('Must specify dataset if specifying dataset_root')
            sys.exit(-1)
        if (args.image_set is None) or (not os.path.exists(args.image_set)):
            print("When using kaist, image set must be defined to a valid file: {}".format(args.image_set))
        cfg = kaist

        if args.show_dataset == True:
            show_dataset(args.dataset_root, args.image_set, args.image_fusion)
        # dataset_mean = compute_KAIST_dataset_mean(args.dataset_root, args.image_set)
        #dataset = KAISTDetection(root=args.dataset_root, image_set=args.image_set, transform=SSDAugmentation(cfg['min_dim'], tuple(dataset_mean)))
        dataset = KAISTDetection(root=args.dataset_root, image_set=args.image_set, transform=SSDAugmentation(cfg['min_dim'], VOC_MEANS), image_fusion=args.image_fusion)
    else:
        print("No dataset specified")
        sys.exit(-1)

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], dataset=args.dataset)
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, 3, args.cuda)

    net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size

    print('Dataset length: {}'.format(len(dataset)))
    print('Epoch size: {}'.format(epoch_size))

    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name + '| lr: ' + repr(args.lr)
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, 'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        t0 = time.time()
        try:
            # load train data
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        data_time = time.time() - t0

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        batch_time = time.time() - t0
        loc_loss += loss_l.data.item()
        conf_loss += loss_c.data.item()

        if iteration % 10 == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " | " + args.dataset + ': iter ' + repr(iteration) + ' || lr: %g || Loss: %.4f ||' %
                  (optimizer.param_groups[0]['lr'], loss.data.item()), end=' ')
            print('data: %.3fms, batch: %.3fs' % (data_time*1000, batch_time))

        if args.visdom:
            update_vis_plot(iteration, loss_l.data.item(), loss_c.data.item(), iter_plot, epoch_plot, 'append')

        # save model at a given frequency during training
        if iteration != 0 and iteration % args.save_frequency == 0:
            print('Saving state, iter:', iteration)
            model_name = os.path.join(args.save_folder, 'ssd300_' + args.dataset + '_' + repr(iteration) + '.pth')
            latest_name = os.path.join(args.save_folder, 'ssd300_' + args.dataset + '_latest.pth')
            torch.save(ssd_net.state_dict(), model_name)
            os.system('ln -sf {} {}'.format(model_name, latest_name))

    # save model at the end of the training
    torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, args.dataset, args.dataset + '.pth'))
    latest_name = os.path.join(args.save_folder, 'ssd300_' + args.dataset + '_latest.pth')
    os.system('ln -sf {} {}'.format(model_name, latest_name))

    print("Finished. Model is saved at {}".format(latest_name))
    return


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type, epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    args = arg_parser()

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if not os.path.exists(os.path.join(args.save_folder, args.dataset)):
        os.makedirs(os.path.join(args.save_folder, args.dataset))

    if len(os.listdir(os.path.join(args.save_folder, args.dataset))) != 0:
        print("Save directory is not empty!")
        sys.exit(-1)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    if args.save_frequency < 0:
        print("save frequency must be > 0")
        sys.exit(-1)

    if args.dataset == "KAIST":
        if args.image_fusion == -1:
            print("image fusion must be specified")
            sys.exit(-1)
        print("Image fusion value: {}".format(args.image_fusion))
    train(args)

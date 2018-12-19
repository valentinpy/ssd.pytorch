from data.kaist import *
from data.voc0712 import *
from augmentations.SSDaugmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from models.vgg16_ssd import build_ssd
import os
import sys
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
import datetime
import cv2
from utils.str2bool import str2bool
from config.parse_config import *
from utils.visualization import vis_plot


def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--image_set', default=None, help='[KAIST] Imageset')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
    parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss visualization')
    parser.add_argument('--save_folder', default='checkpoints', help='Directory for saving checkpoint models')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: lwir inverted][...]') #TODO VPY update when required
    parser.add_argument('--show_dataset', default=False, type=str2bool, help='Show every image used ?')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    args = parser.parse_args()

    return args


def show_dataset(dataset_root, image_set, image_fusion):
    print("showing dataset")

    fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
    writer = cv2.VideoWriter('dataset.avi', fourcc, 30, (2048, 1024), isColor=True)

    dataset = KAISTDetection(root=dataset_root, image_set=image_set, transform=None, image_fusion=image_fusion)
    data_loader = data.DataLoader(dataset, 1, num_workers=1, shuffle=True, collate_fn=detection_collate_KAIST_SSD, pin_memory=True)
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
    data_loader_mean = data.DataLoader(dataset_mean, 1, num_workers=1, shuffle=False, collate_fn=detection_collate_VOC, pin_memory=True)
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

def train(args, viz = None):
    # if args.dataset == 'COCO':
    #     cfg = coco
    #     dataset = COCODetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], VOC_MEANS)) # TODO VPY VOC MEANS ?!
    if args['name'] == 'VOC':
        dataset = VOCDetection(root=args['dataset_root'], transform=SSDAugmentation(args['ssd_min_dim']))

    elif args['name'] == 'KAIST':
        if (args['image_set'] is None) or (not os.path.exists(args['image_set'])):
            print("When using kaist, image set must be defined to a valid file: {}".format(args['image_set']))

        if args['show_dataset'] == True:
            show_dataset(args['dataset_root'], args['image_set'], args['image_fusion'])

        dataset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=SSDAugmentation(args['ssd_min_dim']), image_fusion=args['image_fusion'], target_transform=KAISTAnnotationTransform(output_format="SSD"))
    else:
        print("No dataset specified")
        sys.exit(-1)

    ssd_net = build_ssd('train', args['ssd_min_dim'], args['classes'], dataset=args['name'], cfg=args)
    net = ssd_net

    if args['cuda']:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args['resume']:
        print('Resuming training, loading {}...'.format(args['resume']))
        ssd_net.load_weights(args['resume'])
    else:
        vgg_weights = torch.load(args['ssd_initial_weights'])
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args['cuda']:
        net = net.cuda()

    if not args['resume']:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args['ssd_lr'], momentum=args['ssd_momentum'], weight_decay=args['ssd_weight_decay'])
    criterion = MultiBoxLoss(args['classes'], 0.5, 3, args['cuda'], args['ssd_variance'])

    net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args['ssd_batch_size']

    print('Dataset length: {}'.format(len(dataset)))
    print('Epoch size: {}'.format(epoch_size))

    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args['visdom']:
        vis_title = 'SSD.PyTorch on ' + dataset.name + '| lr: ' + repr(args['ssd_lr'])
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = viz.create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = viz.create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    if args['name'] == 'VOC':
        data_loader = data.DataLoader(dataset, args['ssd_batch_size'], num_workers=args['num_workers'], shuffle=True, collate_fn=detection_collate_VOC, pin_memory=True)
    elif args['name'] == 'KAIST':
        data_loader = data.DataLoader(dataset, args['ssd_batch_size'], num_workers=args['num_workers'], shuffle=True, collate_fn=detection_collate_KAIST_SSD, pin_memory=True)
    else:
        raise NotImplementedError

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args['start_iter'], args['ssd_iters']):
        if args['visdom'] and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            viz.update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, 'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

        if iteration in args['ssd_lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args['ssd_gamma'], step_index, args['ssd_lr'])

        # load train data
        t0 = time.time()
        try:
            # load train data
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        data_time = time.time() - t0

        if args['cuda']:
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
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " | " + args['name'] + ': iter ' + repr(iteration) + ' || lr: %g || Loss: %.4f (loc: %.4f, conf: %.4f) ||' %
                  (optimizer.param_groups[0]['lr'], loss.data.item(), loss_l.data.item(), loss_c.data.item()), end=' ')
            print('data: %.3fms, batch: %.3fs' % (data_time*1000, batch_time))

        if args['visdom']:
            viz.update_vis_plot(iteration, loss_l.data.item(), loss_c.data.item(), iter_plot, epoch_plot, 'append')

        # save model at a given frequency during training and at the end
        if (iteration != 0 and iteration % args['save_frequency'] == 0) or (iteration==args['ssd_iters']-1):
            print('Saving state, iter:', iteration)
            model_name = os.path.join(args['save_folder'], 'ssd300_' + args['name'] + '_' + repr(iteration) + '.pth')
            latest_name = ('ssd300_' + args['name'] + '_latest.pth')
            torch.save(ssd_net.state_dict(), model_name)
            # os.system('ln -sf {} {}'.format(model_name, latest_name))

    print("Finished. Model is saved at {}".format(latest_name))
    return


def adjust_learning_rate(optimizer, gamma, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    args = vars(arg_parser())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}
    del config

    try:
        if torch.cuda.is_available():
            if args['cuda']:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                print("WARNING: It looks like you have a CUDA device, but aren't using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if not os.path.exists(args['save_folder']):
            os.makedirs(args['save_folder'])

        if len(os.listdir(args['save_folder'])) != 0:
            print("Save directory is not empty! : {}".format(args['save_folder']))
            if "tmp" in args['save_folder']:
                print("but as the save folder contains 'tmp', we remove old data:")
                files = glob.glob(args['save_folder']+'/*')
                for f in files:
                    if f.endswith('.weights') or f.endswith('.pth'):
                        print("rm {}".format(f))
                        os.remove(f)
            else:
                print("not a tmp folder, you must fix it yourself!")
                sys.exit(-1)

        if args['visdom']:
            viz =  vis_plot()
        else:
            viz = None

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

    except KeyError as error:
        # Output expected KeyErrors.
        print("Parameter {} must be defined in config file".format(error))
        sys.exit(-1)

    train(args, viz)

import argparse
import os
import glob
import sys

from utils.str2bool import str2bool
from config.parse_config import *
from config.load_classes import load_classes
from models.vgg16_ssd import build_ssd
from models.yolo3 import *

import torch
import torch.backends.cudnn as cudnn

from data import BaseTransform


def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--image_set', default=None, help='Imageset')
    parser.add_argument('--trained_model', default=None, type=str, help='Trained state_dict file path to open')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: inverted LWIR] [...]')  # TODO VPY update when required
    parser.add_argument('--corrected_annotations', default=False, type=str2bool, help='[KAIST] do we use the corrected annotations ? (must ahve compatible imageset (VPY-test-strict-type-5)')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    parser.add_argument("--model", type=str, default=None, help="Model to train, either 'SSD' or 'YOLO'")

    args = vars(parser.parse_args())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}
    return args

def test_net(net, cuda, dataset, conf_thresh, nms_thres, classes, transform):
    num_images = len(dataset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))

        # --------------------------
        # get image and GT
        # --------------------------
        img = dataset.pull_image(i)
        img_gt = img.copy()
        img_det = img.copy()
        img_id, annotation, _ = dataset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()

        # --------------------------
        # forward pass
        # --------------------------
        y = net(x)
        detections = y.data


def main(args):
    model_name = args["model"]
    dataset_name = args["name"]
    imageset_name = args["image_set"].split("/")[-1].split(".")[0]
    cuda = torch.cuda.is_available() and args['cuda']

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    classes = load_classes(args['names'])
    num_classes = len(classes) + 1  # +1 for background

    # load net
    if model_name == "SSD":
        net = build_ssd(phase='test', size=300, num_classes=num_classes, dataset=dataset_name, cfg=args) # initialize SSD
        net.load_state_dict(torch.load(args['trained_model']))
    elif model_name == "YOLO":
        net = Darknet(args['yolo_model_config_path'], img_size=args['yolo_img_size'])
        net.load_weights(args['trained_model'])

    net.eval()

    print('Finished loading model!')

    # get augmentation function
    if model_name == "YOLO":
        from augmentations.YOLOaugmentations import YOLOaugmentation

    # load data
    if dataset_name == 'VOC':
        from data.voc0712 import VOCDetection
        from data.voc0712 import VOCAnnotationTransform
        dataset = VOCDetection(root=args['dataset_root'], image_sets=[('2007', 'test')], transform=None, target_transform=VOCAnnotationTransform())
    elif dataset_name == 'KAIST':
        from data.kaist import KAISTDetection, KAISTAnnotationTransform
        transform_fct = YOLOaugmentation(args['yolo_img_size']) if model_name == "YOLO" else None
        dataset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=transform_fct,
                                 image_fusion=args['image_fusion'], corrected_annotations=args['corrected_annotations'],
                                 output_format=model_name, target_transform=KAISTAnnotationTransform(output_format=model_name))
        # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args['num_workers'],
        #                         collate_fn=detection_collate_KAIST_YOLO)
    elif dataset_name == 'COCO':
        from data.coco_list import ListDataset
        dataset = ListDataset(list_path=args['validation_set'], img_size=args['yolo_img_size'])
    else:
        raise NotImplementedError

    if args['cuda']:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    if model_name == "YOLO":
        conf_thresh = args['yolo_conf_thres']
        nms_thres = args['yolo_nms_thres']
        transform_fct = BaseTransform(args["ssd_min_dim"], (104, 117, 123))#None

    elif model_name == "SSD":
        conf_thresh = args['ssd_visual_threshold']
        nms_thres = None
        transform_fct = BaseTransform(net.size, (104, 117, 123))

    test_net(net=net, cuda=cuda, dataset=dataset, conf_thresh=conf_thresh, nms_thres=nms_thres, classes=classes, transform=transform_fct)
    # test_net(net=net, cuda=args['cuda'], testset=dataset, transform=BaseTransform(net.size, (104, 117, 123)), thresh=args['ssd_visual_threshold'], labelmap=classes) #TODO VPY: MEAN ?!
    # test_net(net=net, dataloader=dataloader, img_size=args['yolo_img_size'], classes=classes, conf_thres=args['yolo_conf_thres'], nms_thres=args['yolo_nms_thres'])#, testset=testset):#, transform=BaseTransform(net.size, (104, 117, 123)), thresh=args['ssd_visual_threshold'], labelmap=labelmap)  # TODO VPY: MEAN ?!




def check_args(args):
    if args['name'] == "KAIST":
        if args['image_fusion'] == -1:
            print("image fusion must be specified")
            sys.exit(-1)
        print("Image fusion value: {}".format(args['image_fusion']))

    if not os.path.exists(args['dataset_root']):
        print('Must specify *existing* dataset_root')
        sys.exit(-1)

    if args["model"].upper() == "SSD":
        if args["name"] not in {"VOC", "KAIST"}:
            print("Dataset {} not supported with model {}".format(args["name"], args["model"]))
            sys.exit(-1)

    elif args["model"].upper() == "YOLO":
        if args["name"] not in {"COCO", "KAIST"}:
            print("Dataset {} not supported with model {}".format(args["name"], args["model"]))
            sys.exit(-1)
    else:
        print("Model {} is not supported".format(args["model"]))
        sys.exit(-1)

if __name__ == '__main__':
    args = arg_parser()
    check_args(args)
    main(args)

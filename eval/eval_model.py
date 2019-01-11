"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
import sys
import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from utils.str2bool import str2bool
from config.parse_config import *
from config.load_classes import load_classes
from eval.get_GT import get_GT
from data import BaseTransform
from models.vgg16_ssd import build_vgg_ssd
from models.mobilenet2_ssd import build_mobilenet_ssd

from models.yolo3 import *
from eval.forward_pass import *
from eval.eval_tools import eval_results_voc



def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--image_set', default=None, help='Imageset')
    parser.add_argument('--trained_model', default=None, type=str, help='Trained state_dict file path to open')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: inverted LWIR] [...]')  # TODO VPY update when required
    parser.add_argument('--corrected_annotations', default=False, type=str2bool, help='[KAIST] do we use the corrected annotations ? (must ahve compatible imageset (VPY-test-strict-type-5)')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    parser.add_argument("--model", type=str, default=None, help="Model to test, either 'VGG_SSD','MOBILENET2_SSD' or 'YOLO'")

    args_cmd = vars(parser.parse_args())
    args_cmd = {key: value for key, value in args_cmd.items() if value is not None} # remove non attributed values from parser

    args_file = parse_data_config(args_cmd['data_config_path'])
    if args_cmd['model'] == "VGG_SSD":
        args_file = {key.replace("vgg_", ""): value for key, value in args_file.items() if (not key.startswith("mobilenet")) and (not key.startswith("yolo"))}
    elif args_cmd['model'] == "MOBILENET2_SSD":
        args_file = {key.replace("mobilenet_", ""): value for key, value in args_file.items() if (not key.startswith("vgg")) and (not key.startswith("yolo"))}
    elif args_cmd["model"] == "YOLO":
        args_file = {key: value for key, value in args_file.items() if (not key.startswith("vgg")) and (not key.startswith("mobilenet"))}

    # remove duplicates entries, command line has priority on config file
    duplicates = (args_file.keys() & args_cmd.keys())
    for key in duplicates:
        del args_file[key]

    args = {**args_cmd, **args_file}

    return args

def main(args):
    model_name = args["model"]
    classes = load_classes(args['names'])
    num_classes = len(classes) + 1  # +1 for background

    cuda = torch.cuda.is_available() and args['cuda']

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # load data
    dataset_mean = (104, 117, 123, 104)  # TODO VPY and for kaist ?

    # get augmentation function
    if model_name == "YOLO":
        from augmentations.YOLOaugmentations import YOLOaugmentation

    if args['name'] == "KAIST":
        from data.kaist import KAISTDetection, KAISTAnnotationTransform
        transform_fct = BaseTransform(args["ssd_min_dim"], dataset_mean) if model_name in {"VGG_SSD", "MOBILENET2_SSD"} else  YOLOaugmentation(args['yolo_img_size'])

        dataset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=transform_fct,
                                    target_transform=KAISTAnnotationTransform(output_format="VOC_EVAL"), dataset_name="KAIST", image_fusion=args['image_fusion'], corrected_annotations=args['corrected_annotations'])

    elif args['name'] == "VOC":
        from data.voc0712 import VOCDetection
        from data.voc0712 import VOCAnnotationTransform
        dataset = VOCDetection(root=args['dataset_root'], image_sets=[('2007', 'test')], transform=BaseTransform(args["ssd_min_dim"], dataset_mean), target_transform=VOCAnnotationTransform(), dataset_name="VOC")

    if args['name'] == 'COCO':
        from data.coco_list import ListDataset
        dataset = ListDataset(list_path=args['validation_set'], img_size=args['yolo_img_size'])


    print("Loading weights from file: {}".format(args["trained_model"]))
    # set up model
    if model_name == "VGG_SSD":
        net = build_vgg_ssd(phase='test', size=300, num_classes=num_classes,cfg=args) # initialize SSD
        try:
            net.load_state_dict(torch.load(args['trained_model']))
        except:
            # If we try to load a model with "vgg" conv instead of "basenet" conv, as the name changed to be generic and we don't want to train again,
            # we just have to change names in the OrderDict loaded form file
            old_model = torch.load(args['trained_model'])
            new_model = OrderedDict()
            for key, value in old_model.items():
                new_model[key.replace("vgg", "basenet")] = value
            new_model._metadata = OrderedDict()
            for key, value in old_model._metadata.items():
                new_model._metadata[key.replace("vgg", "basenet")] = value
            net.load_state_dict(new_model)

    elif model_name == "MOBILENET2_SSD":
        net = build_mobilenet_ssd(phase='test', size=320, num_classes=num_classes, cfg=args)  # initialize SSD
        net.load_state_dict(torch.load(args['trained_model']))
    elif model_name == "YOLO":
        net = Darknet(args['yolo_model_config_path'], img_size=args['yolo_img_size'])
        net.load_weights(args['trained_model'])
        if cuda:
            net.cuda()

    print('Finished loading model!')

    #model in eval mode
    net.eval()  # Set in evaluation mode

    if args['cuda']:
        net = net.cuda()
        cudnn.benchmark = True

    #get gt
    print('Read GT')
    ground_truth = get_GT(dataset, classes)

    #get detections
    print("Forward pass")
    if model_name in {"VGG_SSD", "MOBILENET2_SSD"}:
        det_image_ids, det_BB, det_confidence = forward_pass_ssd(net=net, cuda=args['cuda'], dataset=dataset, labelmap=classes)
    elif model_name == "YOLO":
        det_image_ids, det_BB, det_confidence = forward_pass_yolo(net=net, cuda=args['cuda'], dataloader = None, img_size=args['yolo_img_size'], classes=classes, conf_thres=args['yolo_conf_thres'], nms_thres=args['yolo_nms_thres'], dataset=dataset, labelmap=classes)

    #evaluate
    print('Evaluating detections')
    mAP, aps_dict = eval_results_voc(ground_truth, det_BB, det_image_ids, det_confidence, labelmap=classes, use_voc07_metric=True)

    print("Finished")


def check_args(args):

    if args['name'] == "KAIST":
        if args['image_fusion'] == -1:
            print("image fusion must be specified")
            sys.exit(-1)
        print("Image fusion value: {}".format(args['image_fusion']))

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

if __name__ == '__main__':
    args = arg_parser()
    check_args(args)
    main(args)

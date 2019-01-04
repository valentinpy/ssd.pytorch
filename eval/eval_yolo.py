"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data import BaseTransform
from eval.get_GT import get_GT
from eval.eval_tools import eval
from eval.forward_pass import *
from config.parse_config import *
from models.vgg16_ssd import build_ssd

from utils.str2bool import str2bool
from models.yolo3 import *
from models.yolo3_utils import *
from data.coco_list import *
from data.kaist import KAISTDetection, KAISTAnnotationTransform, detection_collate_KAIST_YOLO
from data.kaist import KAIST_CLASSES as KAISTlabelmap

from augmentations.YOLOaugmentations import YOLOaugmentation
from config.load_classes import load_classes


def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--image_set', default=None, help='Imageset')
    parser.add_argument('--trained_model', default=None, type=str, help='Trained state_dict file path to open')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: inverted LWIR] [...]')  # TODO VPY update when required
    parser.add_argument('--corrected_annotations', default=False, type=str2bool, help='[KAIST] do we use the corrected annotations ? (must ahve compatible imageset (VPY-test-strict-type-5)')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = vars(arg_parser())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}
    del config

    # load data
    if args['name'] == 'COCO':
        dataset = ListDataset(list_path=args['validation_set'], img_size=args['yolo_img_size'])
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args["num_workers"], collate_fn=detection_collate_COCO_YOLO)
    elif args['name'] == 'KAIST':
        kaist_root = args["dataset_root"]
        image_set = args["image_set"]
        image_fusion = args["image_fusion"]

        dataset = KAISTDetection(root=kaist_root, image_set=image_set, transform=YOLOaugmentation(args['yolo_img_size']), image_fusion=image_fusion,
                                 output_format="YOLO", target_transform=KAISTAnnotationTransform(output_format="YOLO"))

        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args['num_workers'], collate_fn=detection_collate_KAIST_YOLO)

    else:
        raise NotImplementedError

    cuda = torch.cuda.is_available() and args['cuda']

    # Set up model
    net = Darknet(args['yolo_model_config_path'], img_size=args['yolo_img_size'])
    net.load_weights(args['trained_model'])

    if cuda:
        net.cuda()

    net.eval()  # Set in evaluation mode
    classes = load_classes(args['names'])  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # load net
    num_classes = len(classes) + 1 # +1 for background
    print('Finished loading model!')

    # load data
    if args['name'] == "KAIST":
        dataset = KAISTDetection(root=kaist_root, image_set=image_set, transform=YOLOaugmentation(args['yolo_img_size']), image_fusion=image_fusion,
                                 output_format="YOLO", target_transform=KAISTAnnotationTransform(output_format="VOC_EVAL"))

        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args['num_workers'],
                                collate_fn=detection_collate_KAIST_YOLO)

        labelmap = KAISTlabelmap
    else:
        print("Dataset not implemented")
        raise NotImplementedError

    if args['cuda']:
        net = net.cuda()
        cudnn.benchmark = True

    print('Read GT')
    ground_truth = get_GT(dataset, labelmap)

    print("Forward pass")
    det_image_ids, det_BB, det_confidence = forward_pass_yolo(net=net, cuda=args['cuda'], dataloader = None, img_size=args['yolo_img_size'], classes=classes, conf_thres=args['yolo_conf_thres'], nms_thres=args['yolo_nms_thres'], dataset=dataset, labelmap=labelmap)

    # evaluation
    print('Evaluating detections')
    # for i in
    mAP, aps_dict = eval(ground_truth, det_BB, det_image_ids, det_confidence, labelmap=labelmap, use_voc07_metric=True)
    # print("mAP: {}".format(mAP))
    # print("AP: {}".format(ap_dict))
    # print("tpfp_dict: {}".format(tpfp_dict))

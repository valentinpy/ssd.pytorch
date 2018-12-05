"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import torch
import torch.backends.cudnn as cudnn

import argparse

from models.ssd import build_ssd

from data import BaseTransform
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection
from data import VOC_CLASSES as VOClabelmap
from data import KAISTAnnotationTransform, KAISTDetection
from data import KAIST_CLASSES as KAISTlabelmap
from eval.get_GT import get_GT
from eval.eval_tools import eval
from eval.forward_pass import forward_pass


from utils.misc import str2bool
from utils.misc import frange

def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--dataset_type', default='VOC', choices=['VOC', 'COCO', "KAIST"], type=str, help='VOC, COCO, KAIST (requires image_set)')
    parser.add_argument('--image_set', default=None, help='Imageset')
    parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth', type=str, help='Trained state_dict file path to open')
    parser.add_argument('--confidence_threshold', default=0.01, type=float, help='Detection confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--dataset_root', default=None, help='Location of dataset root directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # parse arguments
    args = arg_parser()

    # prepare environnement
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't using CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # configure according to dataset used
    if args.dataset_type == "VOC":
        labelmap = VOClabelmap
    elif args.dataset_type == "KAIST":
        labelmap = KAISTlabelmap
    else:
        print("Dataset not implemented")
        raise NotImplementedError

    dataset_mean = (104, 117, 123)  # TODO VPY and for kaist ?
    set_type = 'test'

    # load net
    num_classes = len(labelmap) + 1 # +1 for background
    net = build_ssd('test', 300, num_classes, args.dataset_type) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')

    # load data
    if args.dataset_type == "VOC":
        dataset = VOCDetection(root=args.dataset_root, image_sets=[('2007', set_type)], transform=BaseTransform(300, dataset_mean), target_transform=VOCAnnotationTransform(), dataset_name="VOC")
    elif args.dataset_type == "KAIST":
        #dataset_mean = tuple(compute_KAIST_dataset_mean(args.dataset_root, args.image_set))
        dataset = KAISTDetection(root=args.dataset_root,image_set=args.image_set, transform=BaseTransform(300, dataset_mean), target_transform=KAISTAnnotationTransform(), dataset_name="KAIST")
    else:
        print("Dataset not implemented")
        raise NotImplementedError

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    print('Read GT')
    ground_truth = get_GT(dataset, labelmap)

    print("Forward pass")
    det_image_ids, det_BB, det_confidence = forward_pass(net=net, cuda=args.cuda, dataset=dataset, labelmap=labelmap)

    # evaluation
    print('Evaluating detections')
    # for i in
    mAP, aps_dict = eval(ground_truth, det_BB, det_image_ids, det_confidence, labelmap=labelmap, use_voc07_metric=True)
    # print("mAP: {}".format(mAP))
    # print("AP: {}".format(ap_dict))
    # print("tpfp_dict: {}".format(tpfp_dict))
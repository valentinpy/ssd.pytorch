"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import argparse
import torch
import torch.backends.cudnn as cudnn
from data import BaseTransform
from data.voc0712 import VOCAnnotationTransform, VOCDetection
# from data.voc0712 import VOC_CLASSES as VOClabelmap
from data.kaist import KAISTAnnotationTransform, KAISTDetection
# from data.kaist import KAIST_CLASSES as KAISTlabelmap
from eval.get_GT import get_GT
from eval.eval_tools import eval
from eval.forward_pass import forward_pass_ssd
from config.parse_config import *
from utils.str2bool import str2bool
from models.vgg16_ssd import build_ssd
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

    # parse arguments
    args = vars(arg_parser())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}
    del config

    # prepare environnement
    if torch.cuda.is_available():
        if args['cuda']:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("WARNING: It looks like you have a CUDA device, but aren't using CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # configure according to dataset used
    labelmap = load_classes(args['names'])

    dataset_mean = (104, 117, 123)  # TODO VPY and for kaist ?
    set_type = 'test'

    # load net
    num_classes = len(labelmap) + 1 # +1 for background
    net = build_ssd(phase='test', size=300, num_classes=num_classes, dataset=args['name'], cfg=args) # initialize SSD
    net.load_state_dict(torch.load(args['trained_model']))
    net.eval()
    print('Finished loading model!')

    # load data
    if args['name'] == "VOC":
        dataset = VOCDetection(root=args['dataset_root'], image_sets=[('2007', set_type)], transform=BaseTransform(300, dataset_mean), target_transform=VOCAnnotationTransform(), dataset_name="VOC")
    elif args['name'] == "KAIST":
        #dataset_mean = tuple(compute_KAIST_dataset_mean(args.dataset_root, args.image_set))
        dataset = KAISTDetection(root=args['dataset_root'],image_set=args['image_set'], transform=BaseTransform(300, dataset_mean), target_transform=KAISTAnnotationTransform(output_format="VOC_EVAL"), dataset_name="KAIST", image_fusion=args['image_fusion'], corrected_annotations=args['corrected_annotations'])
    else:
        print("Dataset not implemented")
        raise NotImplementedError

    if args['cuda']:
        net = net.cuda()
        cudnn.benchmark = True

    print('Read GT')
    ground_truth = get_GT(dataset, labelmap)

    print("Forward pass")
    det_image_ids, det_BB, det_confidence = forward_pass_ssd(net=net, cuda=args['cuda'], dataset=dataset, labelmap=labelmap)

    # evaluation
    print('Evaluating detections')
    # for i in
    mAP, aps_dict = eval(ground_truth, det_BB, det_image_ids, det_confidence, labelmap=labelmap, use_voc07_metric=True)
    # print("mAP: {}".format(mAP))
    # print("AP: {}".format(ap_dict))
    # print("tpfp_dict: {}".format(tpfp_dict))

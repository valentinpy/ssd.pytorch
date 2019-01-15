"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from config.load_classes import load_classes
from eval.get_GT import get_GT
from data import BaseTransform
from models.vgg16_ssd import build_vgg_ssd
from models.mobilenet2_ssd import build_mobilenet_ssd

from models.yolo3 import *
from eval.forward_pass import *
from eval.eval_tools import eval_results_voc

from augmentations.YOLOaugmentations import YOLOaugmentation
from data.kaist import KAISTDetection, KAISTAnnotationTransform
from data.voc0712 import VOCDetection
from data.voc0712 import VOCAnnotationTransform
from data.coco_list import ListDataset



def main(args):
    model_name = args["model"]
    classes = load_classes(args['names'])

    cuda = torch.cuda.is_available() and args['cuda']
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # load data
    dataset_mean = (104, 117, 123, 104)  # TODO VPY and for kaist ?

    if args['name'] == "KAIST":
        transform_fct = BaseTransform(args["ssd_min_dim"], dataset_mean) if model_name in {"VGG_SSD", "MOBILENET2_SSD"} else  YOLOaugmentation(args['yolo_img_size'])
        target_transform_fct = KAISTAnnotationTransform(output_format="VOC_EVAL")
        dataset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=transform_fct,
                                 target_transform=target_transform_fct, dataset_name="KAIST",
                                 image_fusion=args['image_fusion'], corrected_annotations=args['corrected_annotations'])

    elif args['name'] == "VOC":
        dataset = VOCDetection(root=args['dataset_root'], image_sets=[('2007', 'test')], transform=BaseTransform(args["ssd_min_dim"], dataset_mean),
                               target_transform=VOCAnnotationTransform(), dataset_name="VOC")

    if args['name'] == 'COCO':
        dataset = ListDataset(list_path=args['validation_set'], img_size=args['yolo_img_size'])

    # set up model
    print("Creating net and loading weights from file: {}".format(args["trained_model"]))
    net = build_eval_net(args, classes)


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
    else:
        raise NotImplementedError

    #evaluate results
    print('Evaluating detections')
    mAP, aps_dict = eval_results_voc(ground_truth, det_BB, det_image_ids, det_confidence, labelmap=classes, use_voc07_metric=True)

    print("Finished")


def build_eval_net(args, classes):
    model_name = args["model"]
    cuda = torch.cuda.is_available() and args['cuda']
    num_classes = len(classes) + 1  # +1 for background

    if model_name == "VGG_SSD":
        net = build_vgg_ssd(phase='test', size=300, num_classes=num_classes, cfg=args)  # initialize SSD
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
    else:
        raise NotImplementedError

    return net


if __name__ == '__main__':
    args = arg_parser(role="eval")
    check_args(args, role="eval")
    main(args)

import argparse
import os
import glob
import sys
import cv2
import signal
from collections import OrderedDict

from utils.timer import Timer
from utils.str2bool import str2bool
from config.parse_config import *
from config.load_classes import load_classes
from models.yolo3 import *
from models.yolo3_utils import *
from models.vgg16_ssd import build_vgg_ssd
from models.mobilenet2_ssd import build_mobilenet_ssd


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
    parser.add_argument("--model", type=str, default=None, help="Model to demo, either 'VGG_SSD','MOBILENET2_SSD' or 'YOLO'")

    args_cmd = vars(parser.parse_args())
    args_cmd = {key: value for key, value in args_cmd.items() if value is not None}  # remove non attributed values from parser

    args_file = parse_data_config(args_cmd['data_config_path'])
    if args_cmd['model'] == "VGG_SSD":
        args_file = {key.replace("vgg_", ""): value for key, value in args_file.items() if
                     (not key.startswith("mobilenet")) and (not key.startswith("yolo"))}
    elif args_cmd['model'] == "MOBILENET2_SSD":
        args_file = {key.replace("mobilenet_", ""): value for key, value in args_file.items() if
                     (not key.startswith("vgg")) and (not key.startswith("yolo"))}
    elif args_cmd["model"] == "YOLO":
        args_file = {key: value for key, value in args_file.items() if (not key.startswith("vgg")) and (not key.startswith("mobilenet"))}

    # remove duplicates entries, command line has priority on config file
    duplicates = (args_file.keys() & args_cmd.keys())
    for key in duplicates:
        del args_file[key]

    args = {**args_cmd, **args_file}

    return args

def test_net(model_name, net, cuda, dataset, conf_thres, nms_thres, classes, transform):
    print('\nPerforming object detection:')

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    _t = {'im_detect': Timer(), 'misc': Timer()}

    num_images = len(dataset)
    num_classes = len(classes)

    for i in range(num_images):
        _t['im_detect'].tic()

        # --------------------------
        # get image and GT
        # --------------------------
        img_gt= dataset.pull_image(i)
        img_det = img_gt.copy()
        img_id, annotation, _ = dataset.pull_anno(i)

        _, img,_,_,_ = dataset[i]
        input_imgs = Variable(img.type(Tensor).unsqueeze(0))

        if cuda:
            input_imgs = input_imgs.cuda()

        if model_name in {"VGG_SSD", "MOBILENET2_SSD"}:
            y = net(input_imgs)
            detections = y.data

        elif model_name == "YOLO":
            with torch.no_grad():
                detections = net(input_imgs)
                detections = non_max_suppression(detections, num_classes, conf_thres, nms_thres)  # (x1, y1, x2, y2, object_conf, class_score, class_pred)

        # Log progress
        detect_time = _t['im_detect'].toc(average=True)
        print('Image: {}, Image inference Time: {}ms'.format(i, int(detect_time*1000)))


        # --------------------------
        # Ground truth
        # --------------------------
        gt_id_cnt = 0
        for box in annotation:
            gt_id_cnt += 1
            label = [key for (key, value) in dataset.target_transform.class_to_ind.items() if value == box[4]][0]
            cv2.rectangle(img_gt,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 255, 0),
                          1)
            cv2.putText(img_gt, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # --------------------------
        # Detections
        # --------------------------
        if model_name in {"VGG_SSD", "MOBILENET2_SSD"}:
            scale = torch.Tensor([img_gt.shape[1], img_gt.shape[0], img_gt.shape[1], img_gt.shape[0]])
            pred_num = 0
            for i in range(detections.size(1)):  # loop for all classes
                j = 0
                while detections[0, i, j, 0] >= conf_thres:  # loop for all detection for the corresponding class
                    score = detections[0, i, j, 0]
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    pred_num += 1
                    j += 1
                    cv2.rectangle(img_det,
                                  (int(pt[0]), int(pt[1])),
                                  (int(pt[2]), int(pt[3])),
                                  (255, 0, 0),
                                  1)
                    cv2.putText(img_det, np.array2string((score.data * 100).cpu().numpy().astype(int)),
                                (int(pt[2]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        elif model_name == "YOLO":
            scale = [img_gt.shape[1], img_gt.shape[0], img_gt.shape[1], img_gt.shape[0]]
            scale = [elem / 416 for elem in scale]
            if (detections is not None) and (detections[0] is not None):
                detections = detections[0].cpu().numpy()
                pred_num = 0
                for i in range(detections.shape[0]):
                    j = 0
                    if detections[i, 4] >= conf_thres:
                        score = detections[i, 4]
                        xmin = int(detections[i, 0] * scale[0])
                        xmax = int(detections[i, 2] * scale[2])
                        ymin = int(detections[i, 1] * scale[1])
                        ymax = int(detections[i, 3] * scale[3])
                        label_name = repr(score)
                        pred_num += 1
                        j += 1
                        cv2.rectangle(img_det,
                                      (xmin, ymin),
                                      (xmax, ymax),
                                      (255, 0, 0),
                                      1)
                        cv2.putText(img_det, label_name, (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # --------------------------
        # Plot image, GT and dets
        # --------------------------
        cv2.imshow("GT || DET", np.hstack((img_gt, img_det))[:, :, (2, 1, 0)])
        cv2.waitKey(0)


def main(args):
    model_name = args["model"]
    dataset_name = args["name"]
    cuda = torch.cuda.is_available() and args['cuda']

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    classes = load_classes(args['names'])
    num_classes = len(classes) + 1  # +1 for background

    # load net
    print("Loading weights from file: {}".format(args["trained_model"]))
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

    print('Finished loading model!')

    net.eval()

    # get augmentation function
    if model_name == "YOLO":
        from augmentations.YOLOaugmentations import YOLOaugmentation
    elif model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
        from augmentations.SSDaugmentations import SSDAugmentationDemo

    # load data
    if dataset_name == 'VOC':
        from data.voc0712 import VOCDetection
        from data.voc0712 import VOCAnnotationTransform
        dataset = VOCDetection(root=args['dataset_root'], image_sets=[('2007', 'test')], transform=None, target_transform=VOCAnnotationTransform())
    elif dataset_name == 'KAIST':
        from data.kaist import KAISTDetection, KAISTAnnotationTransform
        transform_fct = YOLOaugmentation(args['yolo_img_size']) if model_name == "YOLO" else SSDAugmentationDemo(args["ssd_min_dim"])
        dataset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=transform_fct,
                                 image_fusion=args['image_fusion'], corrected_annotations=args['corrected_annotations'],
                                 target_transform=KAISTAnnotationTransform(output_format="VOC_EVAL"))
    elif dataset_name == 'COCO':
        from data.coco_list import ListDataset
        dataset = ListDataset(list_path=args['validation_set'], img_size=args['yolo_img_size'])
    else:
        raise NotImplementedError

    if cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    if model_name == "YOLO":
        conf_thresh = args['yolo_conf_thres']
        nms_thres = args['yolo_nms_thres']
        transform_fct = BaseTransform(args["yolo_img_size"], (104, 117, 123))#None

    elif model_name in {"VGG_SSD", "MOBILENET2_SSD"}:
        conf_thresh = args['ssd_visual_threshold']
        nms_thres = None
        transform_fct = BaseTransform(args["ssd_min_dim"], (104, 117, 123))

    test_net(model_name=model_name, net=net, cuda=cuda, dataset=dataset, conf_thres=conf_thresh, nms_thres=nms_thres, classes=classes, transform=transform_fct)


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

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    args = arg_parser()
    check_args(args)
    main(args)

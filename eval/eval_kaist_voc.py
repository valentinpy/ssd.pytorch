"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import argparse
import numpy as np

from utils.timer import Timer
from models.ssd import build_ssd

from data import BaseTransform
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection
from data import VOC_CLASSES as VOClabelmap
from data import KAISTAnnotationTransform, KAISTDetection
from data import KAIST_CLASSES as KAISTlabelmap


from eval.voc_ap import voc_ap
from utils.misc import str2bool

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

def eval(gt_class_recs, det_BB, det_image_ids, det_confidence, labelmap, use_voc07_metric=True):
    print("\n----------------------------------------------------------------")
    print("Eval")
    print("----------------------------------------------------------------")
    print('VOC07 metric? ' + ('Yes' if use_voc07_metric else 'No'))
    aps = []
    aps_dict = {}

    for i, cls in enumerate(labelmap):
        rec, prec, ap = voc_eval(gt_class_recs[i+1], det_BB[i+1], det_image_ids[i+1], det_confidence[i+1], ovthresh=0.5, use_07_metric=use_voc07_metric)
        aps += [ap]
        aps_dict[cls] = ap
        print('AP for {} = {:.4f}'.format(cls, ap))

    mAP = np.mean(aps)
    print('Mean AP = {:.4f}'.format(mAP))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')

    # details = None #TODO VPY
    return mAP, aps_dict

def voc_eval(gt_class_recs, det_BB, det_image_ids, det_confidence, ovthresh=0.5, use_07_metric=True):
    # TODO VPY document!

    npos = len([x for _, value in gt_class_recs.items() for x in value['difficult'] if x == False])

    # sort by confidence
    det_sorted_ind = np.argsort(-det_confidence)
    det_sorted_scores = np.sort(-det_confidence)
    det_BB = det_BB[det_sorted_ind, :]
    det_image_ids = [det_image_ids[x] for x in det_sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(det_image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        gt_R = gt_class_recs[det_image_ids[d]]
        det_bb = det_BB[d, :].astype(float)
        ovmax = -np.inf
        gt_BBGT = gt_R['bbox'].astype(float)
        if gt_BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_BBGT[:, 0], det_bb[0])
            iymin = np.maximum(gt_BBGT[:, 1], det_bb[1])
            ixmax = np.minimum(gt_BBGT[:, 2], det_bb[2])
            iymax = np.minimum(gt_BBGT[:, 3], det_bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((det_bb[2] - det_bb[0]) * (det_bb[3] - det_bb[1]) +
                   (gt_BBGT[:, 2] - gt_BBGT[:, 0]) *
                   (gt_BBGT[:, 3] - gt_BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not gt_R['difficult'][jmax]:
                if not gt_R['det'][jmax]:
                    tp[d] = 1.
                    gt_R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def forward_pass(net, cuda, dataset):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(len(labelmap)+1)]
    det_image_ids = [[] for _ in range(len(labelmap)+1)]
    det_BB = [np.empty((0,4)) for _ in range(len(labelmap)+1)]
    det_confidence = [np.empty((0)) for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    # loop for all test images
    for i in range(num_images):

        # get image + annotations + dimensions
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if cuda:
            x = x.cuda()

        # forward pass
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=True)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

            img_id = dataset.pull_img_id(i)
            det_image_ids[j] += [img_id for _ in range(dets.size(0))]

            for l in range(dets.size(0)):
                det_BB[j] = np.vstack((det_BB[j], boxes[l].cpu().numpy()))
                det_confidence[j] = np.append(det_confidence[j], scores[l])

        if (i % 100 == 0):
            print('im_detect: {:d}/{:d}. Detection time per image: {:.3f}s'.format(i, num_images, detect_time))

    return all_boxes, det_BB, det_image_ids, det_confidence

def get_GT(dataset, labelmap):
    num_images = len(dataset)

    gt_all_classes_recs = [{} for _ in range(len(labelmap)+1)]

    #loop for all classes
    # skip j = 0, because it's the background class
    for j in range(1, len(labelmap)+1):

        # extract gt objects for this class
        gt_class_recs = {}

        #read all images
        for i in range(num_images):
            img_id, gt, detailed_gt = dataset.pull_anno(i)

            bbox = np.empty((0,4))
            gt_difficult = []

            if dataset.name == "VOC":
                for g in detailed_gt:
                    if (g['name'] == labelmap[j - 1]):
                        bbox = np.append(bbox, np.array([g['bbox']]), axis=0)
                        gt_difficult.append(g['difficult'])
                gt_difficult = np.array(gt_difficult).astype(np.bool)
                det = [False] * len(gt_difficult)

            elif dataset.name == "KAIST":
                for g in gt:
                    if (g[4]==j - 1):
                        bbox = np.append(bbox, np.array([g[0:4]]), axis=0)
                        gt_difficult.append(False)
                gt_difficult = np.array(gt_difficult).astype(np.bool)
                det = [False] * len(gt_difficult)

            else:
                print("Dataset not implemented")
                raise NotImplementedError

            gt_class_recs[img_id] = {'bbox': bbox, 'difficult': gt_difficult, 'det': det}
        gt_all_classes_recs[j] = gt_class_recs

    return gt_all_classes_recs


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
    detections, det_BB, det_image_ids, det_confidence = forward_pass(net=net, cuda=args.cuda, dataset=dataset)

    # evaluation
    print('Evaluating detections')
    mAP, ap_dict = eval(ground_truth, det_BB, det_image_ids, det_confidence, labelmap=labelmap, use_voc07_metric=True)
    print("mAP: {}".format(mAP))
    print("AP: {}".format(ap_dict))


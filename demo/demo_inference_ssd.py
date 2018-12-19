from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from data import BaseTransform

from data.voc0712 import VOCAnnotationTransform, VOCDetection
from data.voc0712 import VOC_CLASSES as VOClabelmap
from data.kaist import KAISTAnnotationTransform, KAISTDetection
from data.kaist import KAIST_CLASSES as KAISTlabelmap

from utils.str2bool import str2bool
from models.vgg16_ssd import build_ssd
import cv2
import numpy as np
from config.parse_config import *


def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--image_set', default=None, help='Imageset')
    parser.add_argument('--trained_model', default=None, type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='demo/', type=str, help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.6, type=float, help='Final confidence threshold')
    # parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--dataset_root', default=None, help='Location of dataset root directory')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: lwir inverted] [...]')  # TODO VPY update when required
    parser.add_argument('--corrected_annotations', default=False, type=str2bool, help='[KAIST] do we use the corrected annotations ? (must ahve compatible imageset (VPY-test-strict-type-5)')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")

    args = parser.parse_args()
    return args


def test_net(save_folder, net, cuda, testset, transform, thresh, labelmap):
    # dump predictions and assoc. ground truth to text file for now
    filename = os.path.join(save_folder,'gt_and_predictions.txt')

    #empty the file
    with open(filename, "w"):
        pass

    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))

        # --------------------------
        # get image and GT
        # --------------------------
        img = testset.pull_image(i)
        img_gt = img.copy()
        img_det = img.copy()
        img_id, annotation, _ = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()

        # --------------------------
        # forward pass
        # --------------------------
        y = net(x)
        detections = y.data

        #--------------------------
        # Ground truth
        # --------------------------
        # with open(filename, mode='a') as f:
        #     f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
        gt_id_cnt=0
        for box in annotation:
            gt_id_cnt+=1
            label = [key for (key, value) in testset.target_transform.class_to_ind.items() if value == box[4]][0]
            # f.write(repr(gt_id_cnt) + ' label: '+ label + ' || ' + ' || '.join(str(b) for b in box[0:4])+'\n')

            cv2.rectangle(img_gt,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 255, 0),
                          1
                          )
            cv2.putText(img_gt, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # --------------------------
        # Detections
        # --------------------------
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)): # loop for all classes
            j = 0
            while detections[0, i, j, 0] >=  thresh: #loop for all detection for the corresponding class (?)
                # if pred_num == 0:
                    # with open(filename, mode='a') as f:
                    #     f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                # with open(filename, mode='a') as f:
                #     f.write(str(pred_num)+' label: '+label_name+' || score: ' + str(repr(score).split('(')[1].split(')')[0]) + ' || '+' || '.join(str(c) for c in coords) + '\n')
                j += 1

                cv2.rectangle(img_det,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              (255, 0, 0),
                              1
                              )
                cv2.putText(img_det, np.array2string((score.data*100).cpu().numpy().astype(int)), (int(pt[2]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # --------------------------
        # Plot image, GT and dets
        # --------------------------
        cv2.imshow("GT || DET", np.hstack((img_gt, img_det))[:,:,(2,1,0)])
        cv2.waitKey(0)



if __name__ == '__main__':
    args = vars(arg_parser())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}
    del config

    if args['cuda'] and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args['save_folder']):
        os.mkdir(args['save_folder'])

    # load net
    if args['name'] == 'VOC':
        num_classes = len(VOClabelmap) + 1 # +1 background
        labelmap = VOClabelmap
        net = build_ssd(phase='test', size=300, num_classes=num_classes, dataset="VOC", cfg=args) # initialize SSD
    elif args['name'] == 'KAIST':
        num_classes = len(KAISTlabelmap) + 1  # +1 background
        labelmap = KAISTlabelmap
        net = build_ssd(phase='test', size=300, num_classes=num_classes, dataset="KAIST", cfg=args)  # initialize SSD
    else:
        raise NotImplementedError

    net.load_state_dict(torch.load(args['trained_model']))
    net.eval()

    print('Finished loading model!')

    # load data
    if args['name'] == 'VOC':
        testset = VOCDetection(root=args['dataset_root'], image_sets=[('2007', 'test')], transform=None, target_transform=VOCAnnotationTransform())
    elif args['name'] == 'KAIST':
        testset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=None, target_transform=KAISTAnnotationTransform(output_format="SSD"), dataset_name='KAIST', image_fusion=args['image_fusion'], corrected_annotations=args['corrected_annotations'])
    else:
        raise NotImplementedError

    if args['cuda']:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(save_folder=args['save_folder'], net=net, cuda=args['cuda'], testset=testset, transform=BaseTransform(net.size, (104, 117, 123)), thresh=args['ssd_visual_threshold'], labelmap=labelmap) #TODO VPY: MEAN ?!


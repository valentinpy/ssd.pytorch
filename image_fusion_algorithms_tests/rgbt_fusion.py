"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import torch
import argparse
from data import BaseTransform
from data.kaist import KAISTAnnotationTransform, KAISTDetection
from data.kaist import KAIST_CLASSES as KAISTlabelmap
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from config.parse_config import *
import math
import scipy
import scipy.signal

import random

def arg_parser():
    parser = argparse.ArgumentParser(description='Image fusion tests')
    parser.add_argument('--image_set_day', default=None, help='Imageset day')
    parser.add_argument('--image_set_night', default=None, help='Imageset night')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    args = parser.parse_args()
    return args


def normalize(img):
    img -= np.min(img)
    img = img / np.max(img) * 255  # normalize
    return img


def intinze(img):
    return img.astype(int)


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def gray_invert(img):
    return 255-img


def myplot(legends, *argv):
    plt.suptitle('Image: ' + repr(index))
    plt.subplots_adjust(left=0.001, bottom=0.001, right=0.999, top=0.95, wspace=0.001, hspace=0.1)
    i=0
    for arg in argv:
        arg = normalize(arg)
        arg = intinze(arg)
        ndim1 = math.ceil(1*math.sqrt(len(argv)))
        ndim2 = math.ceil(len(argv)/ndim1)
        plt.subplot(ndim2, ndim1, i+1)
        plt.title((legends[i]))
        plt.axis('off')
        plt.imshow(arg, interpolation='nearest')
        i+=1
    plt.waitforbuttonpress()


if __name__ == '__main__':

    # parse arguments
    args = vars(arg_parser())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}
    del config

    # prepare environnement
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Maximize figure
    fig = plt.figure("Test")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    #prepare datasets
    labelmap = KAISTlabelmap
    dataset_mean = (104, 117, 123)  # TODO VPY and for kaist ?
    dataset_day = KAISTDetection(root=args['dataset_root'],image_set=args['image_set_day'], transform=BaseTransform(300, dataset_mean), target_transform=KAISTAnnotationTransform(output_format="SSD"), dataset_name="KAIST", output_format="SSD")
    dataset_night = KAISTDetection(root=args['dataset_root'], image_set=args['image_set_night'], transform=BaseTransform(300, dataset_mean), target_transform=KAISTAnnotationTransform(output_format="SSD"), dataset_name="KAIST", output_format="SSD")

    # loop for all test images
    num_images = min(len(dataset_day), len(dataset_night))
    r = list(range(num_images))
    random.shuffle(r)
    for i in r:
        #---------------------------------------------
        # read images
        # ---------------------------------------------
        index = i
        if i%2 == 0:
            #day
            print("\nPulling DAY image: index: {}".format(index))
            img_visible_orig = dataset_day.pull_visible_image(index)
            img_lwir_orig = dataset_day.pull_raw_lwir_image(index)
            img_size = img_visible_orig.shape[::-1][1:3]
        else:
            #night
            print("\nPulling NGT image: index: {}".format(index))
            img_visible_orig = dataset_night.pull_visible_image(index)
            img_lwir_orig = dataset_night.pull_raw_lwir_image(index)
            img_size = img_visible_orig.shape[::-1][1:3]

        #---------------------------------------------
        # normalize before
        # ---------------------------------------------
        img_visible = normalize(img_visible_orig)
        img_lwir = normalize(img_lwir_orig)

        #---------------------------------------------
        # little cooking #TODO
        # ---------------------------------------------
        img_lwir_inverted = gray_invert(img_lwir)

        # p2, p98 = np.percentile(img_lwir_inverted/255, (2, 98))
        # lwir_eq0 = normalize(exposure.rescale_intensity(img_lwir_inverted/255, in_range=(p2, p98)))
        # lwir_eq1 = normalize(exposure.equalize_hist(img_lwir_inverted))

        # # Adaptive Equalization
        lwir_eq2 = normalize(exposure.equalize_adapthist(img_lwir_inverted/255, clip_limit=0.03))

        # convolved = scipy.signal.convolve2d(rgb2gray(img_visible_orig), rgb2gray(img_lwir_inverted))

        img_fused_add = img_visible_orig + img_lwir_inverted
        img_fused_mul = img_visible_orig * img_lwir_inverted

        #---------------------------------------------
        # plot
        # --------------------------------------------
        myplot(['img_visible_orig', 'img_lwir_orig', 'img_lwir_inverted', 'img_fused_add', 'img_fused_mul'], img_visible_orig, img_lwir_orig,
               img_lwir_inverted, img_fused_add, img_fused_mul)
        # myplot(['img_visible_orig', 'img_lwir_orig', 'lwir_eq2', 'img_fused_add', 'img_fused_mul', 'convolved'], img_visible_orig, img_lwir_orig, lwir_eq2, img_fused_add, img_fused_mul, convolved)

    print("finished")
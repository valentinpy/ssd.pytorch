"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import torch
import argparse

from data import BaseTransform
from data import KAISTAnnotationTransform, KAISTDetection
from data import KAIST_CLASSES as KAISTlabelmap
from eval.get_GT import get_GT

import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte

from utils.misc import str2bool
import random

def arg_parser():
    parser = argparse.ArgumentParser(description='Image fusion tests')
    parser.add_argument('--image_set_day', default=None, help='Imageset day')
    parser.add_argument('--image_set_night', default=None, help='Imageset night')
    parser.add_argument('--dataset_root', default=None, help='Location of dataset root directory')
    args = parser.parse_args()
    return args


def normalize(img):
    img -= np.min(img)
    img = img / np.max(img) * 255  # normalize
    return img

def intinze(img):
    return img.astype(int)

def gray_invert(img):
    return 255-img

def myplot(img1, img2, img3, img4, img5, img6, legend1=None, index=-1, legend2=None, legend3=None, legend4=None, legend5=None, legend6=None):

    img1 = intinze(img1)
    img2 = intinze(img2)
    img3 = intinze(img3)
    img4 = intinze(img4)

    # fig = plt.figure("Test")
    plt.suptitle('Image: ' + repr(index))

    plt.subplot(231)
    plt.title((legend1))
    plt.axis('off')
    plt.imshow(img1, interpolation='nearest')

    plt.subplot(232)
    plt.title((legend2))
    plt.axis('off')
    plt.imshow(img2, interpolation='nearest')

    plt.subplot(233)
    plt.title((legend3))
    plt.axis('off')
    plt.imshow(img3, interpolation='nearest')

    plt.subplot(234)
    plt.title((legend4))
    plt.axis('off')
    plt.imshow(img4, interpolation='nearest')

    plt.subplot(235)
    plt.title((legend5))
    plt.axis('off')
    plt.imshow(img5, interpolation='nearest')

    plt.subplot(236)
    plt.title((legend6))
    plt.axis('off')
    plt.imshow(img6, interpolation='nearest')


    # plt.show()
    plt.waitforbuttonpress()

if __name__ == '__main__':

    # parse arguments
    args = arg_parser()

    # prepare environnement
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    #Maximize figure
    fig = plt.figure("Test")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    #prepare datasets
    labelmap = KAISTlabelmap
    dataset_mean = (104, 117, 123)  # TODO VPY and for kaist ?
    dataset_day = KAISTDetection(root=args.dataset_root,image_set=args.image_set_day, transform=BaseTransform(300, dataset_mean), target_transform=KAISTAnnotationTransform(), dataset_name="KAIST")
    dataset_night = KAISTDetection(root=args.dataset_root, image_set=args.image_set_night, transform=BaseTransform(300, dataset_mean), target_transform=KAISTAnnotationTransform(), dataset_name="KAIST")

    # loop for all test images
    num_images = min(len(dataset_day), len(dataset_night))
    r = list(range(num_images))
    random.shuffle(r)
    for i in r:
        #---------------------------------------------
        # read images
        # ---------------------------------------------
        index = i#int(i/2)
        if i%2 == 0:
            #day
            print("\nPulling DAY image: index: {}".format(index))
            img_visible_orig = dataset_day.pull_visible_image(index)
            img_lwir_orig = dataset_day.pull_lwir_image(index)
            img_size = img_visible_orig.shape[::-1][1:3]
        else:
            #night
            print("\nPulling NGT image: index: {}".format(index))
            img_visible_orig = dataset_night.pull_visible_image(index)
            img_lwir_orig = dataset_night.pull_lwir_image(index)
            img_size = img_visible_orig.shape[::-1][1:3]
        # get image + annotations + dimensions

        #---------------------------------------------
        # normalize before
        # ---------------------------------------------
        img_visible = normalize(img_visible_orig)
        img_lwir = normalize(img_lwir_orig)

        #---------------------------------------------
        # little cooking #TODO
        # ---------------------------------------------
        img_lwir = gray_invert(img_lwir)


        p2, p98 = np.percentile(img_lwir/255, (2, 98))
        lwir_eq0 = normalize(exposure.rescale_intensity(img_lwir/255, in_range=(p2, p98)))

        lwir_eq1 = normalize(exposure.equalize_hist(img_lwir))

        # # Adaptive Equalization
        lwir_eq2 = normalize(exposure.equalize_adapthist(img_lwir/255, clip_limit=0.03))

        img_fused = img_visible_orig + img_lwir
        #---------------------------------------------
        # normalize after
        # ---------------------------------------------
        img_visible = normalize(img_visible)
        img_lwir = normalize(img_lwir)
        img_fused = normalize(img_fused)


        #---------------------------------------------
        # plot
        # --------------------------------------------
        myplot(img_visible_orig, img_visible, img_lwir, lwir_eq2, img5, img6, index=index, legend1='img_visible_orig', legend2='img_visible', legend3='img_lwir', legend4='lwir_eq2', legend5='', legend6='')
        #myplot(img_lwir, lwir_eq0, lwir_eq1,lwir_eq2, index=index, legend1='img_lwir', legend2='lwir_eq0', legend3='lwir_eq1', legend4='lwir_eq2')


    print("finished")

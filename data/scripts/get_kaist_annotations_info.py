import torch
import argparse
import numpy as np

from data import BaseTransform
from data.kaist import KAISTAnnotationTransform, KAISTDetection
from data.kaist import KAIST_CLASSES as KAISTlabelmap
from eval.get_GT import get_GT


def arg_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--image_set', default=None, help='Imageset')
    parser.add_argument('--dataset_root', default=None, help='Location of dataset root directory')
    args = parser.parse_args()
    return args

def get_annotations_info(ground_truth):

    number_classes = 0
    number_annotations = 0
    number_images = 0

    aspect_ratio = []

    min_aspect_ratio = np.inf
    max_aspect_ratio = 0
    mean_aspect_ratio = 0
    std_aspect_ratio = 0

    for gt_class in ground_truth[1:]: #remove background (class #0)
        number_classes += 1
        number_images = 0
        for img_key, img_annots in gt_class.items():
            number_images += 1
            number_annotations += img_annots['bbox'].shape[0]
            for bbox in img_annots['bbox']:
                #print(bbox)
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                dx = xmax - xmin
                dy = ymax - ymin
                aspect_ratio.append(dy / dx)

    apect_ratio = np.asanyarray(aspect_ratio)

    print("mean aspect ration: {}".format(np.mean(apect_ratio)))
    print("aspect ratio range [1 sigma => 0.682]: [{};{}]".format(np.mean(apect_ratio) - np.std(apect_ratio), np.mean(apect_ratio) + np.std(apect_ratio)))
    print("aspect ratio range [2 sigma => 0.954]: [{};{}]".format(np.mean(apect_ratio) - (2 * np.std(apect_ratio)), np.mean(apect_ratio) + (2 * np.std(apect_ratio))))
    print("aspect ratio range [3 sigma => 0.996]: [{};{}]".format(np.mean(apect_ratio) - (3*np.std(apect_ratio)), np.mean(apect_ratio) +(3 * np.std(apect_ratio))))
    results = {'number_classes': number_classes,
               'number_annotations' : number_annotations,
               'number_images' : number_images,
               'min_aspect_ratio': np.min(apect_ratio),
               'max_aspect_ratio': np.max(apect_ratio),
               'mean_aspect_ratio': np.mean(apect_ratio),
               'std_aspect_ratio': np.std(apect_ratio)
               }

    print(results)
    return results

if __name__ == '__main__':

    # parse arguments
    args = arg_parser()

    # prepare environnement
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    labelmap = KAISTlabelmap
    dataset_mean = (104, 117, 123)  # TODO VPY and for kaist ?
    dataset = KAISTDetection(root=args.dataset_root,image_set=args.image_set, transform=BaseTransform(300, dataset_mean), target_transform=KAISTAnnotationTransform(output_format='SSD'), dataset_name="KAIST")

    print('Read GT')
    ground_truth = get_GT(dataset, labelmap)

    print("Annotations infos:")
    annotations_infos = get_annotations_info(ground_truth=ground_truth)




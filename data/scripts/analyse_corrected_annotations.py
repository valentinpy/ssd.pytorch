import torch
import argparse
import numpy as np

from data import BaseTransform
from data.kaist import KAISTAnnotationTransform, KAISTDetection
from data.kaist import KAIST_CLASSES as KAISTlabelmap
from eval.get_GT import get_GT
import sys

def arg_parser():
    parser = argparse.ArgumentParser(description='Corrected annotations infos')
    parser.add_argument('--image_set', default=None, help='Imageset')
    parser.add_argument('--dataset_root', default=None, help='Location of dataset root directory')
    args = parser.parse_args()
    return args

def get_annotations_info(ground_truth):

    number_classes = 0
    number_annotations = 0

    aspect_ratio = []


    to_remove = 0

    for gt_class in ground_truth[1:]: #remove background (class #0)
        number_classes += 1
        number_images = 0
        for img_key, img_annots in gt_class.items():
            number_images += 1
            number_annotations += img_annots['bbox'].shape[0]
            if (img_annots['bbox'].shape[0]) == 0:
                 sys.exit(-1)
            for raw_annot in img_annots['raw_annotations']:
                person_detected = False
                people_detected = False
                person_not_sure_detected = False
                cyclist_detected = False
                only_person_detected = False
                occlusion_detected = False
                too_small = False

                # for each file, exact flags
                if raw_annot[0] == 'person':  # only keep images which contains a "person"
                    person_detected = True
                elif raw_annot[0] == 'people':
                    people_detected = True
                    print("people detected")
                elif raw_annot[0] == 'person?':
                    person_not_sure_detected = True
                    print("person not sure detected")
                elif raw_annot[0] == 'cyclist':
                    cyclist_detected = True
                    print("cyclist detected")
                else:
                    print("Annotation not recognized!")
                    sys.exit(-1)

                if int(raw_annot[5]) != 0:
                    occlusion_detected = True
                    print("occlusion detected")

                if int(raw_annot[4]) < 55:
                    too_small = True
                    print("too small")

                if (person_detected) and (not person_not_sure_detected) and (not people_detected) and (not cyclist_detected):
                    only_person_detected = True

                # according to flags, do we keep this entry ?
                keep_file = only_person_detected and (not occlusion_detected) and (not too_small) and person_detected

                if not keep_file:
                    to_remove+=1


            for bbox in img_annots['bbox']:
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                dx = xmax - xmin
                dy = ymax - ymin
                aspect_ratio.append(dy / dx)

    apect_ratio = np.asanyarray(aspect_ratio)

    print("to_remove: {}".format(to_remove))
    return None

if __name__ == '__main__':

    # parse arguments
    args = arg_parser()

    # prepare environnement
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    labelmap = KAISTlabelmap
    dataset_mean = (104, 117, 123)  # TODO VPY and for kaist ?
    dataset = KAISTDetection(root=args.dataset_root,image_set=args.image_set, transform=BaseTransform(300, dataset_mean), target_transform=KAISTAnnotationTransform(output_format='SSD'), dataset_name="KAIST", corrected_annotations=True)

    print('Read GT')
    ground_truth = get_GT(dataset, labelmap)

    print("Annotations infos:")
    _ = get_annotations_info(ground_truth=ground_truth)




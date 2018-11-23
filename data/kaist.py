from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np

KAIST_ROOT = osp.join(HOME, 'data/kaist/')


class KAISTAnnotationTransform(object):
    """Transforms a KAIST annotation into a Tensor of bbox coords and label index
    Initialised with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or {"person": 0}
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
         target (annotation filename) : the target annotation to be made usable
             will be an filename including path
        Returns:
         a list containing lists of bounding boxes  [bbox coords, class name]
        """

        if not osp.exists(target):
            print("annotation file not found: {}".format(target))
            sys.exit(-1)

        res = []
        with open(target) as f: #open annoatation file and read all lines
            for line in f.readlines():
                if line.startswith("person "): # only one class supported: "person"

                    line = line.split(" ")
                    line[1] = float(line[1]) #convert coordinates to float: xmin, ymin, width, height
                    line[2] = float(line[2])
                    line[3] = float(line[3])
                    line[4] = float(line[4])

                    bbox = [(line[1]/width), (line[2]/height), (line[1] + line[3])/width, (line[2] + line[4])/height] # [xmin, ymin, xax, ymax], all coordinates are [0;1] => diveded by witdh or height

                    label_idx = self.class_to_ind[line[0]] # label index is always 0 as we have only the "person" class supported
                    bbox.append(label_idx)

                    res += [bbox]
                    #print("bounding box: {}".format(bbox))

        return res #return all annoations: [[x1min, y1min, x1max, y1max, label_idx=0], [...] ]

class KAISTDetection(data.Dataset):
    """KAIST Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to KAIST root folder. (root folder must contain
            "rgbt-ped-detection" folder)
        image_set (string): imageset to use
            (eg. microset.txt  test-all-01.txt  test-all-20.txt  test-day-01.txt
            test-day-20.txt  test-night-01.txt  test-night-20.txt
            train-all-02.txt  train-all-04.txt  train-all-20.txt
            VPY-test-day.txt  VPY-train-day.txt). This is the name of the file conatining the dataset, which must exist in imageSet folder
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices). Default is: KAISTAnnotationTransform
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
    """

    def __init__(self, root,
                 image_set='VPY-train-day.txt',
                 transform=None, target_transform=KAISTAnnotationTransform(),
                 dataset_name='KAIST'):
        print("{}: ImageSet used is : {}".format(dataset_name, image_set))
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'annotations', '%s', '%s', '%s.txt')
        self._img_vis_root_path = osp.join('%s', 'images', '%s', '%s', 'visible', '%s.jpg')
        self._img_lwir_root_path = osp.join('%s', 'images', '%s', '%s', 'lwir', '%s.jpg')

        self.ids = list()

        rootpath = osp.join(self.root, 'rgbt-ped-detection/data/kaist-rgbt/')
        for line in open(osp.join(rootpath, 'imageSets', image_set)): # read imageSet file and loop for each entry
            annofile = self._annopath % tuple([rootpath] + line.replace('\n', '').replace('\r', '').split('/')) #get annotation file for the current image
            with open(annofile) as f:
                for annoline in f.readlines(): #loop for each line in each annoation
                    if annoline.startswith("person "): # only keep images which contains a "person"
                        self.ids.append(tuple([rootpath] + line.replace('\n', '').replace('\r', '').split('/')))
                        break

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        #TODO VPY: Only visible image is loaded (no lwir)
        img_id = self.ids[index]

        #TODO VPY parse annotations => target
        target = self._annopath % img_id
        img = cv2.imread(self._img_vis_root_path % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        else:
            print("You are required to implement the target_transform method to read annotations!")
            sys.exit(-1)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        #TODO VPY: Only visible image is loaded (no lwir)
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._img_vis_root_path % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  (img_id, [bbox coords, label_id])
                eg: ('001718', [96, 13, 438, 332, 12])
        '''
        raise NotImplementedError
        # TODO VPY implement
        # img_id = self.ids[index]
        #anno = ET.parse(self._annopath % img_id).getroot()
        #gt = self.target_transform(anno, 1, 1)
        #return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

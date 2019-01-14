# from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np

KAIST_CLASSES = ('person', )

# KAIST_ROOT = osp.join(HOME, 'data/kaist/')

def detection_collate_KAIST_YOLO(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[1])
        targets.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), torch.stack(targets,0)


def detection_collate_KAIST_SSD(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[1])
        targets.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), targets

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

    def __init__(self, class_to_ind=None, keep_difficult=False, output_format=None):
        self.class_to_ind = class_to_ind or dict(
            zip(KAIST_CLASSES, range(len(KAIST_CLASSES))))
        self.keep_difficult = keep_difficult
        self.output_format=output_format

    def __call__(self, target, width, height):
        """
        Arguments:
         target (annotation filename) : the target annotation to be made usable
             will be an filename including path
        Returns:
         a list containing lists of bounding boxes[bbox coords, class name]
        """
        #print("[VPY] KAISTAnnotationTransform called: target: {}, width: {}, height: {}".format(target, width, height))

        if not osp.exists(target):
            print("annotation file not found: {}".format(target))
            sys.exit(-1)

        res = []
        raw_details = []
        if self.output_format in {"SSD", "VOC_EVAL"}:
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
                        raw_details += [line]
                        #print("bounding box: {}".format(bbox))

        elif self.output_format == "YOLO":
            res = np.zeros((50, 5))
            i = 0
            with open(target) as f:
                for line in f.readlines():
                    if line.startswith("person "):  # only one class supported: "person"
                        line = line.split(" ")
                        xmin = float(line[1]) / width
                        ymin = float(line[2]) / height
                        xmax = (float(line[1]) + float(line[3])) / width
                        ymax = (float(line[2]) + float(line[4])) / height

                        res[i, 0] = self.class_to_ind[line[0]]
                        res[i, 1] = ((xmax + xmin) / 2)
                        res[i, 2] = ((ymax + ymin) / 2)
                        res[i, 3] = (xmax - xmin)
                        res[i, 4] = (ymax - ymin)
                        i += 1
        else:
            print("VPY: must chose output format for KAISTAnnotationTransform()")
            raise NotImplementedError

        return res, raw_details #return all annoations: [[x1min, y1min, x1max, y1max, label_idx=0], [...] ]

class KAISTDetection(data.Dataset):
    #KAIST Detection Dataset Object
    def __init__(self, root,
                 image_set='VPY-train-day.txt',
                 transform=None, target_transform=None,
                 dataset_name='KAIST',
                 image_fusion=0,
                 corrected_annotations = False):
        """
        Kaist Dataset constructor
        :param root: (string): filepath to KAIST root folder. (root folder must contain "rgbt-ped-detection" folder)
        :param image_set: (string): imageset to use (eg. microset.txt). This is the name of the file containing the dataset, which must exist in imageSet folder
        :param transform: (callable, optional): transformation to perform on the input image
        :param target_transform: (callable, optional): transformation to perform on the target `annotation` (eg: take in caption string, return tensor of word indices). Default is: KAISTAnnotationTransform
        :param dataset_name: (string, optional): which dataset to load (default: 'KAIST')
        :param image_fusion: (int): type of fusion used: [0: visible] [1: LWIR] [2: inverted LWIR] [...]
        :param corrected_annotations: (bool, default: False) do we want to use corrected annotations ? (not really working yet)
        """
        print("{}: ImageSet used is : {}".format(dataset_name, image_set))
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.corrected_annotations = corrected_annotations
        if target_transform == None:
            print("VPY: must add mannually target_transform=KAISTAnnotationTransform(...)")
            raise NotImplementedError

        if self.corrected_annotations:
            self._annopath = osp.join('%s', 'annotations_corrected', '%s', '%s', '%s.txt')
        else:
            self._annopath = osp.join('%s', 'annotations', '%s', '%s', '%s.txt')

        self._img_vis_root_path = osp.join('%s', 'images', '%s', '%s', 'visible', '%s.jpg')
        self._img_lwir_root_path = osp.join('%s', 'images', '%s', '%s', 'lwir', '%s.jpg')
        self.image_fusion = image_fusion

        self.ids = list()

        # open imageSet file and add files which interrest us in the imageList (ids)
        rootpath = osp.join(self.root, 'rgbt-ped-detection/data/kaist-rgbt/')
        for line in open(osp.join(rootpath, 'imageSets', image_set)): # read imageSet file and loop for each entry
            if not line.startswith("#"): # remove comments
                annofile = self._annopath % tuple([rootpath] + line.replace('\n', '').replace('\r', '').split('/')) #get annotation file for the current image
                self.ids.append(tuple([rootpath] + line.replace('\n', '').replace('\r', '').split('/') + [line.replace('\n', '').replace('\r', '')]))

    def __getitem__(self, index):
        img_path, im, gt, h, w = self.pull_item(index)
        return img_path, im, gt, h, w

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = self._annopath % img_id[0:4]

        if self.image_fusion == 0:
            img = self.pull_visible_image(index)
        elif self.image_fusion == 1:
            img = self.pull_raw_lwir_image(index)
        elif self.image_fusion == 2:
            img = self.pull_raw_lwir_image(index)
            img = 255-img
        elif self.image_fusion == 3:
            img = self.pull_raw_RGBT_image(index)
        else:
            print("image fusion not handled")
            sys.exit(-1)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target, _ = self.target_transform(target, width, height)
        else:
            print("You are required to implement the target_transform method to read annotations!")
            sys.exit(-1)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if self.image_fusion < 3:
            # to rgb
            img = img[:, :, (2, 1, 0)]
        else:
            # to rgbt
            img = img[:, :, (2, 1, 0, 3)]

        return "not defined", torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''

        if self.image_fusion == 0:
            img = self.pull_visible_image(index)
        elif self.image_fusion == 1:
            img = self.pull_raw_lwir_image(index)
        elif self.image_fusion == 2:
            img = self.pull_raw_lwir_image(index)
            img = 255-img
        elif self.image_fusion == 3:
            img = self.pull_raw_RGBT_image(index)
        else:
            print("image fusion not handled")
            sys.exit(-1)

        return img

    def pull_visible_image(self, index):
        '''Returns the original image object at index in PIL form (visible image)

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show (RGB)
        Return:
            PIL img
        '''

        img_id = self.ids[index]
        img = cv2.imread(self._img_vis_root_path % img_id[0:4])

        # to rgb
        img = img[:, :, (2, 1, 0)]
        return img

    def pull_raw_lwir_image(self, index):
        '''Returns the original image object at index in PIL form (LWIR image)

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''

        img_id = self.ids[index]
        img = cv2.imread(self._img_lwir_root_path % img_id[0:4])
        return img

    def pull_raw_RGBT_image(self, index):
        '''Returns the original image object at index in PIL form (VISIBLE + LWIR image => RGBT)

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''

        img_id = self.ids[index]
        img_RGB = cv2.imread(self._img_vis_root_path % img_id[0:4])[:, :, (2, 1, 0)]
        img_T = cv2.imread(self._img_lwir_root_path % img_id[0:4], cv2.IMREAD_GRAYSCALE)
        img = np.dstack((img_RGB, img_T))
        return img


    def pull_img_id(self, index):
        img_id = self.ids[index]
        return img_id[4]

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
        img_id = self.ids[index]
        target = self._annopath % img_id[0:4]
        if self.target_transform is not None:
            gt, raw_details = self.target_transform(target, 1, 1) # TODO VPY OK ??
        else:
            print("no target transform function!")
            sys.exit(-1)
        return img_id[4], gt, raw_details

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

def parse_rec_kaist(filename):
    if not os.path.exists(filename):
        print("annotation file not found: {}".format(filename))
        sys.exit(-1)

    with open(filename) as f:  # open annoatation file and read all lines
        objects = []
        for line in f.readlines():
            if line.startswith("person "):  # only one class supported: "person"

                line = line.split(" ")
                line[1] = float(line[1])  # convert coordinates to float: xmin, ymin, width, height
                line[2] = float(line[2])
                line[3] = float(line[3])
                line[4] = float(line[4])

                bbox = [(line[1]), (line[2]), (line[1] + line[3]), (line[2] + line[4])]  # [xmin, ymin, xax, ymax], all coordinates are [0;width or height] => not divided by witdh or height

                obj_struct = {}
                obj_struct['name'] = line[0]
                #obj_struct['pose'] = "n/d"
                #obj_struct['truncated'] = "n/d"
                #obj_struct['difficult'] = "n/d"
                obj_struct['bbox'] = [int(line[1]),
                                      int(line[2]),
                                      int(line[1]) + int(line[3]),
                                      int(line[2]) + int(line[4])]
                objects.append(obj_struct)
    return objects

def compute_KAIST_dataset_mean(dataset_root, image_set):
    print("compute images mean")

    images_mean = np.zeros((3), dtype=np.float64)  # [0,0,0]
    #
    # # create batch iterator
    dataset_mean = KAISTDetection(root=dataset_root, image_set=image_set, transform=None)
    data_loader_mean = data.DataLoader(dataset_mean, 1, num_workers=1, shuffle=True, pin_memory=True)
    batch_iterator = iter(data_loader_mean)
    i = 0
    for i in range(len(dataset_mean)):  # while True:# iteration in range(args.start_iter, cfg['max_iter']):
        # for i in range(100):
        #     print("Debug: not all data!!!!!")
        try:
            # load train data
            image, _ = next(batch_iterator)
            images_mean += image[0].permute(1, 2, 0).numpy().mean(axis=(0, 1))
        except StopIteration:
            break

    images_mean = images_mean / i
    print("image mean is: {}".format(images_mean))

    return images_mean
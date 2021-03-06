import glob
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize

def detection_collate_COCO_YOLO(batch):
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
        targets.append(torch.DoubleTensor(sample[2]).float())
    return torch.stack(imgs, 0), torch.stack(targets, 0) #targets

class ImageFolder(Dataset):
    """
    VPY:
    Looks like this classe list all files in a folder and use those files (images) as dataset.
    Only image and path are returned, so it's meant to be used for inference and not for training/testing
    """
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        """
        VPY:
        :param index: index of the file [0; len]
        :return: tuple: (img_path, input_img)
                img_path: path of the image
                input_img: torch tensor of the image, padded and resized. Lokks like to be in BGR (see line 42), to be confirmed!
        """
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    """
    VPY:
    COCO dataset, with images and annotations initially for YOLO3
    """
    def __init__(self, list_path, img_size=416):
        """
        VPY
        :param list_path: imageSet file. A text file containing path to all images we want to retrieve
        :param img_size: standard image size for YOLO3 is 416
        """
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):
        """
        VPY
        Retrieve image path, image and annotations
        :param index: index of the image+... we want [0; len]
        :return: tuple(img_path, input_img, filled_labels)
            img_path: path of the image
            input_img: image as torch tensor, padded, resized (3 channels, even if gray levels), lokks like to be BGR (see line 106)
            filled_labels: torch tensor of [max_objects] (=50) labels coordinates; normalised [0..1] and each line as follow [x_center, y_center, width, height]
        """
        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path)) #VPY: IMG lokks like to be RGB yet

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        """
        Annotation file is structured as follow:
        1 line par annotation
        [label x_center, y_center, width, height]
        label => refer to coco.names
        dimensions are normalised: [0..1]
        """
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2) # xmin * w, without padding
            y1 = h * (labels[:, 2] - labels[:, 4]/2) # xmin * h, without padding
            x2 = w * (labels[:, 1] + labels[:, 3]/2) # xmax * w , without padding
            y2 = h * (labels[:, 2] + labels[:, 4]/2) # ymax * h, without padding
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            # labels coordinates are normalised [0..1] and as follow [x_center, y_center, width, height]
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        #print("Called Listdataset __getitem__({})\nReturning: img_path={}\nimput_img.shape: {}\nfilled_labels.shape: {}\n\n".format(index, img_path, input_img.shape, filled_labels.shape))
        return img_path, input_img, filled_labels, None, None

    def __len__(self):
        return len(self.img_files)

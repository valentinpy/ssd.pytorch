from utils.timer import Timer
from torch.autograd import Variable
import numpy as np
import torch
import datetime
from models.yolo3_utils import *


def forward_pass_ssd(net, cuda, dataset, labelmap):
    num_images = len(dataset)

    det_image_ids = [[] for _ in range(len(labelmap)+1)]
    det_BB = [np.empty((0,4)) for _ in range(len(labelmap)+1)]
    det_confidence = [np.empty((0)) for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    # loop for all test images
    for i in range(num_images):

        # get image + annotations + dimensions
        _, im, _, h, w = dataset[i]

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
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t() # create a mask to keep only detections with confidence > 0
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()

            img_id = dataset.pull_img_id(i)
            det_image_ids[j] += [img_id for _ in range(dets.size(0))]

            for l in range(dets.size(0)):
                det_BB[j] = np.vstack((det_BB[j], boxes[l].cpu().numpy()))
                det_confidence[j] = np.append(det_confidence[j], scores[l])

        if (i % 10 == 0):
            print('im_detect: {:d}/{:d}. Detection time per image: {:.3f}s'.format(i, num_images, detect_time))

    return det_image_ids, det_BB, det_confidence

def forward_pass_yolo(net, cuda, dataloader, classes, conf_thres, nms_thres, img_size, dataset, labelmap):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    num_images = len(dataset)
    det_image_ids = [[] for _ in range(len(labelmap) + 1)]
    det_BB = [np.empty((0, 4)) for _ in range(len(labelmap) + 1)]
    det_confidence = [np.empty((0)) for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    # loop for all test images
    for i in range(num_images):

        # get image + annotations + dimensions
        _, im, gt, h, w = dataset[i]

        # x = Variable(im.unsqueeze(0))
        # if cuda:
        #     x = x.cuda()

        # forward pass
        _t['im_detect'].tic()
        # detections = net(x).data

        input_imgs = Variable(im.type(Tensor).unsqueeze(0))
        img = np.transpose(input_imgs[0].cpu().numpy(), (1, 2, 0))

        # Get detections
        with torch.no_grad():
            detections = net(input_imgs)
            num_classes = len(classes)
            detections = non_max_suppression(detections, num_classes, conf_thres, nms_thres)  # (x1, y1, x2, y2, object_conf, class_score, class_pred)

        detect_time = _t['im_detect'].toc(average=True)

        for j in range(0, len(detections)):
            # dets = detections[0, j, :]
            # mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            # dets = torch.masked_select(dets, mask).view(-1, 5)
            # if dets.size(0) == 0:
            #     continue
            if detections[j] is not None:
                dets = detections[j]
                boxes = dets[:, 0:4]
                boxes[:, 0] *= w/img_size
                boxes[:, 2] *= w/img_size
                boxes[:, 1] *= h/img_size
                boxes[:, 3] *= h/img_size
                scores = dets[:, 4].cpu().numpy()

                img_id = dataset.pull_img_id(i)
                det_image_ids[j+1] += [img_id for _ in range(dets.size(0))]

                for l in range(dets.size(0)):
                    det_BB[j+1] = np.vstack((det_BB[j+1], boxes[l].cpu().numpy()))
                    det_confidence[j+1] = np.append(det_confidence[j+1], scores[l])

        if (i % 10 == 0):
            print('im_detect: {:d}/{:d}. Detection time per image: {:.3f}s'.format(i, num_images, detect_time))

    return det_image_ids, det_BB, det_confidence
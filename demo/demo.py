import cv2
import signal
from collections import OrderedDict

from utils.timer import Timer
from config.load_classes import load_classes
from models.yolo3 import *
from models.yolo3_utils import *
from models.vgg16_ssd import build_vgg_ssd
from models.mobilenet2_ssd import build_mobilenet_ssd

import torch.backends.cudnn as cudnn

from data import BaseTransform
from data.get_data import *

def add_bb(img, bbox, label):
    cv2.rectangle(img,
                  (int(bbox[0]), int(bbox[1])),
                  (int(bbox[2]), int(bbox[3])),
                  (0, 255, 0),
                  1)
    cv2.putText(img, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def add_img_title(img,title):
    cv2.putText(img, title, (int(img.shape[0]/2), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def test_net(model_name, net, cuda, dataset, conf_thres, nms_thres, classes, transform):
    print('\nPerforming object detection:')

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    _t = {'im_detect': Timer(), 'misc': Timer()}

    num_images = len(dataset)
    num_classes = len(classes)

    for i in range(num_images):
        _t['im_detect'].tic()

        # --------------------------
        # get image and GT
        # --------------------------

        #annotation
        img_id, annotation, _ = dataset.pull_anno(i)

        # raw images
        img_vis= dataset.pull_visible_image(i)
        img_lwir = 255 - dataset.pull_raw_lwir_image(i)

        #images on which we display GT
        img_gt_vis = img_vis.copy()
        img_gt_lwir = img_lwir.copy()

        # image on which we display detections
        img_det = img_lwir.copy()

        # image used for inference (resized,... with BaseTransform)
        _, img,_,_,_ = dataset[i]
        input_imgs = Variable(img.type(Tensor).unsqueeze(0))

        if cuda:
            input_imgs = input_imgs.cuda()

        if model_name in {"VGG_SSD", "MOBILENET2_SSD"}:
            y = net(input_imgs)
            detections = y.data

        elif model_name == "YOLO":
            with torch.no_grad():
                detections = net(input_imgs)
                detections = non_max_suppression(detections, num_classes, conf_thres, nms_thres)  # (x1, y1, x2, y2, object_conf, class_score, class_pred)

        # Log progress
        detect_time = _t['im_detect'].toc(average=True)
        print('Image: {}, Image inference Time: {}ms'.format(i, int(detect_time*1000)))


        # --------------------------
        # Ground truth
        # --------------------------
        gt_id_cnt = 0
        for box in annotation:
            gt_id_cnt += 1
            label = [key for (key, value) in dataset.target_transform.class_to_ind.items() if value == box[4]][0]

            img_gt_lwir = add_bb(img_gt_lwir, box,label)
            img_gt_vis = add_bb(img_gt_vis, box,label)

            img_gt_lwir = add_img_title(img_gt_lwir, "GT on raw LWIR")
            img_gt_vis = add_img_title(img_gt_vis, "GT on raw VIS")


        # --------------------------
        # Detections
        # --------------------------
        if model_name in {"VGG_SSD", "MOBILENET2_SSD"}:
            scale = torch.Tensor([img_gt_lwir.shape[1], img_gt_lwir.shape[0], img_gt_lwir.shape[1], img_gt_lwir.shape[0]])
            pred_num = 0
            for i in range(detections.size(1)):  # loop for all classes
                j = 0
                while detections[0, i, j, 0] >= conf_thres:  # loop for all detection for the corresponding class
                    score = detections[0, i, j, 0]
                    score = (score.data * 100).cpu().numpy().astype(int)
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    pred_num += 1
                    j += 1

                    img_det = add_bb(img_det, pt, str(score))


        elif model_name == "YOLO":
            scale = [img_gt_lwir.shape[1], img_gt_lwir.shape[0], img_gt_lwir.shape[1], img_gt_lwir.shape[0]]
            scale = [elem / 416 for elem in scale]
            if (detections is not None) and (detections[0] is not None):
                detections = detections[0].cpu().numpy()
                pred_num = 0
                for i in range(detections.shape[0]):
                    j = 0
                    if detections[i, 4] >= conf_thres:
                        score = detections[i, 4]
                        xmin = int(detections[i, 0] * scale[0])
                        xmax = int(detections[i, 2] * scale[2])
                        ymin = int(detections[i, 1] * scale[1])
                        ymax = int(detections[i, 3] * scale[3])
                        box = [xmin, ymin, xmax, ymax]
                        label_name = repr(score)
                        pred_num += 1
                        j += 1
                        img_det = add_bb(img_det, box, str(score))

        # --------------------------
        # Plot image, GT and dets
        # --------------------------
        raw = np.hstack((img_gt_lwir, img_gt_vis))
        with_bb = np.hstack((img_det, 0*img_det)) # bottom-right image is not used at the moment.
        all = np.vstack((raw, with_bb))
        cv2.imshow("GT || DET", all[:, :, (2, 1, 0)])
        cv2.waitKey(0)


def main(args):
    model_name = args["model"]
    cuda = torch.cuda.is_available() and args['cuda']

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    classes = load_classes(args['names'])
    num_classes = len(classes) + 1  # +1 for background

    # load net
    print("Loading weights from file: {}".format(args["trained_model"]))
    if model_name == "VGG_SSD":
        net = build_vgg_ssd(phase='test', size=300, num_classes=num_classes,cfg=args) # initialize SSD
        try:
            net.load_state_dict(torch.load(args['trained_model']))
        except:
            # If we try to load a model with "vgg" conv instead of "basenet" conv, as the name changed to be generic and we don't want to train again,
            # we just have to change names in the OrderDict loaded form file
            old_model = torch.load(args['trained_model'])
            new_model = OrderedDict()
            for key, value in old_model.items():
                new_model[key.replace("vgg", "basenet")] = value
            new_model._metadata = OrderedDict()
            for key, value in old_model._metadata.items():
                new_model._metadata[key.replace("vgg", "basenet")] = value
            net.load_state_dict(new_model)

    elif model_name == "MOBILENET2_SSD":
        net = build_mobilenet_ssd(phase='test', size=320, num_classes=num_classes, cfg=args)  # initialize SSD
        net.load_state_dict(torch.load(args['trained_model']))

    elif model_name == "YOLO":
        net = Darknet(args['yolo_model_config_path'], img_size=args['yolo_img_size'])
        net.load_weights(args['trained_model'])
    else:
        raise NotImplementedError

    print('Finished loading model!')

    net.eval()

    # load data
    dataset = get_dataset_demo(args)

    if cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    if model_name == "YOLO":
        conf_thresh = args['yolo_conf_thres']
        nms_thres = args['yolo_nms_thres']
        transform_fct = BaseTransform(args["yolo_img_size"], (104, 117, 123))#None

    elif model_name in {"VGG_SSD", "MOBILENET2_SSD"}:
        conf_thresh = args['ssd_visual_threshold']
        nms_thres = None
        transform_fct = BaseTransform(args["ssd_min_dim"], (104, 117, 123))
    else:
        raise NotImplementedError

    test_net(model_name=model_name, net=net, cuda=cuda, dataset=dataset, conf_thres=conf_thresh, nms_thres=nms_thres, classes=classes, transform=transform_fct)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    args = arg_parser(role="eval")
    check_args(args, role="eval")
    main(args)

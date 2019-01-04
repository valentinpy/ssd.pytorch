from __future__ import division
import datetime
import argparse
import cv2
from torch.utils.data import DataLoader
from utils.str2bool import str2bool
from models.yolo3 import *
from models.yolo3_utils import *
from data.coco_list import *
from data.kaist import KAISTDetection, KAISTAnnotationTransform, detection_collate_KAIST_YOLO
from augmentations.YOLOaugmentations import YOLOaugmentation
from config.load_classes import load_classes

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_set', default=None, help='Imageset')
    parser.add_argument('--trained_model', default=None, type=str, help='Trained state_dict file path to open')
    parser.add_argument('--image_fusion', default=-1, type=int, help='[KAIST]: type of image fusion: [0: visible], [1: lwir] [2: lwir inverted] [...]')  # TODO VPY update when required
    parser.add_argument('--corrected_annotations', default=False, type=str2bool, help='[KAIST] do we use the corrected annotations ? (must ahve compatible imageset (VPY-test-strict-type-5)')
    parser.add_argument("--data_config_path", type=str, default=None, help="path to data config file")
    args = parser.parse_args()
    return args

def test_net(net, dataloader, img_size, classes, conf_thres, nms_thres):
    print('\nPerforming object detection:')
    prev_time = time.time()
    for batch_i, (input_imgs, GT) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        img = np.transpose(input_imgs[0].cpu().numpy(), (1, 2, 0))
        img_gt = img.copy()
        img_det = img.copy()

        # Get detections
        with torch.no_grad():
            detections = net(input_imgs)
            num_classes = len(classes)
            detections = non_max_suppression(detections,num_classes, conf_thres, nms_thres)  # (x1, y1, x2, y2, object_conf, class_score, class_pred)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))
        # --------------------------
        # Ground truth
        # --------------------------
        GT = GT.cpu().numpy()[0]
        gt_id_cnt = 0
        for box in GT:
            gt_id_cnt += 1
            label = "TODO" # classes[int(box[0]) + 1]
            xc = int(img_size * box[1])
            yc = int(img_size * box[2])
            w = int(img_size * box[3])
            h = int(img_size * box[4])
            xmin = int(xc - w / 2)
            xmax = int(xc + w / 2)
            ymin = int(yc - h / 2)
            ymax = int(yc + h / 2)

            cv2.rectangle(img_gt,
                          (xmin, ymin),
                          (xmax, ymax),
                          (0, 255, 0),
                          1
                          )
            cv2.putText(img_gt, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # --------------------------
        # Detections
        # --------------------------
        # scale each detection back up to the image
        if (detections is not None) and (detections[0] is not None):
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            detections = detections[0].cpu().numpy()
            pred_num = 0
            for i in range(detections.shape[0]):
                j = 0
                if detections[i, 4] >= conf_thres:
                    score = detections[i, 4]
                    xmin = detections[i, 0]
                    xmax = detections[i, 2]
                    ymin = detections[i, 1]
                    ymax = detections[i, 3]
                    label_name = "TODO"
                    pred_num += 1
                    j += 1

                    cv2.rectangle(img_det,
                                  (xmin, ymin),
                                  (xmax, ymax),
                                  (255, 0, 0),
                                  1
                                  )
                    cv2.putText(img_det, label_name, (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # --------------------------
        # Plot image, GT and dets
        # --------------------------
        cv2.imshow("GT || DET", np.hstack((img_gt, img_det))[:, :, (2, 1, 0)])
        cv2.waitKey(0)
        # print("wait")

if __name__ == '__main__':
    args = vars(arg_parser())
    config = parse_data_config(args['data_config_path'])
    args = {**args, **config}
    del config

    # load data
    if args['name'] == 'COCO':
        dataset = ListDataset(list_path=args['validation_set'], img_size=args['yolo_img_size'])
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args["num_workers"], collate_fn=detection_collate_COCO_YOLO)
    elif args['name'] == 'KAIST':
        kaist_root = args["dataset_root"]
        image_set = args["image_set"]
        image_fusion = args["image_fusion"]

        dataset = KAISTDetection(root=kaist_root, image_set=image_set, transform=YOLOaugmentation(args['yolo_img_size']), image_fusion=image_fusion,
                                 output_format="YOLO", target_transform=KAISTAnnotationTransform(output_format="YOLO"))

        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args['num_workers'], collate_fn=detection_collate_KAIST_YOLO)

    else:
        raise NotImplementedError

    cuda = torch.cuda.is_available() and args['cuda']

    # Set up model
    model = Darknet(args['yolo_model_config_path'], img_size=args['yolo_img_size'])
    model.load_weights(args['trained_model'])

    if cuda:
        model.cuda()

    model.eval()  # Set in evaluation mode
    classes = load_classes(args['names'])  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    test_net(net=model, dataloader=dataloader, img_size=args['yolo_img_size'], classes=classes, conf_thres=args['yolo_conf_thres'], nms_thres=args['yolo_nms_thres'])#, testset=testset):#, transform=BaseTransform(net.size, (104, 117, 123)), thresh=args['ssd_visual_threshold'], labelmap=labelmap)  # TODO VPY: MEAN ?!


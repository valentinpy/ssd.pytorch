import torch
from torch.utils.data import DataLoader

from data.kaist import detection_collate_KAIST_SSD
from data.kaist import detection_collate_KAIST_YOLO
from data.kaist import KAISTDetection, KAISTAnnotationTransform
from data.coco_list import ListDataset, detection_collate_COCO_YOLO
from data.voc0712 import VOCDetection, detection_collate_VOC
from data.voc0712 import VOCAnnotationTransform

from augmentations.YOLOaugmentations import YOLOaugmentation
from augmentations.SSDaugmentations import SSDAugmentation
from augmentations.SSDaugmentations import SSDAugmentationDemo


def get_dataset_demo(args):
    model_name = args["model"]
    dataset_name = args["name"]

    if dataset_name == 'VOC':
        dataset = VOCDetection(root=args['dataset_root'], image_sets=[('2007', 'test')], transform=None, target_transform=VOCAnnotationTransform())
    elif dataset_name == 'KAIST':
        transform_fct = YOLOaugmentation(args['yolo_img_size']) if model_name == "YOLO" else SSDAugmentationDemo(args["ssd_min_dim"])
        dataset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=transform_fct,
                                 image_fusion=args['image_fusion'], corrected_annotations=args['corrected_annotations'],
                                 target_transform=KAISTAnnotationTransform(output_format="VOC_EVAL"))
    elif dataset_name == 'COCO':
        dataset = ListDataset(list_path=args['validation_set'], img_size=args['yolo_img_size'])
    else:
        raise NotImplementedError
    return dataset

def get_dataset_dataloader_train(args):
    model_name = args["model"]
    dataset_name = args["name"]
    image_fusion = args["image_fusion"]

    if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
        batch_size = args["ssd_batch_size"]
        ssd_min_dim = args["ssd_min_dim"]
    elif model_name == "YOLO":
        batch_size = args["yolo_batch_size"]
        ssd_min_dim = None
    else:
        raise NotImplementedError

    if dataset_name == 'VOC':
        dataset = VOCDetection(root=args['dataset_root'], transform=SSDAugmentation(ssd_min_dim))
        data_loader = DataLoader(dataset, batch_size, num_workers=args['num_workers'], shuffle=True, collate_fn=detection_collate_VOC,
                                 pin_memory=True)
    elif dataset_name == 'KAIST':
        kaist_root = args["dataset_root"]
        image_set = args["image_set"]

        if model_name in {"MOBILENET2_SSD", "VGG_SSD"}:
            dataset = KAISTDetection(root=args['dataset_root'], image_set=args['image_set'], transform=SSDAugmentation(ssd_min_dim),
                                     image_fusion=args['image_fusion'], target_transform=KAISTAnnotationTransform(output_format="SSD"))
            data_loader = DataLoader(dataset, batch_size, num_workers=args['num_workers'], shuffle=True, collate_fn=detection_collate_KAIST_SSD,
                                     pin_memory=True)
        elif model_name == "YOLO":
            dataset = KAISTDetection(root=kaist_root, image_set=image_set, transform=YOLOaugmentation(args['yolo_img_size']),
                                     image_fusion=image_fusion, target_transform=KAISTAnnotationTransform(output_format="YOLO"))
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,  # True,
                num_workers=args['num_workers'],
                collate_fn=detection_collate_KAIST_YOLO)
        else:
            raise NotImplementedError

    elif dataset_name == "COCO":
        dataset = ListDataset(args["train_set"])
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args['num_workers'],
            collate_fn=detection_collate_COCO_YOLO)
    else:
        raise NotImplementedError

    return data_loader, dataset
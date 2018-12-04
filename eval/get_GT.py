import numpy as np

def get_GT(dataset, labelmap):
    num_images = len(dataset)

    gt_all_classes_recs = [{} for _ in range(len(labelmap)+1)]

    #loop for all classes
    # skip j = 0, because it's the background class
    for j in range(1, len(labelmap)+1):

        # extract gt objects for this class
        gt_class_recs = {}

        #read all images
        for i in range(num_images):
            img_id, gt, detailed_gt = dataset.pull_anno(i)

            bbox = np.empty((0,4))
            gt_difficult = []

            if dataset.name == "VOC":
                for g in detailed_gt:
                    if (g['name'] == labelmap[j - 1]):
                        bbox = np.append(bbox, np.array([g['bbox']]), axis=0)
                        gt_difficult.append(g['difficult'])
                gt_difficult = np.array(gt_difficult).astype(np.bool)
                det = [False] * len(gt_difficult)

            elif dataset.name == "KAIST":
                for g in gt:
                    if (g[4]==j - 1):
                        bbox = np.append(bbox, np.array([g[0:4]]), axis=0)
                        gt_difficult.append(False)
                gt_difficult = np.array(gt_difficult).astype(np.bool)
                det = [False] * len(gt_difficult)

            else:
                print("Dataset not implemented")
                raise NotImplementedError

            gt_class_recs[img_id] = {'bbox': bbox, 'difficult': gt_difficult, 'det': det}
        gt_all_classes_recs[j] = gt_class_recs

    return gt_all_classes_recs
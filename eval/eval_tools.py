import numpy as np
from eval.voc_ap import voc_ap


def eval(gt_class_recs, det_BB, det_image_ids, det_confidence, labelmap, use_voc07_metric=True, verbose=True):
    if verbose:
        print("\n----------------------------------------------------------------")
        print("Eval")
        print("----------------------------------------------------------------")
        print('VOC07 metric? ' + ('Yes' if use_voc07_metric else 'No'))
    aps = []
    aps_dict = {}
    tpfp_dict = {}

    # reset detections
    for i, class_rec in enumerate(gt_class_recs):
        for j, value in class_rec.items():
            for k, det in enumerate(value['det']):
                gt_class_recs[i][j]['det'][k] = False

    for i, cls in enumerate(labelmap):
        #rec, prec, ap, npos, tp, fp_thresh, miss = voc_eval_class(gt_class_recs[i+1], det_BB[i+1], det_image_ids[i+1], det_confidence[i+1], ovthresh=0.5, use_07_metric=use_voc07_metric)
        rec, prec, ap = voc_eval_class(gt_class_recs[i + 1], det_BB[i + 1], det_image_ids[i + 1], det_confidence[i + 1], ovthresh=0.5, use_07_metric=use_voc07_metric)
        aps += [ap]
        aps_dict[cls] = ap
        # tpfp_dict[cls] = {'npos': npos, 'tp': tp, 'fp_thresh':fp_thresh, 'miss': miss}
        if verbose:
            print('AP for {} = {:.4f}'.format(cls, ap))

    mAP = np.mean(aps)
    if verbose:
        print('Mean AP = {:.4f}'.format(mAP))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')

    # return mAP, aps_dict, tpfp_dict
    return mAP, aps_dict

def voc_eval_class(gt_class_recs, det_BB, det_image_ids, det_confidence, ovthresh=0.5, use_07_metric=True):
    # TODO VPY document!

    npos = len([x for _, value in gt_class_recs.items() for x in value['difficult'] if x == False])

    # sort by confidence
    det_sorted_ind = np.argsort(-det_confidence)
    det_sorted_scores = np.sort(-det_confidence)
    det_BB = det_BB[det_sorted_ind, :]
    det_image_ids = [det_image_ids[x] for x in det_sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(det_image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # fp_thresh = np.zeros(nd)
    # miss = npos

    # for each detection
    for d in range(nd):
        #find ground truth corresponding to the image used for detections
        gt_R = gt_class_recs[det_image_ids[d]]

        det_bb = det_BB[d, :].astype(float)
        ovmax = -np.inf
        gt_BBGT = gt_R['bbox'].astype(float)

        # if the ground truth contains annotations (should aways be with custom kaist imageset)
        if gt_BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_BBGT[:, 0], det_bb[0])
            iymin = np.maximum(gt_BBGT[:, 1], det_bb[1])
            ixmax = np.minimum(gt_BBGT[:, 2], det_bb[2])
            iymax = np.minimum(gt_BBGT[:, 3], det_bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((det_bb[2] - det_bb[0]) * (det_bb[3] - det_bb[1]) +
                   (gt_BBGT[:, 2] - gt_BBGT[:, 0]) *
                   (gt_BBGT[:, 3] - gt_BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            # fp_thresh[d] = 1
            if not gt_R['difficult'][jmax]:
                if not gt_R['det'][jmax]:
                    tp[d] = 1.
                    gt_R['det'][jmax] = 1
                    # miss -= 1
                    # fp_thresh[d] = 0
                else:
                    fp[d] = 1.

        else:
            fp[d] = 1.
    print("ovthresh: {}, npos: {}, fp_thresh: {}, miss: {}".format(ovthresh, npos,  max(np.cumsum(fp_thresh)), miss))

    # compute precision recall
    fp = np.cumsum(fp)
    # fp_thresh = np.cumsum(fp_thresh)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap#, npos, max(tp), max(fp_thresh), miss
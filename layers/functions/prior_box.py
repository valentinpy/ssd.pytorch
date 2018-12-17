from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
from layers.box_utils import point_form, center_size


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['ssd_min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['ssd_aspect_ratios'])
        self.variance = cfg['ssd_variance'] or [0.1]
        self.feature_maps = cfg['ssd_feature_maps']
        self.min_sizes = cfg['ssd_min_sizes']
        self.max_sizes = cfg['ssd_max_sizes']
        self.steps = cfg['ssd_steps']
        self.aspect_ratios = cfg['ssd_aspect_ratios']
        self.clip = cfg['ssd_clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #TODO VPY
                # We keep squares and vertical boxes, and remove horizonal ones

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                # VPY: only vertical boxes keptq!
                for ar in self.aspect_ratios[k]:
                    #mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)] # VPY: remove horizontal boxes
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output = point_form(output)
            output.clamp_(max=1, min=0)
            output = center_size(output)
        return output

    def demo(self, mode="all"):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                if mode == "all":
                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k]/self.image_size
                    mean += [cx, cy, s_k, s_k]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                elif mode == "vert_med":
                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k] / self.image_size
                    mean += [cx, cy, s_k, s_k]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        #mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                        mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

                elif mode == "vert_only":
                    # rest of aspect ratios
                    s_k = self.min_sizes[k] / self.image_size
                    for ar in self.aspect_ratios[k]:
                        # mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                        mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output = point_form(output)
            output.clamp_(max=1, min=0)
            output = center_size(output)

        import numpy as np
        import cv2

        for i in output:
            i = i.cpu().numpy()
            i*=1000
            cx=i[0]
            cy=i[1]
            dx=i[2]
            dy=i[3]
            xmin = cx - dx
            xmax = cx + dx
            ymin = cy - dy
            ymax = cy + dy

            if (cx > 200) and (cy > 200) and (cx < 800) and (cy < 800):
                print("new")
                print(i)
                print(xmin, xmax, ymin, ymax)

                data = np.zeros((1000, 1000))
                data= cv2.rectangle(data, (xmin, ymin), (xmax, ymax), 255)
                cv2.imshow('test', data)
                cv2.waitKey(200)

        cv2.destroyAllWindows()

        return output
#!/bin/bash

wget https://pjreddie.com/media/files/yolov3.weights
wget -O mobilenet2.pth https://storage.googleapis.com/models-hao/mb2-imagenet-71_8.pth mobilenet2.pth
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
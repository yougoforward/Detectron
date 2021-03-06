#!/usr/bin/env python2
#coding: utf-8
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
#
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
import math
#
sys.path.append('/home/vpa/github/caffe2/build')
from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
#
c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
# import sys
reload(sys)
sys.setdefaultencoding('utf8')

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='/home/vpa/github/Detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/home/vpa/models/detectron/e2e_faster_rcnn_R-50-FPN_2x.pkl',
        type=str
    )
    # parser.add_argument(
    #     '--output-dir',
    #     dest='output_dir',
    #     help='directory for visualization pdfs (default: /tmp/infer_simple)',
    #     default='/tmp/infer_simple',
    #     type=str
    # )
    # parser.add_argument(
    #     '--image-ext',
    #     dest='image_ext',
    #     help='image file name extension (default: jpg)',
    #     default='jpg',
    #     type=str
    # )
    # parser.add_argument(
    #     'im_or_folder', help='image or folder of images',
    # )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def get_class_string(class_index, score, class_names):
    class_text = class_names[class_index] if class_names is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')

def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    u,v = (pos[0]+pos[2])/2.0, 600-pos[3]
    # 0.0230 * u - ((0.9996 * u - 550.3179) * (37.6942 * v - 2.2244e+06)) / (1.6394e+03 * v - 4.1343e+05) - 12.9168
    #
    # ((0.0070 * u - 1.6439e+03) * (37.6942 * v - 2.2244e+06)) / (1.6394e+03 * v - 4.1343e+05) - 1.6046e-04 * u + 0.0902




    txt = class_str
    # cv2 to pil
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pil_im = Image.fromarray(cv2_im)
    # draw pil
    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
    font = ImageFont.truetype("/usr/share/fonts/truetype/simhei.ttf", 20, encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小
    draw.text((x0, y0-20), txt, (0, 0, 255), font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
    # pil to cv2
    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imshow("detect", img)


    # Compute text size.
    # txt = class_str
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    # back_tl = x0, y0 - int(1.3 * txt_h)
    # back_br = x0 + txt_w, y0
    # cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    # txt_tl = x0, y0 - int(0.3 * txt_h)
    # cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, thick=1, color=_GREEN):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img

def demo_vis_one_imageboxes_opencv(im, cls_boxes, thresh=[], show_box=False,dataset=None, show_class=False,
                                   class_names=[], color_list=[], cls_sel=[]):
    """Constructs a numpy array with the detections visualized."""

    box_list = [b for b in [cls_boxes[i] for i in cls_sel] if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    classes = []

    # for j in range(len(cls_boxes)):
    for j in cls_sel:
        # print(len(cls_boxes[j]))
        classes += [j] * len(cls_boxes[j])

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < min(thresh):
        return im

    # for i in sorted_inds:
    for i, cls_id in enumerate(classes[0:]):
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh[cls_id]:
            continue

        # show box (off by default)
        if show_box:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), color=color_list[cls_id])

        # show class (off by default)
        if show_class:
            class_str = get_class_string(classes[i], score, class_names)
            im = vis_class(im, (bbox[0], bbox[1], bbox[2], bbox[3]), class_str)

    return im

def detection(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    start_time = time.time()
    count = 0
    # class_names =[
    #     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #     'bus', 'train', 'truck']
    # color_list=[[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,255,0],[255,0,255],[255,255,255]]

    class_names = [
        '__background__', u'人'.encode('utf-8').decode('utf-8'), 'bicycle', u'车'.decode(), 'motorcycle', 'airplane',
        'car', 'train', 'car']
    color_list = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 255], [255, 255, 0],
                  [255, 0, 255], [0, 0, 255]]

    cls_sel = [1, 2, 3, 4, 6, 8]
    cls_thresh = [1,0.6,0.5,0.8,0.5,0.9,0.7,0.9,0.5]
    if count == 0:
        logger.info(
            ' \ Note: inference on the first image will be slower than the '
            'rest (caches and auto-tuning need to warm up)'
        )
    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(cv2.__version__)
    fourcc = cv2.VideoWriter_fourcc(b'X', b'V', b'I', b'D')
    videoWriter = cv2.VideoWriter('2.avi', fourcc, 10, size)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    while(1):
        # get a frame
        ret, im = cap.read()
        count = count + 1
        # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        timers = defaultdict(Timer)
        # detect one image
        with c2_utils.NamedCudaScope(0):
            cls_boxes, _, _ = infer_engine.im_detect_all(
                model, im, None, timers=timers)
        # logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        # for k, v in timers.items():1440
        #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        # cls_boxes_sel=cls_boxes[[cls_id for cls_ind, cls_id in enumerate(cls_sel[0:])]]
        demo_vis_one_imageboxes_opencv(im, cls_boxes, thresh=cls_thresh, show_box=True, dataset=dummy_coco_dataset,
                                       show_class=True, class_names=class_names, color_list=color_list, cls_sel=cls_sel)


        avg_fps = count / (time.time() - start_time)
        cv2.putText(im, '{:s} {:.3f}/s'.format('fps', avg_fps), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                    lineType=cv2.LINE_AA)
        # show a frame
    return im





if __name__ == '__main__':
    # workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    # utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)

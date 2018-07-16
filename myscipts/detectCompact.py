#!/usr/bin/env python2

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
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
sys.path.append('/home/long/github/caffe2/build')
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

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='/home/long/github/detectron/configs/getting_started/e2e_lighthead_rcnn_R-50-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/home/long/github/detectron/detectron-output/lighthead/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='demo/coco/out',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default='/home/long/Pictures/coco'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    while (1):
        frame = {}
        ret, im = capture.read()
        # if type(cam)==str:
        #    im=cv2.resize(im, None, None, fx= width/size[0], fy= height/size[1], interpolation=cv2.INTER_LINEAR)
        ori_im = copy.deepcopy(im)
        frame["img"] = im
        # _, contours, hierarchy = cv2.findContours(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, [fisherROI], -1, (0, 255, 0), 4)
        # # detect one image
        if running == False:
            if type(cam) == str:
                time.sleep(1 / fps)
            count = 0
            queue.put(frame)
            # count = count+1
            # if count%1000:
            #     queue.put(frame)
        else:
            count = count + 1
            # if count == 2:
            #     start_time = time.time()
            # count = count + 1
            if count % 8 == 1:
                st = time.time()
                with c2_utils.NamedCudaScope(0):
                    cls_boxes, _, _ = infer_engine.im_detect_all(
                        model, im, None, timers=None)
                print('one image detection without visulization cost %f fps' % (1 / (time.time() - st)))
                demo_vis_one_imageboxes_opencv(im, cls_boxes, thresh=cls_thresh, show_box=True, show_class=True,
                                               class_names=class_names, color_list=color_list, cls_sel=cls_sel,
                                               frame=frame)
                # if count2 >= 1:
                #     et = et+time.time()-st
                #     avg_fps = (count2) / et
                #     cv2.putText(frame["img"], '{:s} {:.2f}/s'.format('fps', avg_fps), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255),
                #                 lineType=cv2.LINE_AA)
                # count2 = count2 + 1
                # with c2_utils.NamedCudaScope(0):
                #     cls_boxes, _, _ = infer_engine.im_detect_all(
                #         model, im, None, timers=None)
                # demo_vis_one_imageboxes_opencv(im, cls_boxes, thresh=cls_thresh, show_box=True, show_class=True,
                #                                class_names=class_names, color_list=color_list, cls_sel=cls_sel, frame=frame)
                # if count>=2:
                #     avg_fps = (count-1) / (time.time() - start_time)
                #     cv2.putText(frame["img"], '{:s} {:.2f}/s'.format('fps', avg_fps), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255),
                #             lineType=cv2.LINE_AA)
                img = cv2.resize(frame["img"], (960, 540))
                videoWriter.write(img)  # write frame to video
                queue.put(frame)

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
        t1 = time.time()
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
        t2 = time.time()-t1
        print("vis time %f.2ms"%(t2*1000))

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)

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


from caffe2.python import core,workspace
from caffe2.proto import caffe2_pb2

from caffe2.python import net_drawer

import cv2


def get_model(cfg_file, weights_file):
    merge_cfg_from_file(cfg_file)
    cfg.TRAIN.WEIGHTS = '' # NOTE: do not download pretrained model weights
    cfg.TEST.WEIGHTS = weights_file
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    return model
DETECTRON_ROOT ='/home/long/github/detectron'
cfg_file = '{}/configs/getting_started/res18_1mlp_fpn64.yaml'.format(DETECTRON_ROOT)
weights_file = '/media/E/models/detectron/ImageNetPretrained/R-18.pkl'
model = get_model(cfg_file, weights_file)

from caffe2.python import net_drawer
g = net_drawer.GetPydotGraph(model, rankdir="TB")
g.write_dot(model.Proto().name + '.dot')


g.write_png(model.Proto().name+".png")
img1 = cv2.imread(model.Proto().name+".png",1)

cv2.imshow("Netgraph",img1)

cv2.waitKey(0)

#command line
# png: dot graph1.gv -Tpng -o test.png
#
# pdf: dot graph1.gv -Tpdf -o test.pdf
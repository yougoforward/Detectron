#!/usr/bin/python2
# -*- coding: utf-8 -*-
# # We'll also import a few standard python libraries
# from matplotlib import pyplot
# import numpy as np
# import time
#
# # These are the droids you are looking for.
# from caffe2.python import core, workspace
# from caffe2.proto import caffe2_pb2
#
# X = np.random.randn(2, 3).astype(np.float32)
# Y = core.FloatToHalf(X,'Y')
# print("Generated X from numpy:\n{}".format(X))
# workspace.FeedBlob("X", X)
#
#
# print("Current blobs in the workspace: {}".format(workspace.Blobs()))
# print("Workspace has blob 'X'? {}".format(workspace.HasBlob("X")))
# print("Fetched X:\n{}".format(workspace.FetchBlob("X")))



# import sys
# sys.path.insert(0, '/path/to/caffe2/build')
# from caffe2.python import core, workspace, model_helper
# from caffe2.proto import caffe2_pb2, caffe2_legacy_pb2
#
# # -------- CPU/GPU 模式切换 -----
# workspace.ResetWorkspace()
# # device_opts = core.DeviceOption(caffe2_pb2.CPU, 0) # CPU 模式
# device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0) # GPU 模式，及 gpuid
#
#
# # -------- 加载模型和参数 ------------
# INIT_NET = 'init_net.pb'
# PREDICT_NET = 'predict_net.pb'
#
# init_def = caffe2_pb2.NetDef()
# with open(INIT_NET, 'rb') as f:
#     init_def.ParseFromString(f.read())
#     init_def.device_option.CopyFrom(device_opts)
#     workspace.RunNetOnce(init_def.SerializeToString())
#
# net_def = caffe2_pb2.NetDef()
# with open(PREDICT_NET, 'rb') as f:
#     net_def.ParseFromString(f.read())
#     net_def.device_option.CopyFrom(device_opts)
#     workspace.CreateNet(net_def.SerializeToString())
#
# name = net_def.name
# output_name = net_def.external_output[-1] # 输出 blob 名
# input_name = net_def.external_input[0] # 输入 blob 名
#
# # -------- 送入数据 blob -----------
# input_data = np.random.rand(2, 3, 227, 227).astype(np.float32) # NCHW
# workspace.FeedBlob(input_name, input_data, device_opts) # device_opts：CPU/GPU 模式的选项
#
# # -------- Forward ----------------
# workspace.RunNet(name, 1)
#
# # --------- 读取网络计算结果 --------
# results = workspace.FetchBlob(output_name)
















from caffe2.proto import caffe2_pb2, caffe2_legacy_pb2
from caffe2.python import core, workspace, model_helper
import numpy as np
# Create random tensor of three dimensions
# x = np.random.rand(4, 3, 2)
# print(x)
# print(x.shape)
#
# workspace.FeedBlob("my_x", x)
#
# x2 = workspace.FetchBlob("my_x")
# print(x2)
workspace.ResetWorkspace()
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)

# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data, device_opts)
workspace.FeedBlob("label", label, device_opts)

# Create model using a model helper
m = model_helper.ModelHelper(name="my first net",arg_scope={
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustive_search': True,
        'ws_nbytes_limit': (1024 * 1024 * 1024),
    })

weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])
fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = m.net.Sigmoid(fc_1, "pred")
pred2 = m.net.FloatToHalf(pred, 'pred2')
pred = m.net.HalfToFloat(pred2, 'pred3')
softmax, loss = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])
# softmax2 = m.net.FloatToHalf(softmax, 'softmax2')
print(m.net.Proto())
print(m.param_init_net.Proto())
m.net.RunAllOnGPU(gpu_id=0, use_cudnn=True)
m.param_init_net.RunAllOnGPU(gpu_id=0, use_cudnn=True)
workspace.RunNetOnce(m.param_init_net)
workspace.CreateNet(m.net)
# Run 100 x 10 iterations
for _ in range(100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)

    workspace.FeedBlob("data", data, device_opts)
    workspace.FeedBlob("label", label, device_opts)

    workspace.RunNet(m.name, 10)   # run for 10 times
    pred2 = workspace.FetchBlob('pred2')
    print(pred2)
    print(pred2.dtype)
    # softmax2 = workspace.FetchBlob('softmax2')
    # print(softmax2)
    # print(softmax2.dtype)

print(workspace.FetchBlob("softmax"))
print(workspace.FetchBlob("loss"))


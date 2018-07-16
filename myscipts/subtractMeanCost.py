import numpy as np
import time


for i in range(10):
    st =time.time()
    a = np.random.rand(512,928,3)
    b = np.random.rand(1,1,3)
    # a=a.astype(np.float16)
    # b=b.astype(np.float16)
    st1 =time.time()
    # b = np.tile(b,(720,1280,1))
    c=a-b
    et1= st1-st
    et= time.time()-st1
    print("%f.2 ms"%(et1*1000))
    print("%f.2 ms"%(et*1000))

channels = 3
height =360
width =640
stt =time.time()
for c in range(channels):
    for h in range(height):
        for w in range(width):
            a[h,w,c]=a[h,w,c]-b[0,0,c]
ett =time.time()-stt
print("for cost time %f.2 ms"%(ett*1000))



#
# import cv2
# cv2.CAP_GSTREAMER
# cap = cv2.VideoCapture()

from caffe2.python import core, workspace
from caffe2.python import workspace, model_helper
import numpy as np
# Create random tensor of three dimensions
x = np.random.rand(4, 3, 2)
print(x)
print(x.shape)

workspace.FeedBlob("my_x", x)

x2 = workspace.FetchBlob("my_x")
print(x2)

# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
m = model_helper.ModelHelper(name="my first net")
weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])

fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = m.net.Sigmoid(fc_1, "pred")
softmax, loss = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

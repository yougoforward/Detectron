import cPickle as pickle
import numpy as np
import mxnet as mx
fr = open('/media/E/models/detectron/res18_1mlp_fpn64_320.pkl')
# fr=open('/media/E/models/detectron/res18_1mlp_fpn64_512.pkl')
# fr=open('/home/long/github/detectron/detectron-output/res18_1mlp_fpn64_512/train/fisher_train_221:fisher_val_221/generalized_rcnn/model_final.pkl')
# fr =open('/media/E/models/resnet/resnet18caffe2.pkl')
#fr =open('/home/long/github/detectron/detectron-output/fish_1mlp_fpn64_512/train/fisher_train_221:fisher_val_221/generalized_rcnn/model_final.pkl')
#fr =open('/home/long/github/detectron/detectron-output/fish_1mlp_fpn128_512/train/fisher_train_221:fisher_val_221/generalized_rcnn/model_final.pkl')
# fr = open('/media/E/models/detectron/compactfishfasterfpn50.pkl')
# fr = open('/home/long/github/detectron/detectron-output/fisherall/train/fisher_train:fisher_val/generalized_rcnn/model_final.pkl')
# fr = open('/home/long/github/detectron/detectron-output/lighthead/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl')
# fr = open('/media/E/models/detectron/e2e_faster_rcnn_R-50-FPN_2x.pkl')
# fr = open('/home/long/github/detectron/models/detectron/ImageNetPretrained/R-50.pkl')
# fr = open('/home/long/github/MobileNet-Caffe/mobilenet.caffemodel')
# fr = open('/home/long/github/MobileNet-Caffe/mobilenet-0000.params')
#
inf = pickle.load(fr)
lines = fr.readlines()
fr.close()

blobs = inf['blobs']
# blobs=inf

max=0
min=0
for k,v in blobs.items():
    if np.max(v)>max:
        max =np.max(v)
    if np.min(v)<min:
        min=np.min(v)
nomomentums = {i:v for i,v in blobs.items() if 'momentum' not in i}
compactdict = {}

for k,v in nomomentums.items():
    name_split = k.split('_')
#    if 'fc1000' in k:
#        continue

    if name_split[-1] == 'b':
        if 'res' in k and 'branch' in k:
            if 'bn' not in k:
                continue
    if 'bn' in k:
        bnIndex= name_split.index('bn')
        nm = '_'.join(name_split[:bnIndex+1])
        # # name = nm+'_b'
        # s = nomomentums[nm+'_s']
        # b = nomomentums[nm+'_b']
        # rm = nomomentums[nm+'_rm']
        # riv = nomomentums[nm + '_riv']
        #
        # scale = s/np.sqrt(np.abs(riv))
        # bias = b-rm*s/np.sqrt(np.abs(riv))
        # compactdict[nm+"_s"]=scale
        # compactdict[nm+"_b"]=bias
        compactdict[k] = v.astype(np.float32)
        compactdict['_'.join(name_split[:-1])+'_rm'] = np.zeros(v.shape,dtype=np.float32)
        compactdict['_'.join(name_split[:-1])+'_riv'] = np.ones(v.shape,dtype=np.float32)
    # elif 'rpn' in k:
    #     compactdict[k] = v.astype(np.float16)
    # elif 'fc' in k or 'cls_score' in k or 'bbox_pred' in k:
    #     compactdict[k] = v
    else:
        compactdict[k]=v.astype(np.float16)
#          compactdict[k] = v.astype(np.float32)
    # compactdict[k] = v
compactmodel = open('/media/E/models/detectron/res18_1mlp_fpn64_320BN.pkl','wb')
# names = []
# for k,v in compactdict.items():
#     names.append(k)
# with open('resnetfpnnames.txt','w') as f:
#     f.write('\n'.join(names))
#     f.close()
# compactmodel = open('/media/E/models/detectron/compact1mlpfpn128BNfp16full.pkl','wb')
# compactmodel = open('/media/E/models/detectron/compactfishfasterfpn50BNfp16full.pkl','wb')
pickle.dump(compactdict,compactmodel,pickle.HIGHEST_PROTOCOL)
compactmodel.close()



# frcompact = open('/media/E/models/detectron/compactfishfasterfpn50.pkl')
# infcompact = pickle.load(frcompact)
# frcompact.close()

# momentums = {i:v for i,v in blobs.items() if 'momentum' in i}
# fpns = {i:v for i,v in blobs.items() if 'fpn' in i}
# scores = {i:v for i,v in blobs.items() if 'score' in i}
# fcs = {i:v for i,v in blobs.items() if 'fc' in i}
# bns = {i:v for i,v in blobs.items() if 'bn' in i}
# ws = {i:v for i,v in blobs.items() if '_w' in i}
# bs = {i:v for i,v in blobs.items() if '_b' in i}

"""
convert pytorch to  caffe2
"""












#####
"""
convert mobilenet from caffe, first to mxnet
"""
save_dict = mx.nd.load('/home/long/github/MobileNet-Caffe/mobilenet-0000.params')

# print(save_dict)

caffe2dict = {}
for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        # if tp == 'arg':
        #     arg_params[name] = v
        # if tp == 'aux':
        #     aux_params[name] = v
        name_split = name.split('_')
        print(name)
        print(name_split)
        # if 'beta' in name_split:
        #     name_split[-1] = 'b'
        #     name = '_'.join(name_split)
        # if 'gamma' in name_split:
        #     name_split[-1] = 's'
        #     name = '_'.join(name_split)
        if 'fc7' in name_split:
            continue
        # convert bn to AffineChannel
        if 'bn' in name_split:
            bnIndex= name_split.index('bn')
            nm = '_'.join(name_split[:bnIndex+1])
            # name = nm+'_b'
            gamma = save_dict['arg:'+nm+'_gamma'].asnumpy()
            beta = save_dict['arg:'+nm+'_beta'].asnumpy()
            mean = save_dict['aux:'+nm+'_moving_mean'].asnumpy()
            var = save_dict['aux:'+nm + '_moving_var'].asnumpy()

            scale = gamma/np.sqrt(np.abs(var))
            bias = beta-mean*gamma/np.sqrt(np.abs(var))

            caffe2dict[nm+'_b'] = bias
            caffe2dict[nm+'_s'] = scale


        if 'weight' in name_split:
            name_split[-1] = 'w'
            name = '_'.join(name_split)
            print(name)
            caffe2dict[name] = v.asnumpy()





print(caffe2dict)
fpkl = open('/media/E/models/detectron/ImageNetPretrained/mobilenet.pkl','wb')
# fpkl = open('/home/long/github/MobileNet-Caffe/mobilenet.pkl','w')
pickle.dump(caffe2dict,fpkl,pickle.HIGHEST_PROTOCOL)
fpkl.close()
fpkl = open('/home/long/github/MobileNet-Caffe/mobilenet.pkl','r')
# fpkl = open('/home/long/github/detectron/models/detectron/ImageNetPretrained/mobilenet.pkl','r')
mobilenet = pickle.load(fpkl)

print(mobilenet)
# inf = pickle.load(open('mnist.pkl'))

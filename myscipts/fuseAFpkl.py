import cPickle as pickle
import numpy as np
pkl = open('/media/E/models/detectron/res18_1mlp_fpn64_512.pkl')

infr = pickle.load(pkl)
model_dict = infr['blobs']
new_model_dict = {}
for k, v in model_dict.items():
    if 'bn' not in k:
        if 'momentum' not in k:
            if 'fc1000' not in k:
                new_model_dict[k] = model_dict[k]
for k, v in model_dict.items():
    if 'bn' in k:
        if 'res_conv1_bn' in k:
            if '_s' in k:
                continue
            A = model_dict['res_conv1_bn_s'].reshape(-1, 1, 1, 1)
            new_model_dict['conv1_w'] = model_dict['conv1_w'] * A
            new_model_dict['conv1_b'] = model_dict['res_conv1_bn_b']

        else:
            if '_s' in k:
                continue
            split_name = k.split('_')
            baseBNconv = '_'.join(split_name[:-2])
            baseBN = '_'.join(split_name[:-1])
            A = model_dict[baseBN + '_s'].reshape(-1, 1, 1, 1)
            new_model_dict[baseBNconv + '_w'] = model_dict[baseBNconv + '_w'] * A
            new_model_dict[baseBNconv + '_b'] = model_dict[baseBN + '_b']

for k,v in new_model_dict.items():
    new_model_dict[k]=v.astype(np.float16)

fuse_model_file=open('/media/E/models/detectron/res18_1mlp_fpn64_512_fuseBN.pkl','wb')
pickle.dump(new_model_dict, fuse_model_file, pickle.HIGHEST_PROTOCOL)
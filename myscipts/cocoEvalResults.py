# from datasets.json_dataset import JsonDataset
# json_dataset = JsonDataset('coco_2014_minival')
# import datasets.json_dataset_evaluator as json_dataset_evaluator
# res_file = '/home/long/github/detectron/test/fisher_valtt_221/res18_1mlp_fpn64_512_fp32/bbox_fisher_valtt_221_results.json'
# output_dir = '/home/long/github/detectron/detectron-output'
# coco_eval = json_dataset_evaluator._do_detection_eval(json_dataset, res_file, output_dir)


import  cPickle as pickle
import numpy as np
f=open('/home/long/github/detectron/test/coco_2014_minival/generalized_rcnn/detection_results.pkl')
coco_results=pickle.load(f)

coco_person_eval=coco_results.eval['precision'][:,:,0,:,2]
coco_person_stats = np.mean(coco_person_eval,axis=1)
coco_person_stats_0595 = np.mean(coco_person_stats,axis=0)
f.close()
from datasets.json_dataset import JsonDataset
test_DATASET = '/media/E/fisher_train/all/annotations/test.json'
json_dataset = JsonDataset('qiyan_train_coco3')
# json_dataset = JsonDataset('fisher_val')
output_dir = '/home/long/github/detectron/detectron-output'
import datasets.json_dataset_evaluator as json_dataset_evaluator
# res_file ='res18_1mlp_fpn64_512_fp32'
res_file=''
# res_file = '/home/long/github/detectron/detectron-output/qiyan_finetune/test/qiyan_train_coco3/generalized_rcnn/bbox_qiyan_train_coco3_results.json'
#res_file ='/home/long/github/detectron/detectron-output/fisherall/test/fisher_val/generalized_rcnn/bbox_fisher_val_results.json'
#res_file = '/home/long/github/detectron/detectron-output/fish221800/test/fisher_valtt_221/generalized_rcnn/bbox_fisher_valtt_221_results.json'
# res_file ='/home/long/github/detectron/test/fisher_valtt_221/generalized_rcnn/bbox_fisher_valtt_221_results.json'
coco_eval = json_dataset_evaluator._do_detection_eval(json_dataset, res_file, output_dir)



# f = open("precisionresultfisher22180050.txt","w")
# for i in range(101):
#     f.write(str(coco_eval.eval['precision'][0,i,0,:,2])+'\n')
# f.close()
# f = open("precisionresultfisher22180060.txt","w")
# for i in range(101):
#     f.write(str(coco_eval.eval['precision'][2,i,0,:,2])+'\n')
# f.close()
# f = open("precisionresultfisher22180065.txt","w")
# for i in range(101):
#     f.write(str(coco_eval.eval['precision'][3,i,0,:,2])+'\n')
# f.close()
# f = open("precisionresultfisher22180070.txt","w")
# for i in range(101):
#     f.write(str(coco_eval.eval['precision'][4,i,0,:,2])+'\n')
# f.close()
# #
#
# json_dataset = JsonDataset('fisher_train_221')
# json_dataset = JsonDataset('fisher_val_221')
# json_dataset = JsonDataset('fisher_valtt_221')
# len_anns=len(json_dataset.COCO.anns)
# print(len_anns)
# area_list=[json_dataset.COCO.anns[i]['area'] for i in range(0,len(json_dataset.COCO.anns))]
# # print(area_list)
# area_list.sort()
# # print(area_list)
# print(len([i for i in area_list if i<=1600]))
# print(len([i for i in area_list if i<=2500]))
# print(len([i for i in area_list if i>2500 and i<=5000]))
# print(len([i for i in area_list if i>5000 and i<=250000]))
# print(len([i for i in area_list if i>250000]))
#




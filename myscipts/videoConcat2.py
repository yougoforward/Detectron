#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division, print_function

import os.path as path
import six
import cv2 as ocv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':

    # ROOT_PATH = '/media/jnie/Seagate Backup Plus Drive/Projects/20180504FishEyeCarDet'
    # VIDEO_NAME = '01010002855000000'
    # ori_vid = Video(path.join(ROOT_PATH, 'their_video', VIDEO_NAME) + '.avi')
    # cur_vid = Video(path.join(ROOT_PATH, 'our_video', VIDEO_NAME) + '.avi')

    ori_vid = ocv.VideoCapture('/media/E/广角汽车检测项目/video/01010002855000000.mp4_out.avi')
    cur_vid = ocv.VideoCapture('/media/E/广角汽车检测项目/video/vpa_01010002855000000_out_select.avi')
    fourcc = ocv.VideoWriter_fourcc(b'U', b'2', b'6', b'3')
    fourcc = 1196444237.0
    videoWriter = ocv.VideoWriter('ObjectDetection.avi', int(fourcc), 3.0, (int(1920), int(540)))
    # init_windows()

    while True:
        try:
            ret1,ori_frame = ori_vid.read()
            ret2,cur_frame = cur_vid.read()
            ocv.putText(ori_frame, 'Company', (40, 40), ocv.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), thickness=2, lineType=ocv.LINE_AA)
            ocv.putText(cur_frame, 'VI Lab', (40, 40), ocv.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), thickness=2, lineType=ocv.LINE_AA)
            print(ori_frame.shape)
            print(cur_frame.shape)
        except:
            continue

        # except StopIteration:
            # break

        # ocv.imshow(WINDOW_NAME['ori_vid'], ori_frame)
        # ocv.imshow(WINDOW_NAME['cur_vid'], cur_frame)
        # alpha = 0.5
        # blend_frame = ocv.addWeighted(ori_frame, alpha, cur_frame, 1. - alpha, 0.)
        # ocv.imshow(WINDOW_NAME['blend_vid'], blend_frame)
        # diff_frame = ori_frame - cur_frame
        # ocv.imshow(WINDOW_NAME['diff_vid'], diff_frame)



        combine=ocv.hconcat([ori_frame, cur_frame])
        videoWriter.write(combine)  # write frame to video
        ocv.imshow('videoConcat', combine)
        key = ocv.waitKey(1)
        if key < 0:
            continue
        else:
            if key == 27:  # ESC
                break
            if key == ord('p'):
                key = ocv.waitKey(0)
                continue

    ocv.destroyAllWindows()
    ori_vid.release()
    cur_vid.release()
    videoWriter.release()
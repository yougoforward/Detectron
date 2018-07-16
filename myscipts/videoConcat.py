#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division, print_function

import os.path as path
import six
import cv2 as ocv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
class Video(object):
    _vid_cap = None
    _identifier = None

    def __init__(self, identifier=0):
        """

        :type identifier: int|str
        """
        if isinstance(identifier, str):
            if not path.exists(identifier):
                raise IOError('Video file not found!')
        self._identifier = identifier
        self._vid_cap = ocv.VideoCapture(identifier)

    @property
    def video_capture(self):
        return self._vid_cap

    @property
    def frames(self):
        cap = self._vid_cap
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            raise StopIteration

    def __getattr__(self, name):
        """
        POS_MSEC: Current position of the video file in milliseconds or video capture timestamp.
        POS_FRAMES: 0-based index of the frame to be decoded/captured next.
        POS_AVI_RATIO: Relative position of the video file: 0 - start of the film, 1 - end of the film.
        FRAME_WIDTH: Width of the frames in the video stream.
        FRAME_HEIGHT: Height of the frames in the video stream.
        FPS: Frame rate.
        FOURCC: 4-character code of codec.
        FRAME_COUNT: Number of frames in the video file.
        FORMAT: Format of the Mat objects returned by retrieve() .
        MODE: Backend-specific value indicating the current capture mode.
        BRIGHTNESS: Brightness of the image (only for cameras).
        CONTRAST: Contrast of the image (only for cameras).
        SATURATION: Saturation of the image (only for cameras).
        HUE: Hue of the image (only for cameras).
        GAIN: Gain of the image (only for cameras).
        EXPOSURE: Exposure (only for cameras).
        CONVERT_RGB: Boolean flags indicating whether images should be converted to RGB.
        WHITE_BALANCE: Currently not supported
        RECTIFICATION: Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
        """
        return self._vid_cap.get(getattr(ocv, 'CAP_PROP_' + name.upper()))

    def __str__(self):
        return self._identifier


WINDOW_NAME = {
    'ori_vid': 'Original video',
    'cur_vid': 'Current video',
    'blend_vid': 'Blended video',
    'diff_vid': 'Diff video',
}


def init_windows():
    for _, name in six.iteritems(WINDOW_NAME):
        ocv.namedWindow(name, flags=ocv.WINDOW_KEEPRATIO)


if __name__ == '__main__':

    # ROOT_PATH = '/media/jnie/Seagate Backup Plus Drive/Projects/20180504FishEyeCarDet'
    # VIDEO_NAME = '01010002855000000'
    # ori_vid = Video(path.join(ROOT_PATH, 'their_video', VIDEO_NAME) + '.avi')
    # cur_vid = Video(path.join(ROOT_PATH, 'our_video', VIDEO_NAME) + '.avi')
    ori_vid = Video('/media/E/广角汽车检测项目/video/01010002855000000.mp4_out.avi')
    cur_vid = Video('/media/E/广角汽车检测项目/video/vpa_01010002855000000_out_select.avi')
    fourcc = ocv.VideoWriter_fourcc(b'U', b'2', b'6', b'3')
    fourcc = 1196444237.0
    videoWriter = ocv.VideoWriter('ObjectDetection.avi', int(fourcc), 3.0, (int(1920), int(540)))
    # init_windows()

    while True:
        try:
            ori_frame = next(ori_vid.frames)
            cur_frame = next(cur_vid.frames)
            ocv.putText(ori_frame, 'Company', (40, 40), ocv.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), thickness=2, lineType=ocv.LINE_AA)
            ocv.putText(cur_frame, 'VI Lab', (40, 40), ocv.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), thickness=2, lineType=ocv.LINE_AA)
            print(ori_frame.shape)
            print(cur_frame.shape)
        except StopIteration:
            break
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
    videoWriter.release()
#!/usr/bin/env python
import cv2
import numpy as np
import rospy


class DepthProcessor:

    def __init__(self, debug_stream=None):
        self.MAX_UINT8 = 255
        self.debug_stream = debug_stream

        # Initialize debug stream
        if self.debug_stream:
            self.debug_stream.create_stream('depth_processing', 'debug/depth_processing')

    def ground(self, gray_img, x, y, n, T1, T2):
        if gray_img[y][x] > T1 or gray_img[y][x] < T2:
            return 0
        if int(gray_img[y][x]) - int(gray_img[y + n][x]) >= 1:
            for i in range(n):
                if int(gray_img[y + i][x]) - int(gray_img[y + i + 1][x]) < 0:
                    return gray_img[y][x]
            return 0
        else:
            return gray_img[y][x]

    def resize_np(self, img_np, percent):
        h, w = img_np.shape
        w = int(w * percent)
        h = int(h * percent)
        resized_img = cv2.resize(img_np, (w, h))
        return resized_img

    def pre_processing_depth_img(self, img_np, n=2, T1=200, T2=10, min_width=10, min_height=10):

        # Resize
        gray_img = self.resize_np(img_np, 0.125)
        # cv2.imshow('src', gray_img)
        # cv2.waitKey(1)
	
        # CLOSE
        kernel_close = np.ones((3, 3))
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel_close)

        # DILATE
        kernel_dilate = np.ones((3, 3))
        gray_img = cv2.dilate(gray_img, kernel_dilate)
        # print(np.max(gray_img))
        # cv2.imshow('depth', gray_img)
        # cv2.waitKey()

        height, width = gray_img.shape
        # print(height, width)

        # remove floor and wall far away...
        gray_img[gray_img > T1] = 0
        gray_img[gray_img < T2] = 0
        gray_img[height-n:, :] = 0

        for x in range(width):
            for y in range(height - n):
                gray_img[y][x] = self.ground(gray_img, x, y, n, T1, T2)

        # cv2.imshow('after remove floor', gray_img)

        # CLOSE
        kernel_close = np.ones((3, 3), np.uint8)
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel_close)
        # cv2.imshow('removed_ground_CLOSE', gray_img)

        # resize
        gray_img = self.resize_np(gray_img, 8)
        # cv2.imshow('preprocessed', gray_img)

        h, w = gray_img.shape[:2]
        gray_img[0:h//4, :] = 0
        gray_img[gray_img > 0] = 255

        if self.debug_stream:
            self.debug_stream.update_image('depth_processing', gray_img)

        return gray_img
      
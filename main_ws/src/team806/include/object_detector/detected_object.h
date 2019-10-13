#ifndef DETECTED_OBJECT_H
#define DETECTED_OBJECT_H

#include <ros/package.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class DetectedObject {

   public:
    enum ObjectLabel { OBJECT_1 = 1, OBJECT_2 = 2, OBJECT_3 = 3, TURN_LEFT_SIGN = 4, TURN_RIGHT_SIGN = 5};

    ObjectLabel label;
    cv::Rect rect;
    double weight;

    unsigned int hit_history = 1; // Hit/miss history in binary

    DetectedObject(ObjectLabel label, cv::Rect rect);
    DetectedObject(ObjectLabel label, cv::Rect rect, double weight);
};

#endif
#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <dirent.h>
#include <limits.h>
#include <ros/package.h>
#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include "config.h"
#include "image_publisher.h"


#include "detected_object.h"


class ObjectDetector : ImagePublisher {
   public:

    // Detect the objects
    // Return the number of objects in the input image
    virtual int detect(const cv::Mat &img,
           std::vector<DetectedObject> &detected_objects, bool debug) = 0;


};

#endif
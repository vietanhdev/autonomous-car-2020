#ifndef HOG_BASED_OBJECT_DETECTOR_H
#define HOG_BASED_OBJECT_DETECTOR_H

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
#include "object_detector.h"


class HogBasedObjectDetector : public ObjectDetector {
    public:

    DetectedObject::ObjectLabel label;
    cv::HOGDescriptor hog;
    cv::Size winstride;
    double threshold;


    HogBasedObjectDetector(DetectedObject::ObjectLabel label, cv::HOGDescriptor hog, cv::Size winstride = cv::Size(4,4));
    HogBasedObjectDetector(DetectedObject::ObjectLabel label, const std::string & hog_file, cv::Size winstride = cv::Size(4,4));
    HogBasedObjectDetector(DetectedObject::ObjectLabel label, const std::string & hog_file, double threshold, cv::Size winstride = cv::Size(4,4));


    // Detect the objects
    // Return the number of objects in the input image
    int detect(const cv::Mat &img,
           std::vector<DetectedObject> &detected_objects, bool debug = false);


};

#endif
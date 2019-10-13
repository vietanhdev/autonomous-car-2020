#ifndef TEMPL_MATCHING_OBJECT_DETECTOR_H
#define TEMPL_MATCHING_OBJECT_DETECTOR_H

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


class TemplMatchingObjectDetector : public ObjectDetector {
    public:

    DetectedObject::ObjectLabel label;
    double threshold;
    std::shared_ptr<Config> config;
    std::vector<cv::Mat> object_templs;

    TemplMatchingObjectDetector(DetectedObject::ObjectLabel label, std::string templ_folder_name, double threshold);


    // Detect the objects
    // Return the number of objects in the input image
    int detect(const cv::Mat &img,
           std::vector<DetectedObject> &detected_objects, bool debug = false);

    // Matching object using template matching
    bool matching(const cv::Mat &img, const cv::Mat &templ,
                  std::vector<cv::Rect> &rects, bool debug = false);

    // Get file extension from filepath/filename
    static std::string getFileExt(const std::string &s);


};

#endif
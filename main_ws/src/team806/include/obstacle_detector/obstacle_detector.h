#ifndef OBSTACLE_DETECTOR_H
#define OBSTACLE_DETECTOR_H

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


class ObstacleDetector {

    public:

    // TODO: LEGACY way to detect obstacle. Remove in the future
    static void printLineDiff(const std::vector<cv::Point> &line) {
        for (int i = line.size() - 2; i >= 0; --i) {
            std::cout << (int)abs(line[i].x - line[i + 1].x) << ",";
        }
        std::cout << std::endl;
    }

};

#endif
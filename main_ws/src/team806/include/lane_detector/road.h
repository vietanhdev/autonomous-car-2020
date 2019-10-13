#ifndef LANE_H
#define LANE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Road {

    public:

    static const int road_width = 90;
    static const int road_center_line_x = 160;

    int lane_area;

    cv::Mat lane_mask;

    std::vector<cv::Point> left_points;
    std::vector<cv::Point> right_points;
    std::vector<cv::Point> middle_points;

    

};

#endif
#ifndef COLOR_RANGE_H
#define COLOR_RANGE_H

#include <opencv2/opencv.hpp>

typedef struct ColorRangeStruct {
    cv::Scalar begin;
    cv::Scalar end;
    ColorRangeStruct(double h1, double s1, double v1,
    double h2, double s2, double v2) : begin(cv::Scalar(h1, s1, v1)), end(cv::Scalar(h2, s2, v2)) {}
} ColorRange;


#endif
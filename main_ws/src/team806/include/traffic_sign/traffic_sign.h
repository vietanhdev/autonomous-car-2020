#ifndef TRAFFIC_SIGN_H
#define TRAFFIC_SIGN_H

#include <opencv2/opencv.hpp>
#include "timer.h"


class TrafficSign {
    
    public:
        enum SignType {
            NO_SIGN = 0,
            TURN_LEFT = 1,
            TURN_RIGHT = 2
        };
        int id;
        cv::Rect rect;
        Timer::time_point_t observe_time;

        TrafficSign(int id, cv::Rect rect);
};

#endif
#include "traffic_sign.h"

TrafficSign::TrafficSign(int id, cv::Rect rect) {
    this->id = id;
    this->rect = rect;
    this->observe_time = Timer::getCurrentTime();
}
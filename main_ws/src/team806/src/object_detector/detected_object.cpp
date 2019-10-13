#include "object_detector.h"


DetectedObject::DetectedObject(ObjectLabel label, cv::Rect rect) {
    this->label = label;
    this->rect = rect;
}

DetectedObject::DetectedObject(ObjectLabel label, cv::Rect rect, double weight) {
    this->label = label;
    this->rect = rect;
    this->weight = weight;
}
#ifndef OBJECT_DETECTOR_MANAGER_H
#define OBJECT_DETECTOR_MANAGER_H

#include <bitset>
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
#include "object_detector.h"
#include "hog_based_object_detector.h"
#include "templ_matching_object_detector.h"

class ObjectDetectorManager : ImagePublisher {
   public:
    bool debug_flag = true;
    image_transport::Publisher debug_img_publisher;
    std::shared_ptr<Config> config;

    std::vector <ObjectDetector*> detectors;
    size_t detector_iter = 0;
    size_t num_of_detectors_each_global_search = 3;
    size_t num_of_frames_bw_global_searchs = 0;
    size_t num_of_frames_to_global_search = 0;

    std::vector <DetectedObject> detected_objects;

    ObjectDetectorManager();


    size_t getDetectorIter() {
        return detector_iter;
    }

    size_t increaseDetectorIter() {
        ++detector_iter;
        if (detector_iter >= detectors.size()) { // Reset after a round
            detector_iter = 0;
        }
    }

     // Detect the obstacles
     // Return the number of obstacles in the input image
     int detect(const cv::Mat &img,
            std::vector<DetectedObject> &detected_objects);

    // Filter new detected object.
    // Merge the result if new object is the same one as a detected one
    // Or update the results
    void filterNewDetectedObjects(std::vector<DetectedObject> new_detected_objects, std::vector<DetectedObject> &output_list);

    // Count bit 1 from the last position of an unsigned number
    // input: n: number; num_of_last_bit_to_count: number of bits to count from the last bit
    int countNonZeroBits(unsigned int n, unsigned int num_of_last_bit_to_count);

};

#endif
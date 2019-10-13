#include "object_detector_manager.h"

ObjectDetectorManager::ObjectDetectorManager() {

    config = Config::getDefaultConfigInstance();
    debug_flag = config->get<bool>("debug_object_detector");

    // // Init detectors
    // detectors.push_back(
    //     dynamic_cast<ObjectDetector *>(new HogBasedObjectDetector(
    //         DetectedObject::ObjectLabel::OBJECT_1,
    //         ros::package::getPath(config->getROSPackage()) +
    //             "/data/object_hog_files/object1_v3.yml",
    //         0.5)));
    // detectors.push_back(
    //     dynamic_cast<ObjectDetector *>(new HogBasedObjectDetector(
    //         DetectedObject::ObjectLabel::OBJECT_2,
    //         ros::package::getPath(config->getROSPackage()) +
    //             "/data/object_hog_files/object2_v2.yml",
    //         0.5)));
    // detectors.push_back(
    //     dynamic_cast<ObjectDetector *>(new HogBasedObjectDetector(
    //         DetectedObject::ObjectLabel::OBJECT_3,
    //         ros::package::getPath(config->getROSPackage()) +
    //             "/data/object_hog_files/object3_v1.yml",
    //         0.5)));
    detectors.push_back(
        dynamic_cast<ObjectDetector *>(new TemplMatchingObjectDetector(
            DetectedObject::ObjectLabel::OBJECT_1,
            ros::package::getPath(config->getROSPackage()) +
                "/data/object_templates/1",
            0.8)));
    detectors.push_back(
        dynamic_cast<ObjectDetector *>(new TemplMatchingObjectDetector(
            DetectedObject::ObjectLabel::OBJECT_2,
            ros::package::getPath(config->getROSPackage()) +
                "/data/object_templates/2",
            0.8)));
    detectors.push_back(
        dynamic_cast<ObjectDetector *>(new TemplMatchingObjectDetector(
            DetectedObject::ObjectLabel::OBJECT_3,
            ros::package::getPath(config->getROSPackage()) +
                "/data/object_templates/3",
            0.8)));
}

// Filter new detected object.
// Merge the result if new object is the same one as a detected one
// Or update the results
void ObjectDetectorManager::filterNewDetectedObjects(
    std::vector<DetectedObject> new_detected_objects,
    std::vector<DetectedObject> &output_list) {

    // Processed flag. To mark a object in detected_object is processed or not
    std::vector<bool> processed(detected_objects.size(), false);

    // For new added objects (not detected before)
    std::vector<DetectedObject> new_added_objects;

    // Iterate over new detected
    for (size_t i = 0; i < new_detected_objects.size(); ++i) {
        // Check if new detected object appeared in the pass or not (check the
        // detect_object array)
        bool is_old_object;  // We already have seen the object before
        for (size_t j = 0; j < detected_objects.size(); ++j) {
            // If processed detected object, skip it
            if (processed[i]) {
                continue;
            }

            // If the object is an old object
            // Check the area of intersect over area of union

            if (
                (   static_cast<double>((new_detected_objects[i].rect & detected_objects[j].rect) .area()) 
                        / (new_detected_objects[i].rect | detected_objects[j].rect).area() > 0 
                    || cv::norm(new_detected_objects[i].rect.tl()-detected_objects[j].rect.tl()) < 20
                )
                && new_detected_objects[i].label == detected_objects[j].label) {
                is_old_object = true;


                // Update the position of the object
                detected_objects[j].rect = new_detected_objects[i].rect;

                // Update old object detection result (hit_history)
                detected_objects[j].hit_history <<= 1;
                detected_objects[j].hit_history += 1;

                // Check processed
                processed[j] = true;
            }
        }

        // If the object is not an old object, add to new_added_objects
        if (!is_old_object) {
            new_added_objects.push_back(new_detected_objects[i]);
        }
    }

    // Iterate over detected objects
    // To process all the un-updated results
    for (size_t i = 0; i < detected_objects.size(); ++i) {
        if (!processed[i]) {
            // Update old object detection result (hit_history)
            detected_objects[i].hit_history <<= 1;
        }
    }

    // Join new added objects with detected objects
    detected_objects.insert(detected_objects.end(),
                            std::make_move_iterator(new_added_objects.begin()),
                            std::make_move_iterator(new_added_objects.end()));

    // Remove all objects that have not been detected for a long time (5 frames
    // =))) )
    for (auto it = detected_objects.begin(); it != detected_objects.end();) {
        if (countNonZeroBits(it->hit_history, 5) == 0) {
            it = detected_objects.erase(it);
        } else {
            ++it;
        }
    }

    // Output a object that having good predict weight
    output_list.clear();
    for (size_t i = 0; i < detected_objects.size(); ++i) {

        if (debug_flag) {
            std::cout << i << " : " << std::bitset<32>(detected_objects[i].hit_history) << std::endl;
        }
        
        // If we detect 3/5 last frame, we trust the result
        if (countNonZeroBits(detected_objects[i].hit_history, 5) >= 0) {
            output_list.push_back(detected_objects[i]);
        }
    }

    
}

// Detect the obstacles
// Return the number of obstacles in the input image
int ObjectDetectorManager::detect(
    const cv::Mat &img, std::vector<DetectedObject> &detected_objects) {
    // New detected objects in this global search
    std::vector<DetectedObject> new_detected_objects;

    // Search global
    if (num_of_frames_to_global_search < num_of_frames_bw_global_searchs) {
        ++num_of_frames_to_global_search;
    } else {
        // Reset the counter to a global search
        num_of_frames_to_global_search = 0;

        // Iterate between detectors and search global
        for (size_t i = 0; i < num_of_detectors_each_global_search; ++i) {
            std::vector<DetectedObject> objects;

            detectors[getDetectorIter()]->detect(img, objects, debug_flag);

            // Concat results into global results
            new_detected_objects.insert(
                new_detected_objects.end(),
                std::make_move_iterator(objects.begin()),
                std::make_move_iterator(objects.end()));

            increaseDetectorIter();
        }
    }

    // Filter new results
    filterNewDetectedObjects(new_detected_objects, detected_objects);

    return detected_objects.size();
}

// Count bit 1 from the last position of an unsigned number
// input: n: number; num_of_last_bit_to_count: number of bits to count from the
// last bit
int ObjectDetectorManager::countNonZeroBits(
    unsigned int n, unsigned int num_of_last_bit_to_count) {
    int nonZeroBits = 0;
    for (size_t i = 0; i < num_of_last_bit_to_count; ++i) {
        if (n % 2 == 1) ++nonZeroBits;
        n >>= 1;
    }
    return nonZeroBits;
}

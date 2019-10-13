#include "templ_matching_object_detector.h"

using namespace std;
using namespace cv;

TemplMatchingObjectDetector::TemplMatchingObjectDetector(DetectedObject::ObjectLabel label, std::string templs_folder_path, double threshold) {

    this->label = label;
    this->threshold = threshold;

    // List all image files in objects folder => read images into obstacle
    // database Currently only support jpg, png and bmp
    std::vector<std::string> obstacle_files;
    DIR *dir;
    char buffer[PATH_MAX + 1];
    struct dirent *ent;
    if ((dir = opendir(templs_folder_path.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") != 0 &&
                strcmp(ent->d_name, "..") != 0) {
                std::string file_name(ent->d_name);
                std::string file_path = templs_folder_path + "/" + file_name;

                // Check file extension
                std::string file_ext = TemplMatchingObjectDetector::getFileExt(file_path);
                std::transform(file_ext.begin(), file_ext.end(),
                               file_ext.begin(), ::tolower);
                if (file_ext == "jpg" || file_ext == "png" ||
                    file_ext == "bmp") {
                    // std::cout << "Loading Object Template: " << file_path << std::endl;

                    cv::Mat templ = cv::imread(file_path, 1);
                    if (templ.empty()) {
                        std::cerr
                            << "Could not read template file: " << file_path
                            << std::endl;
                    } else {
                        this->object_templs.push_back(templ);
                    }
                }
            }
        }
        closedir(dir);
    } else {
        /* could not open directory */
        perror("");
        return;
    }
    

}



// Detect the objects
// Return the number of objects in the input image
int TemplMatchingObjectDetector::detect(const cv::Mat &img,
        std::vector<DetectedObject> &detected_objects, bool debug) {

    std::vector<cv::Rect> new_objects;

    // Detect each object in the image
    for (size_t i = 0; i < object_templs.size(); ++i) {
        cv::Mat obstacle_templ = object_templs[i];
        std::vector<cv::Rect> found_objects;

        // Do template matching for object
        if (matching(img, obstacle_templ, found_objects, debug)) {
            new_objects.insert(new_objects.end(),
                               std::make_move_iterator(found_objects.begin()),
                               std::make_move_iterator(found_objects.end()));
        }
    }

    detected_objects.clear();
    for (size_t i = 0; i < new_objects.size(); ++i) {
        detected_objects.push_back(DetectedObject(this->label, new_objects[i]));
    }
    
    return detected_objects.size();
    
}


// Matching object using template matching
bool TemplMatchingObjectDetector::matching(const Mat &img, const Mat &templ, std::vector<cv::Rect> &rects, bool debug) {
    // Clear the result
    rects.clear();

    // Matching object using template matching
    int match_method = TM_CCOEFF_NORMED;

    /// Create the result matrix
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    cv::Mat result;
    result.create(result_rows, result_cols, CV_32FC1);

    matchTemplate(img, templ, result, match_method);

    /// Localizing the best match with minMaxLoc
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;

    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For
    /// all the other methods, the higher the better
    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
        matchLoc = minLoc;
    } else {
        matchLoc = maxLoc;
    }

    /// Show me what you got
    if (maxVal > threshold) {

        if (debug) {
            cout << "OBJECT TEMPLATE MATCHED" << endl;
        }
        
        rects.push_back(cv::Rect(
            matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows)));
        return true;
    }

    return false;
}

// Get file extension from filepath/filename
std::string TemplMatchingObjectDetector::getFileExt(const std::string &s) {
    size_t i = s.rfind('.', s.length());
    if (i != std::string::npos) {
        return (s.substr(i + 1, s.length() - i));
    }

    return ("");
}
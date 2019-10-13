#ifndef TRAFFIC_SIGN_DETECTOR_H
#define TRAFFIC_SIGN_DETECTOR_H


#include <ros/package.h>
#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#include "config.h"
#include "traffic_sign.h"
#include "image_publisher.h"

typedef struct RecordStructure {
	std::vector<TrafficSign> prev_rects;
	std::vector<TrafficSign> curr_rects;
} Record;

class TrafficSignDetector: ImagePublisher {
    private:

        image_transport::Publisher debug_img_publisher;
        image_transport::Publisher debug_img_publisher_inrange;

        std::shared_ptr<Config> config;
        std::shared_ptr<Config> config_trafficsign;

        cv::Mat img;
        int width;
        int height;

        cv::Ptr<cv::ml::SVM> model;
        cv::HOGDescriptor hog;

        Record record;

        cv::Scalar low_HSV;
        cv::Scalar high_HSV;

        int size;
        float eps_diff;

        int min_prev_check;
        float min_prob;

        float min_area_contour, max_area_contour;

        float min_accepted_size, max_accepted_size;
        float min_accepted_ratio, max_accepted_ratio;
        

    public:

        bool debug_flag = true;

        // ==================================================
        // ********** INITIALIZE **********
        // ==================================================

        TrafficSignDetector();


        // ==================================================
        // ********** HELPER **********
        // ==================================================

        void createHOG(cv::HOGDescriptor &hog, std::vector<std::vector<float>> &HOG, std::vector<cv::Mat> &cells);

        void cvtVector2Matrix(std::vector<std::vector<float>> &HOG, cv::Mat &mat);


        // ==================================================
        // ********** SVM **********
        // ==================================================
        
        void svmPredict(cv::Ptr<cv::ml::SVM> svm, cv::Mat &response, cv::Mat &mat);


        // ==================================================
        // ********** CLASSIFY **********
        // ==================================================

        int classifySVM(cv::HOGDescriptor &hog, cv::Ptr<cv::ml::SVM> &model, cv::Mat &img);


        // ==================================================
        // ********** TRAFFIC SIGN DETECTOR **********
        // ==================================================

        // Preprocessing, auto adjustment brightness and contrast image
        void BrightnessAndContrastAuto(cv::Mat src, cv::Mat &dst, bool clipHistPercent);
        
        //  Threshold in range to get the objects with specific color
        void inRangeHSV(cv::Mat &bin_img);

        // Create bounding rects contain the objects
        void boundRectBinImg(cv::Mat bin_img, std::vector<cv::Rect> &bound_rects);

        // Combine of inRangeHSV() and boundRectBinImg()
        void boundRectByColor(std::vector<cv::Rect> &bound_rects);

        // Merge all rects have intersection
        void mergeRects(std::vector<cv::Rect> &bound_rects);

        // Extend all rects 1px in 4 directions
        void extendRect(cv::Rect &rect, int extend_dist);

        // use eps_diff to check similar rects
        // (A|B).area() / [ A + B + (A&B).area()] < eps_diff
        bool checkSimilarityRect(cv::Rect A, cv::Rect B);

        // Classify rect with training model
        void classifyCurrRect();

        // Delete expired rects and add new current rects
        void updatePrevRect();

        // Detect and classify traffic sign
        void recognize(const cv::Mat & input, std::vector<TrafficSign> &traffic_signs);
        void recognize(const cv::Mat & input, std::vector<TrafficSign> &traffic_signs, cv::Mat & draw, bool draw_result = true);
};

#endif
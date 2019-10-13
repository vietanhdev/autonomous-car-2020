#ifndef LANE_DETECTOR_H
#define LANE_DETECTOR_H

#include <iostream> 
#include <sstream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ros/ros.h>
#include <ros/console.h>

#include "config.h"
#include "road.h"
#include "image_publisher.h"


class LaneDetector: ImagePublisher {

    public:

    bool debug_flag = false;

    image_transport::Publisher debug_img_publisher_edge_points;
    image_transport::Publisher debug_img_publisher_watershed_marker_mask;
    image_transport::Publisher debug_img_publisher_no_centerline_floodfill;
    image_transport::Publisher debug_img_publisher_canny;
    image_transport::Publisher debug_img_publisher_lane_mask_floodfill;
    image_transport::Publisher debug_img_publisher_watershed_transformed;
    image_transport::Publisher debug_img_publisher_lane_mask_transformed;

    std::shared_ptr<Config> config;

    // ==== CONFIGURATION ====

    cv::Size img_size;

    // ** Floodfill
    bool canny_edge_before_floodfill = true;
    cv::Scalar floodfill_lo;
    cv::Scalar floodfill_hi;
    std::vector<cv::Point> floodfill_points;

    bool use_watershed = true;
    cv::Mat watershed_static_mask;


    // ** Perspective transform
    cv::Mat perspective_matrix_;
	cv::Mat inverse_perspective_matrix_;
    cv::Size perspective_img_size;
    cv::Mat interested_area; // Area of the transformed image appearing in the original image


    void initConfig();

    // ** Initialize perspective transform matrices
    void initPerspectiveTransform();

    // ** Constructor
    LaneDetector();

    static cv::Point getNullPoint();

    /*--- Floodfill ---*/
    void laneFloodFill(const cv::Mat & img, cv::Mat & dst, cv::Point start_point);

    void laneFloodFillPoints(const cv::Mat & img, cv::Mat & mask);

    void doCannyEdges(const cv::Mat & img, cv::Mat & mask);

    int getPerspectiveMatrix(const std::vector<cv::Point2f> corners_source, const std::vector<cv::Point2f> corners_trans);

    bool findLaneMaskFloodfill(const cv::Mat & img, cv::Mat & mask);

    void perspectiveTransform(const cv::Mat & src, cv::Mat & dst);

    void revertPerspectiveTransform(const cv::Mat &src, cv::Mat &dst);

    void removeCenterLaneLine(const cv::Mat & mask, cv::Mat & output_mask);

    void findEdgePoints(const cv::Mat & mask, size_t row, cv::Point & left_point, cv::Point & right_point);

    void findLaneEdges(const cv::Mat & img, Road & road);

    void findLaneArea(const cv::Mat &birdview_floodfill, Road &road);

    void findLanes(const cv::Mat & input, Road & road);



    // Create static mask used in watershed step
    // This step require interested_area created in initPerspectiveTransform() step
    cv::Mat createWatershedStaticMask();

    cv::Mat watershedLaneSegment(const cv::Mat &input, const cv::Mat &birdview_floodfill);
   
   
    private:
    
    // SUPPPORTING FUNCTIONS
    // Find the index of the contour that has the biggest area
    size_t findLargestContourIndex( std::vector<std::vector<cv::Point> >  & contours);

    // Extract the largest contour of a grayscale image
    // Return a binary mask of largest contour
    cv::Mat findLargestContour(const cv::Mat & input);


};

#endif
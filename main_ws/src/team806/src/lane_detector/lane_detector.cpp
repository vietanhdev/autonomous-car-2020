#include "lane_detector.h"

using namespace cv;
using namespace std;

void LaneDetector::initConfig()
{

    img_size = config->getSize("image_size");

    // ** Watershed
    use_watershed = config->get<bool>("lane_use_watershed_segmentation");

    // ** Floodfill
    floodfill_lo = config->getScalar3("lane_floodfill_lo");
    floodfill_hi = config->getScalar3("lane_floodfill_hi");
    canny_edge_before_floodfill = config->get<bool>("canny_edge_before_floodfill");

    std::string floodfill_points_str = config->get<std::string>("lane_floodfill_points");

    // Extract the lane floodfill points
    ROS_INFO_STREAM("Reading lane_floodfill_point");
    std::vector<int> numbers = Config::extractIntegers(floodfill_points_str);
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        ROS_INFO_STREAM(numbers[i] << " ");
    }

    ROS_ASSERT_MSG(!numbers.empty() && numbers.size() % 2 == 0, "Failed on reading lane_floodfill_points from config file. Config read from string: %s", floodfill_points_str.c_str());

    for (size_t i = 0; i < numbers.size() / 2; ++i)
    {
        floodfill_points.push_back(cv::Point(numbers[i], numbers[i + 1]));
    }
}

// ** Initialize perspective transform matrices
void LaneDetector::initPerspectiveTransform()
{

    // TRANSFORM
    std::vector<cv::Point2f> corners_source(4);
    std::vector<cv::Point2f> corners_trans(4);

    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input
    corners_source[0] = cv::Point2f(config->getPoint("lane_transform_src_tl"));
    corners_source[1] = cv::Point2f(config->getPoint("lane_transform_src_tr"));
    corners_source[2] = cv::Point2f(config->getPoint("lane_transform_src_br"));
    corners_source[3] = cv::Point2f(config->getPoint("lane_transform_src_bl"));

    // The 4 points where the mapping is to be done , from top-left in clockwise order
    corners_trans[0] = cv::Point2f(config->getPoint("lane_transform_dst_tl"));
    corners_trans[1] = cv::Point2f(config->getPoint("lane_transform_dst_tr"));
    corners_trans[2] = cv::Point2f(config->getPoint("lane_transform_dst_br"));
    corners_trans[3] = cv::Point2f(config->getPoint("lane_transform_dst_bl"));

    getPerspectiveMatrix(corners_source, corners_trans);

    perspective_img_size = cv::Size(config->getSize("perspective_img_size"));

    cv::Mat tmp(img_size, CV_8UC1, cv::Scalar(255));
    perspectiveTransform(tmp, interested_area);
}

// This step require interested_area created in initPerspectiveTransform() step
cv::Mat LaneDetector::createWatershedStaticMask() {
    cv::Mat static_mask;
    cv::threshold(interested_area, static_mask, 0.5, 255, THRESH_BINARY_INV);

    // Reduce the white area
    for (size_t i = 0; i < static_mask.cols; ++i) {

        // Skip black area
        size_t j = 0;
        while (j < static_mask.rows) {
            if (static_mask.at<uchar>(j, i) == 0) {
                ++j;
            } else {
                break;
            }
        }

        size_t k = j;
        // Remove 5 rows
        while (k < static_mask.rows && k < j + 5) {
            static_mask.at<uchar>(k, i) = 0;
            ++k;
        }

    }

    return static_mask;

}

// ** Constructor
LaneDetector::LaneDetector()
{

    config = Config::getDefaultConfigInstance();

    debug_flag = config->get<bool>("debug_lane_detector");

    // Init debug image publishers
    if (debug_flag) {
        debug_img_publisher_edge_points = createImagePublisher("lane_detector/edge_points", 1);
        debug_img_publisher_watershed_marker_mask = createImagePublisher("lane_detector/watershed_marker_mask", 1);
        debug_img_publisher_no_centerline_floodfill = createImagePublisher("lane_detector/floodfill_and_remove_centerline", 1);
        debug_img_publisher_canny = createImagePublisher("lane_detector/canny", 1);
        debug_img_publisher_lane_mask_floodfill = createImagePublisher("lane_detector/lane_mask_floodfill", 1);
        debug_img_publisher_watershed_transformed = createImagePublisher("lane_detector/watershed_transformed", 1);
        debug_img_publisher_lane_mask_transformed = createImagePublisher("lane_detector/lane_mask_transformed", 1);
    }
    


    initConfig();
    initPerspectiveTransform();

    // Watershed experiment
    // This step require interested_area created in initPerspectiveTransform() step
    watershed_static_mask = createWatershedStaticMask();

    // Failback: read from file
    // watershed_static_mask = cv::imread(Config::getDataFile("watershed_mask.png"), CV_LOAD_IMAGE_GRAYSCALE);

}

cv::Point LaneDetector::getNullPoint()
{
    return cv::Point(-1, -1);
}

/*--- Floodfill ---*/
void LaneDetector::laneFloodFill(const cv::Mat &img, cv::Mat &dst, cv::Point start_point)
{

    int ffillMode = 2; // 2: gradient fill, 1: Fixed Range floodfill

    int connectivity = 4; // or 8?
    int newMaskVal = 255;
    int flags = connectivity + (newMaskVal << 8) +
                (ffillMode == 1 ? cv::FLOODFILL_FIXED_RANGE : 0);

    // Working in HSV color space
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    int area;
    cv::Point seed = start_point;
    cv::Scalar newVal(255);

    // Flood fill
    cv::Mat result;
    cv::Mat mask = cv::Mat::zeros(cv::Size(img.cols + 2, img.rows + 2), CV_8UC1);
    cv::cvtColor(img, result, cv::COLOR_BGR2GRAY);
    area = cv::floodFill(result, mask, seed, newVal, 0, floodfill_lo, floodfill_hi, flags);

    dst = mask;
}

void LaneDetector::laneFloodFillPoints(const cv::Mat &input, cv::Mat &mask)
{

    if (floodfill_points.empty())
        return;

    cv::Mat img = input.clone();

    // Floodfill from the first point
    cv::Mat m1;
    laneFloodFill(img, m1, floodfill_points[0]);
    mask = m1.clone();

    // Floodfill from reamining points
    for (size_t i = 0; i < floodfill_points.size(); ++i)
    {
        laneFloodFill(img, m1, floodfill_points[i]);
        mask |= m1;
    }
}

void LaneDetector::doCannyEdges(const cv::Mat &img, cv::Mat &mask)
{
    blur(img, mask, cv::Size(3, 3));
    int low_threshold = 50;
    int high_threshold = 200;
    cv::Canny(img, mask, low_threshold, high_threshold, 3);
}

int LaneDetector::getPerspectiveMatrix(const std::vector<cv::Point2f> corners_source,
                                       const std::vector<cv::Point2f> corners_trans)
{

    ROS_ASSERT_MSG(corners_source.size() == 4 && corners_trans.size() == 4, "Error in GetPerspectiveMatrix.");

    perspective_matrix_ = cv::getPerspectiveTransform(corners_source, corners_trans);
    inverse_perspective_matrix_ = cv::getPerspectiveTransform(corners_trans, corners_source);
    return true;
} //GetPerspectiveMatrix

// Find the index of the contour that has the biggest area
size_t LaneDetector::findLargestContourIndex(std::vector<std::vector<cv::Point>> &contours)
{
    int largest_area = 0;
    int largest_contour_index = -1;

    // iterate through each contour.
    for (int i = 0; i < contours.size(); i++)
    {
        double a = contourArea(contours[i], false); //  Find the area of contour
        if (a > largest_area)
        {
            largest_area = a;
            largest_contour_index = i; //Store the index of largest contour
        }
    }

    return largest_contour_index;
}

// Extract the largest contour of a grayscale image
// Return a mask of largest contour
cv::Mat LaneDetector::findLargestContour(const cv::Mat &input)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(input, contours, hierarchy, cv::RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    cv::Mat largest_contour_mask = cv::Mat(input.rows, input.cols, CV_8UC1, cv::Scalar(0));
    if (contours.empty())
    {
        return largest_contour_mask;
    }

    size_t largest_contour_index = findLargestContourIndex(contours);

    drawContours(largest_contour_mask, contours, largest_contour_index, cv::Scalar(255), CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.

    return largest_contour_mask;
}

bool LaneDetector::findLaneMaskFloodfill(const cv::Mat &img, cv::Mat &mask)
{

    // ** do Floodfill
    cv::Mat flood;
    laneFloodFillPoints(img, flood);

    // ** Separate the white area to distinct road and other things
    int dilate_size = 1;
    cv::Mat dilate_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                       cv::Size(2 * dilate_size + 1, 2 * dilate_size + 1),
                                                       cv::Point(dilate_size, dilate_size));
    dilate(flood, flood, dilate_element);

    int erosion_size = 5;
    cv::Mat erosion_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                        cv::Point(erosion_size, erosion_size));
    erode(flood, flood, erosion_element);

    dilate_size = 2;
    dilate_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(2 * dilate_size + 1, 2 * dilate_size + 1),
                                               cv::Point(dilate_size, dilate_size));
    dilate(flood, flood, dilate_element);

    // ** Find the biggest area => road

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(flood, contours, hierarchy, cv::RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        return false;
    }

    mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    int largest_contour_index = findLargestContourIndex(contours);

    drawContours(mask, contours, largest_contour_index, cv::Scalar(255), CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.

    return true;
}

void LaneDetector::perspectiveTransform(const cv::Mat &src, cv::Mat &dst)
{
    cv::warpPerspective(src, dst, perspective_matrix_, perspective_img_size);
}

void LaneDetector::revertPerspectiveTransform(const cv::Mat &src, cv::Mat &dst)
{
    cv::warpPerspective(src, dst, inverse_perspective_matrix_, img_size);
}

void LaneDetector::removeCenterLaneLine(const cv::Mat &mask, cv::Mat &output_mask)
{

    cv::Mat inner_objects = mask.clone();
    floodFill(inner_objects, cv::Point(0, perspective_img_size.height - 1), cv::Scalar(255));
    floodFill(inner_objects, cv::Point(perspective_img_size.width - 1, perspective_img_size.height - 1), cv::Scalar(255));

    cv::threshold(inner_objects, inner_objects, 1, 255, CV_THRESH_BINARY_INV);

    int dilate_size = 2;
    cv::Mat dilate_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                       cv::Size(2 * dilate_size + 1, 2 * dilate_size + 1),
                                                       cv::Point(dilate_size, dilate_size));
    dilate(inner_objects, inner_objects, dilate_element);

    output_mask = mask.clone();
    output_mask |= inner_objects;
}

void LaneDetector::findEdgePoints(const cv::Mat &mask, size_t row, cv::Point &left_point, cv::Point &right_point)
{

    // ** Find left point (this point lies on the left lane line)
    int col = 0;

    // Skip area not in interested area
    while (interested_area.at<uchar>(cv::Point(col, row)) == 0)
    {
        ++col;
        if (col >= perspective_img_size.width)
            break;
    }

    // Find a not-lane-line pixel
    while (col < perspective_img_size.width && mask.at<uchar>(cv::Point(col, row)) > 0)
    {
        ++col;
        if (col >= perspective_img_size.width)
            break;
    }

    // Find left point
    while (col < perspective_img_size.width && mask.at<uchar>(cv::Point(col, row)) == 0)
    {
        ++col;
        if (col >= perspective_img_size.width)
            break;
    }

    // Check if found a left point
    if (col < perspective_img_size.width)
    {
        left_point = cv::Point(col, row);
    }
    else
    {
        left_point = getNullPoint();
    }

    // ** Find right point (this point lies on the right lane line)
    col = perspective_img_size.width - 1;

    // Skip area not in interested area
    while (interested_area.at<uchar>(cv::Point(col, row)) == 0)
    {
        --col;
        if (col < 0)
            break;
    }

    // Find a not-lane-line pixel
    while (col >= 0 && mask.at<uchar>(cv::Point(col, row)) > 0)
    {
        --col;
        if (col < 0)
            break;
    }

    // Find right point
    while (col >= 0 && mask.at<uchar>(cv::Point(col, row)) == 0)
    {
        --col;
        if (col < 0)
            break;
    }

    // Check if found a right point
    if (col >= 0)
    {
        right_point = cv::Point(col, row);
    }
    else
    {
        right_point = getNullPoint();
    }
}

void LaneDetector::findLaneEdges(const cv::Mat &img, Road &road)
{

    cv::Mat tmp;
    cvtColor(img, tmp, cv::COLOR_GRAY2BGR);

    cv::Point left, right;
    cv::Point middle;

    road.left_points.clear();
    road.right_points.clear();
    road.middle_points.clear();

    for (int i = 0; i < perspective_img_size.height - 1; i += 5)
    {
        findEdgePoints(img, i, left, right);

        if (left != getNullPoint() && right == getNullPoint())
        {
            right = cv::Point(left.x + Road::road_width, left.y);
        }

        if (right != getNullPoint() && left == getNullPoint())
        {
            left = cv::Point(right.x - Road::road_width, right.y);
        }

        if (right != getNullPoint() && left != getNullPoint())
        {
            middle = (left + right) / 2;

            if (debug_flag)
            {
                circle(tmp, middle, 1, cv::Scalar(255, 0, 0), 2);
            }

            road.left_points.push_back(left);
            road.right_points.push_back(right);
            road.middle_points.push_back(middle);
        }

        if (debug_flag)
        {
            circle(tmp, left, 1, cv::Scalar(0, 255, 0), 2);
            circle(tmp, right, 1, cv::Scalar(0, 0, 255), 2);
        }
    }

    if (debug_flag)
    {
        // cv::imshow("edge points", tmp);
        // cv::waitKey(1);
        publishImage(debug_img_publisher_edge_points, tmp);
    }
}

cv::Mat LaneDetector::watershedLaneSegment(const cv::Mat &input, const cv::Mat &floodfill_mask) {

    cv::Mat watershed_mask;
    cv::Mat img = input.clone();

    // ===== Create watershed mask =====
    // Watershed mask is created from :
    //      + Loading a file: watershed_mask.png into watershed_static_mask
    //      + Using floodfill result ===> watershed contour of lane
    //      + Using floodfill result ===> watershed contours of other parts (2 side)

    
    // STEP: Using floodfill result ===> watershed contour of lane

    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(floodfill_mask, dist, CV_DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);

    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);

    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Threshold to visualize
    threshold(dist_8u, dist_8u, 0.5, 255, THRESH_BINARY);

    dist_8u = findLargestContour(dist_8u);


    // Combine mask loaded from image with the mask created using lane
    cv::Mat marker_mask = watershed_static_mask | dist_8u;


    // STEP: Find the watershed contour of 2 sides (2 parts on the left and on the right of lane)
    // Idea: Create 2 white rectangle on the left and on the right (which dont have any interect part with the lane area)

    int x_min = 999;
    int x_max = -999;
    for (int i = 0; i < floodfill_mask.rows; ++i)
    {
        for (int j = 0; j < floodfill_mask.cols; ++j)
        {
            if (j < x_min && floodfill_mask.at<uchar>(i, j) > 0)
                x_min = j;
            if (j > x_max && floodfill_mask.at<uchar>(i, j) > 0)
                x_max = j;
        }
    }

    int lane_bound_left, lane_bound_right;
    if (x_min > 45)
    {
        lane_bound_left = x_min - 30;
    }

    if (x_max < perspective_img_size.width - 45)
    {
        lane_bound_right = x_max + 30;
    }

    // Write the mask part of 2 sides to the watershed mask
    rectangle(marker_mask, cv::Rect(0, 0, lane_bound_left, marker_mask.rows), cv::Scalar(255), -1);
    rectangle(marker_mask, cv::Rect(lane_bound_right, 0, marker_mask.cols - lane_bound_right, marker_mask.rows), cv::Scalar(255), -1);

    if (debug_flag)
    {
        // cv::imshow("watershed mask", marker_mask);
        // cv::waitKey(1);
        publishImage(debug_img_publisher_watershed_marker_mask, marker_mask);
    }

    // Doing watershed
    cv::Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cvtColor(gray, gray, COLOR_GRAY2BGR);

    int i, j, compCount = 0;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(marker_mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return watershed_mask;

    Mat markers(marker_mask.size(), CV_32S);
    markers = Scalar::all(0);
    int idx = 0;
    for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
        drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
    if (compCount == 0)
        return watershed_mask;

    watershed(img, markers);

    Mat wshed(markers.size(), CV_8UC3);
    // paint the watershed image
    int lane_index = markers.at<int>(145, 160);
    for (i = 0; i < markers.rows; i++)
        for (j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index == -1)
                wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            else if (index != lane_index)
            {
                wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            }
            else
            {
                wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            }
        }

    cv::cvtColor(wshed, watershed_mask, COLOR_BGR2GRAY);

    if (debug_flag)
    {
        wshed = wshed * 0.5 + gray * 0.5;
        // imshow("watershed transform", wshed);
        // cv::waitKey(1);
        publishImage(debug_img_publisher_watershed_transformed, wshed);
    }
}

void LaneDetector::findLaneArea(const cv::Mat &birdview_floodfill, Road &road) {
    int lane_area = 0;

    cv::Mat no_centerline_floodfill;
    removeCenterLaneLine(birdview_floodfill, no_centerline_floodfill);

    if (debug_flag)
    {
        // cv::imshow("Floodfill -> remove centerline", no_centerline_floodfill);
        // cv::waitKey(1);
        publishImage(debug_img_publisher_no_centerline_floodfill, no_centerline_floodfill);
    }

    // Calculate lane area by counting white point
    for (size_t i = 0; i < no_centerline_floodfill.rows; ++i)
    {
        for (size_t j = 0; j < no_centerline_floodfill.cols; ++j)
        {
            if (no_centerline_floodfill.at<uchar>(i, j) > 0)
            {
                ++lane_area;
            }
        }
    }

    road.lane_area = lane_area;
}

void LaneDetector::findLanes(const cv::Mat &input, Road &road)
{

    cv::Mat img = input.clone();


    // ================================================
    // *** Canny Edge
    // This step is for create edges between lane and other area
    // to have better floodfill result (prevent filling other parts as lane)
    // ================================================
    cv::Mat img_canny = input.clone();
    if (canny_edge_before_floodfill)
    {
        cv::Mat canny_edges;
        doCannyEdges(img_canny, canny_edges);

        cv::Mat canny_edges_bgr;
        cv::cvtColor(canny_edges, canny_edges_bgr, cv::COLOR_GRAY2BGR);

        img_canny = img_canny | canny_edges_bgr;

        if (debug_flag)
        {
            // cv::imshow("img + canny_edges", img);
            // cv::waitKey(1);
            publishImage(debug_img_publisher_canny, img);
        }
    }

    // ================================================
    // *** Find Lane Mask - Floodfilling
    // Find lane mask by floodfilling
    // ================================================
    cv::Mat lane_mask_floodfill;
    bool floodfill_result = findLaneMaskFloodfill(img_canny, lane_mask_floodfill);

    // TODO: FIX this
    // Dont just return
    if (!floodfill_result)
        return;

    if (debug_flag)
    {
        // cv::imshow("Lane mask by Floodfill", lane_mask_floodfill);
        // cv::waitKey(1);
        publishImage(debug_img_publisher_lane_mask_floodfill, lane_mask_floodfill);
    }

    // ================================================
    // *** Perspective transform
    // Perspective transform the floodfill result
    // ================================================
    cv::Mat birdview_floodfill;
    perspectiveTransform(lane_mask_floodfill, birdview_floodfill);



    // ================================================
    // *** Find the lane area
    // Find lane area on non-centerline floodfilled lane mask
    // to determine if we are at the turning position
    // ================================================
    findLaneArea(birdview_floodfill, road);


    // ================================================
    // *** Find the lane mask using Watershed
    // ================================================

    cv::Mat lane_mask;

    if (!use_watershed) {
        lane_mask = birdview_floodfill;
    } else {

        // Create a birdview transform of original image
        // For watershed
        cv::Mat birdview_img;
        perspectiveTransform(img, birdview_img);

        cv::Mat watershed_result;
        watershed_result = watershedLaneSegment(birdview_img, birdview_floodfill);

        lane_mask = watershed_result;

        if (debug_flag)
        {
            // cv::imshow("lane_mask > perspective transform", lane_mask);
            // cv::waitKey(1);
            publishImage(debug_img_publisher_lane_mask_transformed, lane_mask);
        }

    }


    // ================================================
    // *** Find lane edges
    // Converting lane mask into road model
    // ================================================

    findLaneEdges(lane_mask, road);

    // Xuất lane mask cho An
    cv::Mat lane_mask_origin_view;
    revertPerspectiveTransform(lane_mask, lane_mask_origin_view);
    road.lane_mask = lane_mask_origin_view;

}

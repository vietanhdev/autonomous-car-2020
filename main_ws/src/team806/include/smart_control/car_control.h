#ifndef CARCONTROL_H
#define CARCONTROL_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include "timer.h"

#include <ros/ros.h>
#include <ros/package.h>
#include "std_msgs/Float32.h"

#include <vector>
#include <math.h>

#include "config.h"
#include "road.h"
#include "lane_detector.h"
#include "obstacle_detector.h"
#include "traffic_sign.h"

// #include "PID.h"



class CarControl 
{
public:

    bool debug_flag = false;

    float MIN_SPEED = 15;
    float MAX_SPEED = 60;
    float MAX_ANGLE = 60;

    CarControl();
    ~CarControl();
    void driverCar(float speed_data, float angle_data);
    void driverCar(Road & road, const std::vector<TrafficSign> & traffic_signs);

    // Reset parameter to start a new racing round
    void resetRound();


private:

    void publishSignal(float speed_data, float angle_data);

    std::shared_ptr<Config> config;

    // Manage control signal publish interval
    Timer::time_point_t last_signal_publish_time_point;
    Timer::time_duration_t signal_publish_interval = 1;


    int obstacle_avoid_coeff = 0;
    Timer::time_point_t obstacle_avoiding_time_point;


    int last_sign_id = -1;
    Timer::time_point_t last_sign_time_point;
    bool prepare_to_turn = false;
    bool is_turning = false;
    int success_turning_times = 0;
    int turning_coeff = 0;

    Timer::time_point_t turning_time_point;

    ros::NodeHandle node_obj1;
    ros::NodeHandle node_obj2;
    
    ros::Publisher steer_publisher;
    ros::Publisher speed_publisher;

    // Kalman filter
    cv::KalmanFilter KF;
    cv::Mat state; /* (phi, delta_phi) */
    cv::Mat process_noise;
    cv::Mat measurement;


    // TODO: complete this function
    double kalmanFilterAngle(double angle) {

        return angle;

    }


    // Starting time of the round
    Timer::time_point_t round_start_time;


    // Quick start
    // We assume that at the begining, the road is straight so we increase the speed as much as possible
    bool quick_start = false;
    float quick_start_speed = 80;
    Timer::time_duration_t quick_start_time = 3000;



    // ======================== LANE ==============================


    float delta_to_angle_coeff = -0.5;
    float middle_interested_point_pos = 0.6;

    // Adjust middle point. 
    // Negative value brings car to the left, 
    // Positive value brings car to the right
    // Pratical safe range [-20, 20]
    int middle_point_adjustment = 0;

    float line_diff_to_angle_coeff = -1;
    float line_diff_effect_speed_coeff = 1; // How the angle of line effects the speed

    // Minimum number of middle points found by lane detector.
    // If the number of middle points less than this value, do nothing with car controlling
    int min_num_of_middle_points = 10;


    // ======================== TRAFFIC SIGN ==============================

    // Conditions to turn
    bool turn_on_trafficsign_by_lane_area = true; // Rẽ khi đến ngã ba, ngã tư (diện tích đường tăng lên)
    bool turn_on_trafficsign_by_passed_time = false; // Rẽ sau khi thấy biển 1 thời gian
    int traffic_sign_passed_time_lower_bound = 2000; // Cận dưới của thời gian bắt đầu rẽ.
    int traffic_sign_passed_time_higher_bound = 2500; // Cận trên của thời gian bắt đầu rẽ.

    // Minimum area of traffic sign rectangle boundary
    int min_traffic_sign_bound_area = 1000;

    // Valid duration for traffic sign recognition
    int traffic_sign_valid_duration = 3000;

    int num_of_crossed_trafficsign = 0;

    // Speed when preparing to turn (because of the apearance of a traffic sign)
    float speed_on_preparing_to_turn_trafficsign = 30;

    // Using lane area as a signal to turn
    // If the lane > this value, change the direction of the car
    int lane_area_to_turn = 18000;

    // The angle we use to change direction of the car when we meet a traffic sign
    float turning_angle_on_trafficsign = 50;

    float speed_on_turning_trafficsign = 10;

    Timer::time_duration_t turning_duration_trafficsign = 1000;



    // ======================== SPEEDUP!!! ==============================
    Timer::time_duration_t duration_speedup_after_traffic_sign_1 = 0;
    float speed_on_speedup_after_traffic_sign_1 = 80;
    Timer::time_point_t crossed_traffic_sign_1_time_point;

    Timer::time_duration_t duration_speedup_after_traffic_sign_2 = 0;
    float speed_on_speedup_after_traffic_sign_2 = 80;
    Timer::time_point_t crossed_traffic_sign_2_time_point;


    float last_speed_data, last_angle_data;

    void readConfig();

};

#endif
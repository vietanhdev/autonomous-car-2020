#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <thread>

#include <ros/console.h>
#include "car_control.h"
#include "config.h"
#include "image_publisher.h"
#include "lane_detector.h"
#include "obstacle_detector.h"
#include "road.h"
#include "timer.h"
#include "traffic_sign.h"
#include "traffic_sign_detector.h"
#include "traffic_sign_detector_2.h"
#include "opencv2/ximgproc/segmentation.hpp"
#include "object_detector_manager.h"


using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;

// This flag turn to true on the first time receiving image from simulator
bool racing = false;

bool draw_road_area = false;

bool use_traffic_sign_detector_2 = false;
bool debug_show_fps = false;

std::shared_ptr<CarControl> car;
std::shared_ptr<LaneDetector> lane_detector;
std::shared_ptr<TrafficSignDetector> sign_detector;
std::shared_ptr<TrafficSignDetector2> sign_detector_2;
std::shared_ptr<ObjectDetectorManager> object_detector_manager;

std::shared_ptr<ImagePublisher> img_publisher;
// ============ Current State ==================

std::mutex current_img_mutex;
cv::Mat current_img;  // current image received by car

std::mutex road_mutex;
Road road;

std::mutex traffic_signs_mutex;
std::vector<TrafficSign> traffic_signs;

std::mutex obstacles_mutex;
std::vector<cv::Rect> obstacles;

std::mutex detected_objects_mutex;
std::vector<DetectedObject> detected_objects;

// To calculate the speed of image transporting
// ==> The speed of the hold system => Adjust the car controlling params
long long int num_of_frames = 0;
Timer::time_point_t start_time_point;

Mat markerMask;

void setLabel(cv::Mat &im, const std::string label, const cv::Point &org) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    cv::Size text =
        cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, org + cv::Point(0, baseline),
                  org + cv::Point(text.width, -text.height), CV_RGB(0, 0, 0),
                  CV_FILLED);
    cv::putText(im, label, org, fontface, scale, CV_RGB(0, 255, 0), thickness,
                8);
}

void drawResultRound1() {
    while (true) {
        cv::Mat draw;

        // Copy current image to draw
        {
            std::lock_guard<std::mutex> guard(current_img_mutex);
            draw = current_img.clone();
        }

        if (draw.empty()) {
            Timer::delay(500);
            continue;
        }

        // Draw the road
        if (draw_road_area) {
            std::lock_guard<std::mutex> guard(road_mutex);
            cv::Mat red = Mat(draw.size(), draw.type(), Scalar(0,0,255));

            cv::bitwise_and(red, draw, draw, road.lane_mask);
        }

        // Draw the lane center
        {
            std::lock_guard<std::mutex> guard(road_mutex);
            if (!road.middle_points.empty()) {
                circle(draw, road.middle_points[road.middle_points.size() - 1],
                       1, cv::Scalar(255, 0, 0), 3);
            }
        }

        // Draw traffic signs
        {
            std::lock_guard<std::mutex> guard(traffic_signs_mutex);
            for (int i = 0; i < traffic_signs.size(); ++i) {
                rectangle(draw, traffic_signs[i].rect, Scalar(0, 0, 255), 2);
                std::string text;
                if (traffic_signs[i].id == TrafficSign::SignType::TURN_LEFT) {
                    text = "turn_left";
                } else if (traffic_signs[i].id ==
                           TrafficSign::SignType::TURN_RIGHT) {
                    text = "turn_right";
                }
                setLabel(draw, text, traffic_signs[i].rect.tl());
            }
        }


        // Draw obstacles
        {
            std::lock_guard<std::mutex> guard(detected_objects_mutex);
            
            cv::Mat overlay; // overlay image to draw a transparent rectangles
            double alpha = 0.3;

            // copy the source image to an overlay
            draw.copyTo(overlay);

            for (int i = 0; i < detected_objects.size(); ++i) {
                cv::rectangle(overlay, detected_objects[i].rect, cv::Scalar(0, 0, 255), -1);
                setLabel(draw, "obstruction", detected_objects[i].rect.tl());
            }

            // blend the overlay with the draw
            cv::addWeighted(overlay, alpha, draw, 1 - alpha, 0, draw);
        }

        imshow("RESULT", draw);
        waitKey(1);

        Timer::delay(100);
    }
}

Timer::time_point_t last_lane_detect_time;
void laneDetectorThread() {
    float lane_detector_fps = 40;
    float current_fps;
    Timer::time_duration_t thread_delay_time = 0;
    Timer::time_duration_t thread_delay_time_step = 4;

    while (true) {
        // Copy current image
        cv::Mat img;
        {
            std::lock_guard<std::mutex> guard(current_img_mutex);
            img = current_img.clone();
        }

        // Find lane lines
        if (!img.empty()) {
            Road new_road;
            lane_detector->findLanes(img, new_road);

            {
                std::lock_guard<std::mutex> guard(road_mutex);
                road = new_road;
            }

            Timer::delay(thread_delay_time);

            current_fps = 1000.0 / Timer::calcTimePassed(last_lane_detect_time);
            if (current_fps > lane_detector_fps) {
                thread_delay_time += thread_delay_time_step;
            } else if (current_fps < lane_detector_fps &&
                       thread_delay_time >= thread_delay_time_step) {
                thread_delay_time -= thread_delay_time_step;
            }

            // cout << "Lane Detect FPS: " << current_fps << endl;

            last_lane_detect_time = Timer::getCurrentTime();
        }
    }
}

Timer::time_point_t last_car_control_time;
void carControlThread() {
    float config_fps = 40;
    float current_fps;
    Timer::time_duration_t thread_delay_time = 0;
    Timer::time_duration_t thread_delay_time_step = 4;

    while (true) {
        {
            std::lock_guard<std::mutex> guard(road_mutex);
            std::lock_guard<std::mutex> guard2(traffic_signs_mutex);
            car->driverCar(road, traffic_signs);
        }

        Timer::delay(thread_delay_time);

        current_fps = 1000.0 / Timer::calcTimePassed(last_car_control_time);
        if (current_fps > config_fps) {
            thread_delay_time += thread_delay_time_step;
        } else if (current_fps < config_fps &&
                   thread_delay_time >= thread_delay_time_step) {
            thread_delay_time -= thread_delay_time_step;
        }

        // cout << "Car Control FPS: " << current_fps << endl;

        last_car_control_time = Timer::getCurrentTime();
    }
}

Timer::time_point_t last_trafficsign_detect_time;
void trafficSignThread() {
    float config_fps = 20;
    float current_fps;
    Timer::time_duration_t thread_delay_time = 0;
    Timer::time_duration_t thread_delay_time_step = 4;

    while (true) {
        // Copy current image
        cv::Mat img;
        {
            std::lock_guard<std::mutex> guard(current_img_mutex);
            img = current_img.clone();
        }

        // Detect traffic sign
        if (!img.empty()) {
            if (!use_traffic_sign_detector_2) {
                sign_detector->recognize(img, traffic_signs);
            } else {
                sign_detector_2->recognize(img, traffic_signs);
            }

            Timer::delay(thread_delay_time);

            current_fps =
                1000.0 / Timer::calcTimePassed(last_trafficsign_detect_time);
            if (current_fps > config_fps) {
                thread_delay_time += thread_delay_time_step;
            } else if (current_fps < config_fps &&
                       thread_delay_time >= thread_delay_time_step) {
                thread_delay_time -= thread_delay_time_step;
            }

            // cout << "Traffic sign detect FPS: " << current_fps << endl;

            last_trafficsign_detect_time = Timer::getCurrentTime();
        }

        ROS_ERROR("LOI");
    }
}

Timer::time_point_t last_obstacle_detect_time;
void obstacleDetectorThread() {
    float config_fps = 30;
    float current_fps;
    Timer::time_duration_t thread_delay_time = 0;
    Timer::time_duration_t thread_delay_time_step = 4;

    while (true) {
        // Copy current image
        cv::Mat img;
        {
            std::lock_guard<std::mutex> guard(current_img_mutex);
            img = current_img.clone();
        }

        if (!img.empty()) {

            vector<DetectedObject> objects;
            object_detector_manager->detect(img, objects);
            
            // Update the result
            {
                std::lock_guard<std::mutex> guard(detected_objects_mutex);
                detected_objects = objects;
            }

         
            Timer::delay(thread_delay_time);

            current_fps =
                1000.0 / Timer::calcTimePassed(last_obstacle_detect_time);
            if (current_fps > config_fps) {
                thread_delay_time += thread_delay_time_step;
            } else if (current_fps < config_fps &&
                       thread_delay_time >= thread_delay_time_step) {
                thread_delay_time -= thread_delay_time_step;
            }

            // cout << "Obstacle detect FPS: " << current_fps << endl;

            last_obstacle_detect_time = Timer::getCurrentTime();
        }
    }
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    Mat out;
    try {
        if (racing == false) {
            racing = true;
            car->resetRound();
        }

        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        // Update current image
        {
            std::lock_guard<std::mutex> guard(current_img_mutex);
            current_img = cv_ptr->image.clone();
        }

    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.",
                  msg->encoding.c_str());
    }

    ++num_of_frames;

    double sim_speed = static_cast<double>(num_of_frames) /
                       Timer::calcTimePassed(start_time_point) * 1000;

    if (debug_show_fps) ROS_INFO_STREAM("Simulation speed: " << sim_speed);

    // cout << "Simulation speed: " << sim_speed << endl;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "team806_node");
    std::shared_ptr<Config> config = Config::getDefaultConfigInstance();

    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                       ros::console::levels::Debug)) {
        ros::console::notifyLoggerLevelsChanged();
    }


    img_publisher = std::make_shared<ImagePublisher>();

    debug_show_fps = config->get<bool>("debug_show_fps");
    draw_road_area = config->get<bool>("draw_road_area");

    lane_detector = std::shared_ptr<LaneDetector>(new LaneDetector());

    // obstacle_detector =
    //     std::shared_ptr<ObstacleDetector>(new ObstacleDetector());

    use_traffic_sign_detector_2 =
        config->get<bool>("use_traffic_sign_detector_2");
    if (!use_traffic_sign_detector_2) {
        sign_detector =
            std::shared_ptr<TrafficSignDetector>(new TrafficSignDetector());
    } else {
        sign_detector_2 =
            std::shared_ptr<TrafficSignDetector2>(new TrafficSignDetector2());
    }

    car = std::shared_ptr<CarControl>(new CarControl());

    object_detector_manager = std::shared_ptr<ObjectDetectorManager>(new ObjectDetectorManager());


    // SET DEBUG OPTIONS
    car->debug_flag = config->get<bool>("debug_car_control");

    start_time_point = Timer::getCurrentTime();

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub =
        it.subscribe(config->getTeamName() + "_image", 1, imageCallback);

    std::thread round_1_result_thread(drawResultRound1);
    std::thread lane_detector_thread(laneDetectorThread);
    std::thread obstacle_detector_thread(obstacleDetectorThread);
    std::thread trafficsign_detector_thread(trafficSignThread);
    std::thread car_control_thread(carControlThread);

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    cv::destroyAllWindows();
}

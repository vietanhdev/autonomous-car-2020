#include "car_control.h"


CarControl::CarControl()
{

    config = Config::getDefaultConfigInstance();

    readConfig();

    steer_publisher = node_obj1.advertise<std_msgs::Float32>(config->getTeamName() + "_steerAngle", 1);
    speed_publisher = node_obj2.advertise<std_msgs::Float32>(config->getTeamName() + "_speed", 1);

    publishSignal(0,0);
    last_signal_publish_time_point = Timer::getCurrentTime();

}

CarControl::~CarControl() {}

void CarControl::readConfig() {

    // Control Signal
    MAX_SPEED = config->get<float>("max_speed");
    MAX_ANGLE = config->get<float>("max_angle");

    line_diff_to_angle_coeff = config->get<float>("line_diff_to_angle_coeff");
    delta_to_angle_coeff = config->get<float>("delta_to_angle_coeff");
    middle_interested_point_pos = config->get<float>("middle_interested_point_pos");
    min_num_of_middle_points = config->get<int>("min_num_of_middle_points");

    turn_on_trafficsign_by_lane_area =  config->get<bool>("turn_on_trafficsign_by_lane_area");; // Rẽ khi đến ngã ba, ngã tư (diện tích đường tăng lên)
    turn_on_trafficsign_by_passed_time =  config->get<bool>("turn_on_trafficsign_by_passed_time");; // Rẽ sau khi thấy biển 1 thời gian
    traffic_sign_passed_time_lower_bound =  config->get<int>("traffic_sign_passed_time_lower_bound");; // Cận dưới của thời gian bắt đầu rẽ.
    traffic_sign_passed_time_higher_bound =  config->get<int>("traffic_sign_passed_time_higher_bound");; // Cận trên của thời gian bắt đầu rẽ.

    min_traffic_sign_bound_area = config->get<int>("min_traffic_sign_bound_area");
    traffic_sign_valid_duration = config->get<int>("traffic_sign_valid_duration");
    speed_on_preparing_to_turn_trafficsign = config->get<float>("speed_on_preparing_to_turn_trafficsign");
    lane_area_to_turn = config->get<int>("lane_area_to_turn");
    turning_angle_on_trafficsign = config->get<float>("turning_angle_on_trafficsign");
    speed_on_turning_trafficsign = config->get<float>("speed_on_turning_trafficsign");
    turning_duration_trafficsign = config->get<int>("turning_duration_trafficsign");
    line_diff_effect_speed_coeff = config->get<float>("line_diff_effect_speed_coeff");


    // # Quick start
    // #   We assume that at the begining, the road is straight so we increase the speed as much as possible
    quick_start = config->get<bool>("quick_start");
    quick_start_speed = config->get<float>("quick_start_speed");
    quick_start_time = config->get<Timer::time_duration_t>("quick_start_time");


    middle_point_adjustment = config->get<int>("middle_point_adjustment");


    // SpeedUp!!!
    duration_speedup_after_traffic_sign_1 = config->get<int>("duration_speedup_after_traffic_sign_1");
    speed_on_speedup_after_traffic_sign_1 = config->get<int>("speed_on_speedup_after_traffic_sign_1");
    duration_speedup_after_traffic_sign_2 = config->get<int>("duration_speedup_after_traffic_sign_2");
    speed_on_speedup_after_traffic_sign_2 = config->get<int>("speed_on_speedup_after_traffic_sign_2");
    

   
}


void CarControl::resetRound() {
    round_start_time = Timer::getCurrentTime();
}


void CarControl::publishSignal(float speed_data, float angle_data) {

    if (Timer::calcTimePassed(last_signal_publish_time_point) > signal_publish_interval) {
        std_msgs::Float32 angle;
        std_msgs::Float32 speed;

        angle.data = angle_data;
        speed.data = speed_data;

        // Fit to value ranges
        if (angle_data > MAX_ANGLE) angle_data = MAX_ANGLE;
        if (angle_data < -MAX_ANGLE) angle_data = -MAX_ANGLE;
        if (speed_data > MAX_SPEED) speed_data = MAX_SPEED;
        if (speed_data < MIN_SPEED) speed_data = MIN_SPEED;

        // Publish message
        if (debug_flag) {
            ROS_INFO_STREAM("SPEED: " << speed_data);
            ROS_INFO_STREAM("ANGLE: " << angle_data);
        }

        last_speed_data = speed_data;
        last_angle_data = angle_data;

        // Control quick start: Overwrite speed on quick start
        if (quick_start && Timer::calcTimePassed(round_start_time) < quick_start_time) {

            if (debug_flag)
                std::cout << "Quick Starting at: " << Timer::calcTimePassed(round_start_time) << std::endl;

            speed.data = quick_start_speed;
            
        }


        // Control speed up
        if (num_of_crossed_trafficsign == 1 && Timer::calcTimePassed(crossed_traffic_sign_1_time_point) < duration_speedup_after_traffic_sign_1) {
            speed.data = speed_on_speedup_after_traffic_sign_1;
        } else if (num_of_crossed_trafficsign == 2 && Timer::calcTimePassed(crossed_traffic_sign_2_time_point) < duration_speedup_after_traffic_sign_2) {
            speed.data = speed_on_speedup_after_traffic_sign_2;
        }


        steer_publisher.publish(angle);
        speed_publisher.publish(speed);

        last_signal_publish_time_point = Timer::getCurrentTime();
    }
    
}


void CarControl::driverCar(float speed_data, float angle_data) {
    publishSignal(speed_data, angle_data);
}


void CarControl::driverCar(Road & road, const std::vector<TrafficSign> & traffic_signs) {

    //  STEP 1: FIND THE CAR POSITION

    float angle_data;
    float speed_data;

    // Do nothing when we find a bad result from lane detector
    if (road.middle_points.size() >= min_num_of_middle_points) {

        // Sort the middle points asc based on y
        std::sort(std::begin(road.middle_points), std::end(road.middle_points),
                [] (const cv::Point& lhs, const cv::Point& rhs) {
            return lhs.y < rhs.y;
        });

        // Choose an interested point (point having y = 60% ymax)
        int index_of_interested_point = static_cast<int>(road.middle_points.size() * middle_interested_point_pos);


        cv::Point middle_point = road.middle_points[index_of_interested_point];
        
        if (debug_flag) {
            std::cout << middle_point << std::endl;
        }
        
        cv::Point center_point = cv::Point(Road::road_center_line_x, middle_point.y);


        //  STEP 3: FIND THE BASE CONTROLLING PARAMS ( BASED ON LANE LINES )
        speed_data = MAX_SPEED;
        float delta = center_point.x - middle_point.x - middle_point_adjustment;


        // line_diff_to_angle_coeff
        float line_diff = 0;
        if (index_of_interested_point > 12) {
            float diff1 = road.middle_points[index_of_interested_point].x -  road.middle_points[index_of_interested_point - 3].x;
            float diff2 = road.middle_points[index_of_interested_point - 3].x -  road.middle_points[index_of_interested_point - 6].x;
            if (abs(diff1 - diff2) < 20) {
                line_diff = diff1;
            }
            // std::cout << "diff1: " << diff1 << std::endl;
            // std::cout << "diff2: " << diff2 << std::endl;
        }

        angle_data = delta * delta_to_angle_coeff + line_diff * line_diff_to_angle_coeff;

        speed_data -= abs(line_diff) * line_diff_effect_speed_coeff;

    }
    else {
        speed_data = last_speed_data;
        angle_data = last_angle_data;
    }


    //  STEP 4: ADJUST CONTROLLING PARAMS USING TRAFFIC SIGN DETECTOR
    if (!traffic_signs.empty() && !is_turning) {

        if (debug_flag) {
            ROS_INFO_STREAM("TRAFFIC SIGN DETECTED!: " << "Number: " << traffic_signs.size());
            for (int i = 0; i < traffic_signs.size(); ++i) {
                std::cout << traffic_signs[i].id << " : " << traffic_signs[i].rect << std::endl;
            }
        }

        if (traffic_signs[0].rect.area() > min_traffic_sign_bound_area
            && (
                traffic_signs[0].id == TrafficSign::SignType::TURN_LEFT
                || traffic_signs[0].id == TrafficSign::SignType::TURN_RIGHT
            )
        ) {

            prepare_to_turn = true;
            last_sign_id = traffic_signs[0].id;
            last_sign_time_point = Timer::getCurrentTime();

        }
    }


    // Giảm tốc khi đến khúc cua
    if (prepare_to_turn && Timer::calcTimePassed(last_sign_time_point) < traffic_sign_valid_duration) {
        speed_data = speed_on_preparing_to_turn_trafficsign;
    }

    // Khi nhận thấy đang có tín hiệu rẽ,
    // và diện tích đường mở rộng (đã đến ngã ba, ngã tư) thì thực hiện rẽ
    if ( prepare_to_turn && Timer::calcTimePassed(last_sign_time_point) < traffic_sign_valid_duration // Biển báo còn trong thời gian valid
    ) {
        // Check nếu xe đã chạy đến ngã ba, ngã tư (TH rẽ ở ngã ba, ngã tư)
        // hoặc xe đã chạy đến thời gian phải rẽ (rẽ sau biển 1 khoảng thời gian)
        Timer::time_duration_t traffic_sign_passed_time =  Timer::calcTimePassed(last_sign_time_point);
        if (
            (
                turn_on_trafficsign_by_lane_area // Rẽ theo diện tích đường
                && road.lane_area > lane_area_to_turn
            ) || (
                turn_on_trafficsign_by_passed_time // Rẽ theo thời gian biển đã qua
                && traffic_sign_passed_time > traffic_sign_passed_time_lower_bound
                && traffic_sign_passed_time < traffic_sign_passed_time_higher_bound
            )
        ) {
            prepare_to_turn = false;
            if (debug_flag) ROS_INFO_STREAM("TURNING: " << last_sign_id);

            std::cout << "TURNING: " << last_sign_id << std::endl;

            if (last_sign_id == TrafficSign::SignType::TURN_LEFT) {
                turning_coeff = -turning_angle_on_trafficsign;
            } else if (last_sign_id == TrafficSign::SignType::TURN_RIGHT) {
                turning_coeff = +turning_angle_on_trafficsign;
            }
            
            speed_data = speed_on_turning_trafficsign;

            turning_time_point = std::chrono::system_clock::now();
            is_turning = true;
        }

    }


    if (debug_flag) {
        ROS_INFO_STREAM("turning_coeff: " << turning_coeff);
    }


    // Triggle turning to bring car back to normal state
    if (Timer::calcTimePassed(turning_time_point) > turning_duration_trafficsign && is_turning) {
        turning_coeff = 0;
        is_turning = false;
        ++num_of_crossed_trafficsign;

        // Save time point of crossed traffic signs for speedup
        if (num_of_crossed_trafficsign == 1)
            crossed_traffic_sign_1_time_point = Timer::getCurrentTime();
        else if (num_of_crossed_trafficsign == 2)
            crossed_traffic_sign_2_time_point = Timer::getCurrentTime();
    }

    // if (num_of_crossed_trafficsign > 0) {
    //     lane_area_to_turn = 17000;
    // }

    if (is_turning) {
        speed_data = speed_on_turning_trafficsign;
    }
    


    // STEP 5: FIND AND AVOID OBSTACLE
    
    // Print line diff
    if (debug_flag) {
        ObstacleDetector::printLineDiff(road.middle_points);
    }
    

    if (debug_flag) {
        std::cout << "LAST OBSTACLE TIME: " << Timer::calcTimePassed(obstacle_avoiding_time_point) << std::endl;
        std::cout << "OBSTACLE COEFF: " << obstacle_avoid_coeff << std::endl;
    }
    

    //Find the obstacle and adjust obstacle_avoid_coeff
    // for (int i = road.middle_points.size()-5; i >=  10; --i) {
    //     int diff = road.middle_points[i+3].x - road.middle_points[i].x;
    //     int distance_to_obstacle;

    //     if (abs(diff) > 8 and abs(diff) < 20) {
    //         distance_to_obstacle = Road::road_center_line_x - road.middle_points[i].y;
            
    //         if (debug_flag) {
    //             std::cout << "OBSTACLE DISTANCE: " << distance_to_obstacle << std::endl;
    //         }

    //         obstacle_avoiding_time_point = std::chrono::high_resolution_clock::now();
        
    //         if (diff > 0) {
    //             obstacle_avoid_coeff = -10;
    //             break;
    //         } else if (diff < 0) {
    //             obstacle_avoid_coeff = +10;
    //             break;
    //         }
        
    //     }

    // }
    

    angle_data += obstacle_avoid_coeff + turning_coeff;

    if (turning_coeff != 0) {
        angle_data = turning_coeff;
    }

    if (debug_flag) ROS_INFO_STREAM("lane_area: " << road.lane_area);


    // STEP 6: FINAL ADJUSTMENT AND PUBLISH

    publishSignal(speed_data, angle_data);

}

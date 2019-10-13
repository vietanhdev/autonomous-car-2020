#ifndef CONFIG_H
#define CONFIG_H

#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include "yaml-cpp/yaml.h"
#include <ros/ros.h>
#include <ros/console.h>


class Config {

    public:


    std::string path;
    YAML::Node config;

    // Instance for default config file
    static std::shared_ptr<Config> default_instance;

    static std::string racecar_pkg_name;
    

    Config() {

        std::string path = ros::package::getPath(getROSPackage()) + std::string("/data/config.yaml");

        this->path = path;

        // Read config file
        config = YAML::LoadFile(path);

    }

    Config(std::string config_file_name) {

        std::string path = ros::package::getPath(getROSPackage()) + std::string("/data/" + config_file_name);

        this->path = path;

        // Read config file
        config = YAML::LoadFile(path);

    }


    public:

    Config(Config * c) {
        this->path = c->path;
        this->config = c->config;
    }

    static std::shared_ptr<Config> getDefaultConfigInstance() {
        if (Config::default_instance == nullptr) {
            Config::default_instance = std::make_shared<Config>(new Config());
        }
        return Config::default_instance;
    }

    static std::shared_ptr<Config> getNewConfigInstance(std::string config_file) {
        return std::make_shared<Config>(new Config(config_file));
    }

    // Copy constructor 
    Config(const Config &c) {
        config = YAML::Clone(config);
    }


    static const std::string getDataFile(const std::string & filename) {
        return  ros::package::getPath(getROSPackage()) + std::string("/data/" + filename);
    }

    
    static std::string getROSPackage() {
        if (racecar_pkg_name.empty()) {
            ros::NodeHandle private_node_handle("~");
            if (!private_node_handle.getParam("racecar_pkg_name", racecar_pkg_name)) {
                racecar_pkg_name = "team806";
            }
        }
        return racecar_pkg_name;
    }

    std::string getTeamName() {
        return get<std::string>("team_name");
    }

    template <class T>
    T get(std::string key) {
        return config[key].as<T>();
    }

    cv::Point getPoint(std::string key) {
        std::string point_str = config[key].as<std::string>();
        std::vector<int> nums =  extractIntegers(point_str);

        ROS_ASSERT_MSG(nums.size() == 2, "Error  on reading %s", key.c_str());

        return cv::Point(nums[0], nums[1]);
    }

    cv::Size getSize(std::string key) {
        std::string size_str = config[key].as<std::string>();
        std::vector<int> nums =  extractIntegers(size_str);

        ROS_ASSERT_MSG(nums.size() == 2, "Error  on reading %s", key.c_str());

        return cv::Size(nums[0], nums[1]);
    }

    cv::Scalar getScalar3(std::string key) {
        std::string size_str = config[key].as<std::string>();
        std::vector<int> nums =  extractIntegers(size_str);

        ROS_ASSERT_MSG(nums.size() == 3, "Error  on reading %s", key.c_str());

        return cv::Scalar(nums[0], nums[1], nums[2]);
    }

    static std::vector<int> extractIntegers(std::string str)  { 
        std::stringstream ss;  
        std::vector<int> results;
    
        /* Storing the whole string into string stream */
        ss << str; 
    
        /* Running loop till the end of the stream */
        std::string temp; 
        int found; 
        while (!ss.eof()) { 
    
            /* extracting word by word from stream */
            ss >> temp; 
    
            /* Checking the given word is integer or not */
            if (std::stringstream(temp) >> found) 
                results.push_back(found); 
    
            /* To save from space at the end of string */
            temp = ""; 
        }

        return results;
    } 


};

#endif
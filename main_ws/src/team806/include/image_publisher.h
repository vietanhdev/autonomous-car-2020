#ifndef IMAGE_PUBLISHER__
#define IMAGE_PUBLISHER__
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImagePublisher {
    std::shared_ptr<ros::NodeHandle> nh_;
    std::shared_ptr<image_transport::ImageTransport> it_;

    public:

    ImagePublisher() {
        nh_ = std::shared_ptr<ros::NodeHandle>(new ros::NodeHandle("~"));
        it_ = std::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(*nh_));
    }

    image_transport::Publisher createImagePublisher(std::string topic_name, int queue_size) {
        return it_->advertise(topic_name, queue_size);
    }

    cv_bridge::CvImagePtr getImageMsgPtr(const cv::Mat & img) {
       
        cv::Mat bgr_img;
        if (img.channels() == 1) {
            cvtColor(img, bgr_img, cv::COLOR_GRAY2BGR);
        } else {
            bgr_img = img.clone();
        }
       
        cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
        ros::Time time = ros::Time::now();
        cv_ptr->encoding = "bgr8";
        cv_ptr->header.stamp = time;
        // cv_ptr->header.frame_id = "/hello_word";
        cv_ptr->image = bgr_img;
        
        return cv_ptr;
    }

    void publishImage(image_transport::Publisher publisher, const cv::Mat & img) {
        publisher.publish(getImageMsgPtr(img)->toImageMsg());
    }

};


#endif
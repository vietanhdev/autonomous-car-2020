#include <iostream>
#include <thread>         // std::thread
#include <mutex>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>

#include "car_control.h"

#include <unistd.h>   //_getch*/
#include <termios.h>  //_getch*/
#include <unistd.h>


using namespace std;


using namespace std;
using namespace cv;

unsigned int microseconds;

CarControl *car;
float speed(0);
float angle(0);
std::mutex control_mutex;


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{

    cv_bridge::CvImagePtr cv_ptr;
    Mat out;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}


char getch(){
    /*#include <unistd.h>   //_getch*/
    /*#include <termios.h>  //_getch*/
    char buf=0;
    struct termios old={0};
    fflush(stdout);
    if(tcgetattr(0, &old)<0)
        perror("tcsetattr()");
    old.c_lflag&=~ICANON;
    old.c_lflag&=~ECHO;
    old.c_cc[VMIN]=1;
    old.c_cc[VTIME]=0;
    if(tcsetattr(0, TCSANOW, &old)<0)
        perror("tcsetattr ICANON");
    if(read(0,&buf,1)<0)
        perror("read()");
    old.c_lflag|=ICANON;
    old.c_lflag|=ECHO;
    if(tcsetattr(0, TCSADRAIN, &old)<0)
        perror ("tcsetattr ~ICANON");
    // printf("%c\n",buf);
    return buf;
 }


void control_receiver() {
    while (true) {
        char c = getch();
        if (c == '\033') { // if the first value is esc
            getch(); // skip the [
            // control_mutex.lock();
            switch(getch()) { // the real value
                case 'A':
                    // code for arrow up
                    // cout << "up" << endl;
                    if (speed < 100) speed += 10;
                    break;
                case 'B':
                    // code for arrow down
                    // cout << "down" << endl;
                    speed = 0;
                    break;
                case 'C':
                    // code for arrow right
                    // cout << "right" << endl;
                    if (angle < 0) angle = 0;
                    else if (angle < 100) angle += 10;
                    break;
                case 'D':
                    // code for arrow left
                    // cout << "left" << endl;
                    if (angle > 0) angle = 0;
                    else if (angle > -100) angle -= 10;
                    break;
            }
            // control_mutex.unlock();
        } else if (c = 'q') {
            exit(0);
        }

    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener_manual_mode");

    car = new CarControl();

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("Team1_image", 1, imageCallback);

    cout << "Manual Mode ..." << endl;

    std::thread t1(control_receiver);

    while (true) {

        // control_mutex.lock();
        car->driverCar(speed, angle);
        cout << "Speed: " << speed << " - Angle: " << angle << "\n";
        // if (speed > 0) --speed;
        // if (angle > 0) --angle;
        // if (angle < 0) ++angle;
        // control_mutex.unlock();

        usleep(20000);

        ros::spinOnce();
    }

    cv::destroyAllWindows();

    return 0;
}
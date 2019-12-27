FROM dungpb/dira_ros:ros-cuda


RUN echo "deb http://archive.ubuntu.com/ubuntu bionic main universe" >> /etc/apt/sources.list
RUN echo "deb http://archive.ubuntu.com/ubuntu bionic-security main universe" >> /etc/apt/sources.list
RUN echo "deb http://archive.ubuntu.com/ubuntu bionic-updates main universe " >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y python-pip
RUN pip install tensorflow-gpu==2.0.0b1 opencv-contrib-python rospkg tqdm simple_pid easydict
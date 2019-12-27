FROM dungpb/dira_ros:ros-cuda


RUN echo "deb http://archive.ubuntu.com/ubuntu bionic main universe" >> /etc/apt/sources.list
RUN echo "deb http://archive.ubuntu.com/ubuntu bionic-security main universe" >> /etc/apt/sources.list
RUN echo "deb http://archive.ubuntu.com/ubuntu bionic-updates main universe " >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y python-pip
RUN pip install rospkg
RUN pip install opencv-python==3.4.1.15 tqdm==4.41.0 simple_pid==0.2.4 easydict==1.9
RUN pip install tensorflow-gpu==2.0.0b1 
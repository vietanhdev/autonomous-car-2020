# RaceCar 2019-2020 - Team: ICT K60

## I. Project Info:

- Team name: teamict

## II. Environment Setup

### 1. Ubuntu 18.04

### 2. Robot Operating System
  
- Install ROS Melodic:
  http://wiki.ros.org/melodic/Installation/Ubuntu

- Using full desktop version.
  
### 3. Initialize Catkin workspace

We need to initialize catkin workspace at the first time (build folders for projects).

```terminal
cd main_ws
catkin_make
```

Add workspace to PATH:

```terminal
echo "source !(pwd)/main_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
  
#### 4. Dependencies: 

- rosbridge-suite

```terminal
sudo apt-get install ros-melodic-rosbridge-server
```
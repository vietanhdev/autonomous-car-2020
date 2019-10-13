# RaceCar 2019-2020 - Team: ICT K60

## I. Project Info:

- Team name: teamict

## II. Environment Setup

### 1. Ubuntu 18.04

### 2. Miniconda or Anaconda

### 3. OpenCV 3.4.3

- You can use following script to install OpenCV 3.4.3: <scripts/install_opencv_3.sh>

### 4. Robot Operating System
  
- Install ROS Melodic - full desktop version: <http://wiki.ros.org/melodic/Installation/Ubuntu>
  
### 5. Initialize Catkin workspace

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

### 6. Create Conda environment 

```terminal
cd main_ws
conda env create -n cds -f environment.yml 
conda activate cds
```
  
### 7. Dependencies: 

- rosbridge-suite

```terminal
sudo apt-get install ros-melodic-rosbridge-server
```

### Other stuffs

https://medium.com/@zuxinl/ubuntu-18-04-ros-python3-anaconda-cuda-environment-configuration-cb8c8e42c68d
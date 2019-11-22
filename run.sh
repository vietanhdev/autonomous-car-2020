#!/bin/bash
sudo docker build --tag=teamict_env .

# sudo docker run --rm -it --gpus=all --network=host --memory=6000M -d -v main_ws/src/:/root/catkin_ws/src/ --name teamict teamict_env bash

sudo docker run -it --gpus=all --network=host --memory=6000M -d -v main_ws/src/:/root/catkin_ws/src/ --name teamict teamict_env bash

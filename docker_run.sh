#!/bin/bash
sudo docker stop teamict
sudo docker rm teamict
sudo docker run -it --gpus=all --network=host --memory=6000M -d -v /mnt/DATA/CUOC_DUA_SO/sources/SuperCar/main_ws/src/:/root/catkin_ws/src/ --name teamict teamict_env bash
sudo docker run --rm -it --gpus=all --network=host --memory=6000M -v /mnt/DATA/CUOC_DUA_SO/sources/SuperCar/main_ws/src/:/root/catkin_ws/src/ teamict /bin/bash
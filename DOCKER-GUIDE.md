# Guide to Docker

```
sudo docker build -t team613 .
sudo docker run --rm -it --network=host --gpus=all -v /mnt/DATA/CUOC_DUA_SO/sources/SuperCar/main_ws/src/team613:/catkin_ws/src/team613 --name team613 team613 bash
```

```
sudo docker build -t team613 .
sudo docker run --rm -it --network=host --gpus=all -v /home/an/catkin_ws/racecar/main_ws/src/team613 --name team613 team613 bash
```
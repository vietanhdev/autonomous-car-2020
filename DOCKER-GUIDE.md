# Guide to Docker

```
sudo docker build -t teamict .
sudo docker run --rm -it --network=host -v /mnt/DATA/CUOC_DUA_SO/sources/SuperCar/main_ws/src/teamict:/catkin_ws/src/teamict --name teamict teamict bash
```
FROM diradocker/dira_env:first_stable

RUN cd /root/catkin_ws
RUN catkin_make

EXPOSE 9090


ENV NAME World

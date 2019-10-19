# -*- coding: utf-8 -*- 

TEAM_NAME = 'teamict'
PACKAGE_NAME = 'teamict'

# /tênđội/set_speed: Topic được publish từ ROS_node được định nghĩa dưới dạng số thực (Float32). Là tốc độ xe cần đạt. ( Mặc định đang để là /team1/set_speed, nếu nhập tên đội khác, cần sửa lại topic trong code /lane_detect/src/main.cpp theo tên đội đã nhập ở app)
TOPIC_SET_SPEED = '/{}/set_speed'.format(TEAM_NAME)

# /tênđội/set_angle: Topic được publish từ ROS_node định nghĩa dưới dạng số thực (Float32). Truyền góc lái của xe. ( Mặc định đang để là /team1/set_angle, nếu nhập tên đội khác, cần sửa lại topic trong code /lane_detect/src/carcontrol.cpp theo tên đội đã nhập ở app)
TOPIC_SET_ANGLE = '/{}/set_angle'.format(TEAM_NAME)

# /tênđội/set_camera_angle: Topic được publish từ ROS_node định nghĩa dưới dạng số thực (Float32). Truyền quay của camera.
TOPIC_SET_CAMERA_ANGLE = '/{}/set_camera_angle'.format(TEAM_NAME)

# /tênđội/camera/rgb/compressed: Topic dùng để subcribe ảnh rgb thu được trên xe. Ảnh thu được là ảnh nén theo chuẩn “img”.( Mặc định đang để là /team1/camera/rgb/compressed, nếu nhập tên đội khác, cần sửa lại topic trong code /lane_detect/src/carcontrol.cpp theo tên đội đã nhập ở app)
TOPIC_GET_IMAGE = '/{}/camera/rgb/compressed'.format(TEAM_NAME)

# /tênđội/camera/depth/compressed: Topic dùng để subcribe ảnh depth thu được trên xe. Ảnh thu được là ảnh nén theo chuẩn “img”.( Mặc định đang để là /team1/camera/depth/compressed, nếu nhập tên đội khác, cần sửa lại topic trong code /lane_detect/src/carcontrol.cpp theo tên đội đã nhập ở app)
TOPIC_GET_DEPTH_IMAGE = '/{}/camera/depth/compressed'.format(TEAM_NAME)
import threading
from sensor_msgs.msg import CompressedImage, Image
import rospy
import time
from cv_bridge import CvBridge, CvBridgeError

class ImageStream():
    '''
    Publishing stream for an image
    Protected by mutex
    '''

    # image_size in the  (height, width, channels) or (height, width)
    def __init__(self, publish_topic):
        self.image = None
        self.mutex = threading.Lock()
        self.publisher = rospy.Publisher(publish_topic, Image, queue_size=1)

    def update_image(self, image):
        self.mutex.acquire()
        self.image = image.copy()
        self.mutex.release()

    def get_image(self):
        if self.image is None:
            return None
        self.mutex.acquire()
        image = self.image.copy()
        self.mutex.release()
        return image

    def publish(self, bridge):
        if self.image is not None:
            try:
                if len(self.image.shape) == 2:
                    self.publisher.publish(bridge.cv2_to_imgmsg(self.get_image(), "mono8"))
                else:
                    self.publisher.publish(bridge.cv2_to_imgmsg(self.get_image(), "bgr8"))
            except CvBridgeError as e:
                print(e)



class DebugStream(threading.Thread):
    '''
    Thread to publish debug images
    '''

    def __init__(self, image_bridge):
        self.image_streams = {}
        self.image_bridge = image_bridge
        threading.Thread.__init__(self)

    def run(self):
        while not rospy.is_shutdown():
            rospy.Rate(1).sleep()
            for (stream_name, stream) in self.image_streams.items():
                stream.publish(self.image_bridge)

    def create_stream(self, stream_name, publish_topic):
        self.image_streams[stream_name] = ImageStream(publish_topic)

    def update_image(self, stream_name, image):
        # Only update image if stream exist
        if stream_name in self.image_streams.keys():
            self.image_streams[stream_name].update_image(image)
        
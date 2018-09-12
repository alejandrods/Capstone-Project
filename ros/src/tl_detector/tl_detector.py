#!/usr/bin/env python
import rospy
import numpy as np
from scipy.spatial import KDTree
from std_msgs.msg import Int32, UInt8
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3
BRAKING_DISTANCE_IN_NUM_WAYPOINTS = 100

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector',log_level=rospy.DEBUG)

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.car_wp_idx = None
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.camera_subsample_factor = 6
        self.camera_frame_counter = 0
        self.last_wp = -1
        self.state_count = 0
        self.last_known_line_wp_idx = -1
        self.last_known_state = TrafficLight.UNKNOWN
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.is_site = self.config['is_site']

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.upcoming_light_pub_for_display = rospy.Publisher('/traffic_waypoint_for_display', Int32, queue_size=1)        
        self.camera_image_publisher = rospy.Publisher('/image_debug', Image, queue_size=1)
        self.traffic_signal_color_pub = rospy.Publisher('/signal_color', UInt8, queue_size=1)

        self.light_classifier = TLClassifier(self.is_site)
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
            # TODO: Implement
            self.waypoints = waypoints
            if self.waypoints_2d is None:
                self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
                self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        self.camera_frame_counter = ((self.camera_frame_counter + 1) % self.camera_subsample_factor)
        
        if self.camera_frame_counter == 0 and \
            self.waypoint_tree is not None:
            light_wp, state = self.process_traffic_lights()
        else:
            light_wp = self.last_known_line_wp_idx
            state = self.last_known_state
        self.upcoming_light_pub_for_display.publish(Int32(light_wp))
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            if (state == TrafficLight.GREEN):
                light_wp = -1
            elif ((state == TrafficLight.YELLOW) and \
                  (light_wp - self.car_wp_idx > BRAKING_DISTANCE_IN_NUM_WAYPOINTS)):
                light_wp = -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x,y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #return light.state
        if(not self.has_image):
            self.prev_light_loc = None
            return light.state

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8") #bgr8
        
        #Get classification
        signal, cv_image_debug = self.light_classifier.get_classification(cv_image)
        if signal == TrafficLight.GREEN or signal == TrafficLight.RED or signal == TrafficLight.YELLOW:
	    resized = cv2.resize(cv_image_debug, (400,300,), interpolation=cv2.INTER_LINEAR)
	    image_message = self.bridge.cv2_to_imgmsg(resized, encoding="rgb8")
	    self.camera_image_publisher.publish(image_message)
        self.traffic_signal_color_pub.publish(signal)
        return signal

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            self.car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint_index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - self.car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

            if closest_light:
                state = self.get_light_state(closest_light)
                self.last_known_line_wp_idx = line_wp_idx
                self.last_known_state = state
                return line_wp_idx, state

 	if self.is_site:
            self.last_known_state =  self.get_light_state(0)
            self.last_known_line_wp_idx = -1
        else:
            self.last_known_state =  TrafficLight.UNKNOWN
            self.last_known_line_wp_idx = -1
        return self.last_known_line_wp_idx, self.last_known_state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

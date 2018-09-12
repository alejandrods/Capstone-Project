#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

MAX_DECEL = .5 # Max deceleraton TBD

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater',log_level=rospy.DEBUG)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.distance_to_stopline_pub = rospy.Publisher('/distance_to_stopline', Int32, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.lookahead_wps = None
        self.lookahead_wps_for_traffic_signal = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/lookahead_wps', Int32, self.lookahead_wps_cb)
        rospy.Subscriber('/lookahead_wps_for_traffic_signal', Int32, self.lookahead_wps_for_traffic_signal_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.loop()

    def loop(self):
       rate = rospy.Rate(25)
       while not rospy.is_shutdown():
          if self.pose is not None and self.waypoint_tree is not None and self.lookahead_wps_for_traffic_signal is not None:
             self.publish_waypoints()
             rate.sleep()

    def get_closest_waypoint_idx(self):
       x = self.pose.pose.position.x
       y = self.pose.pose.position.y
       closest_idx = self.waypoint_tree.query([x,y],1)[1]
       
       # Check if closest is ahead or behind vehicle
       closest_coord = self.waypoints_2d[closest_idx]
       if (closest_idx == 0):
          prev_coord = self.waypoints_2d[len(self.waypoints_2d) - 1]
       else:
          prev_coord = self.waypoints_2d[closest_idx - 1]
       
       #Equation for hyperplane through closest_coords
       cl_vect = np.array(closest_coord)
       prev_vect = np.array(prev_coord)
       pos_vect = np.array([x,y])
       
       val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
       
       if val > 0:
          closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
       
       return closest_idx
    
    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
        
    def generate_lane(self):
        lane = Lane()
        # Get closest waypoint
        closest_idx = self.get_closest_waypoint_idx()
        if closest_idx + self.lookahead_wps - 1 < len(self.waypoints_2d):
           farthest_idx = closest_idx + self.lookahead_wps - 1
           base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx+1]
        else:
           base_waypoints = self.base_waypoints.waypoints[closest_idx:len(self.waypoints_2d)]
           farthest_idx = self.lookahead_wps - (len(self.waypoints_2d) - closest_idx) - 1
           base_waypoints.extend(self.base_waypoints.waypoints[0:farthest_idx+1])

        if closest_idx + self.lookahead_wps_for_traffic_signal <= len(self.waypoints_2d):
           farthest_idx_for_traffic_signal_detection =  closest_idx + self.lookahead_wps_for_traffic_signal
           wrap_around = False
        else:
           farthest_idx_for_traffic_signal_detection =  self.lookahead_wps_for_traffic_signal - (len(self.waypoints_2d) - closest_idx)
           wrap_around = True

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx_for_traffic_signal_detection) or \
           (wrap_around == False and self.stopline_wp_idx < self.lookahead_wps_for_traffic_signal):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, self.base_waypoints.waypoints, closest_idx)
        return lane

    def decelerate_waypoints(self, waypoints_partial, waypoints_full, closest_idx):
        temp = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - 4, 0) # Two waypoints back from line so front of the car stops at given line
        self.distance_to_stopline_pub.publish(stop_idx)
        for i, wp in enumerate(waypoints_partial):
            p = Waypoint()
            p.pose = wp.pose
            dist = self.distance(waypoints_full, closest_idx + i, self.stopline_wp_idx - 4)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            #vel = MAX_DECEL * dist
            if vel < 1.:
                vel = 0. 
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x) # Make sure the top speed is limited
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints
        if self.waypoints_2d is None:
           self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
           self.waypoint_tree = KDTree(self.waypoints_2d)
        
    def lookahead_wps_cb(self, msg):
        # Number of waypoints we will publish. You can change this number
        self.lookahead_wps = msg.data

    def lookahead_wps_for_traffic_signal_cb(self, msg):
        # Number of waypoints ahead we will inspect for traffic signals (not planning)
        self.lookahead_wps_for_traffic_signal = msg.data

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

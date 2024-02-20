#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import csv
import sensor_msgs.point_cloud2 as pc2
import laser_geometry.laser_geometry as lg
import matplotlib.pyplot as plt
import sys
import rosbag
import os

lp = lg.LaserProjection()

def scan2csv(rosbagFile, outFile):

    rospy.loginfo("Opening bag file: " + rosbagFile)
    try:
        bag = rosbag.Bag(rosbagFile)
    except:
        rospy.logerr("Error opening specified bag file : %s"%rosbagFile)
        sys.exit(2)

    rospy.loginfo ("Bag file opened.")

    rospy.loginfo("Opening " + outFile + " for writing..")
    try:
        fileH = open(outFile, 'wt')
        writer = csv.writer(fileH)
    except:
        rospy.logerr("Error opening specified output file : %s"%outFile)
        sys.exit(2)
    
    rospy.loginfo ("Output file opened.")
    #get the directory if we need it
    outDir = os.path.dirname(outFile)

    i = 1
    for topic, msg, t in bag.read_messages(topics=['/scan']):

        # convert the laser scan to a pointcloud
        cloud = lp.projectLaser(msg)
        
        # convert cloud into a numpy array
        cloud = np.array(list(pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))))

        # Add iterator value as the first column
        iterator_value = np.full((cloud.shape[0], 1), i)
        cloud_with_iterator = np.hstack((iterator_value, cloud))

        # append cloud to CSV file
        if fileH.tell() == 0:  # Check if file is empty
            writer.writerow(['iterator', 'x', 'y', 'z'])  # Write header
        writer.writerows(cloud_with_iterator)
        i = i + 1

    bag.close()

def main():
    rospy.init_node('scan2csv')

    #default values
    rosbagFile = "/data/python_ws/bag_files/rosbag2_human_walking_around.bag"
    outFile = os.path.splitext(rosbagFile)[0] + ".csv"

    scan2csv(rosbagFile, outFile)

if __name__ == "__main__":
    main()


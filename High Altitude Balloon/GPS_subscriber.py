# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:44:29 2020

@author: Bryan
"""

import rospy
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt

def GPS_callback(data):
 
    # Plot the data is it comes in 
    plt.figure()
    plt.plot(data[:,0], label = 'GPS_x')
    plt.plot(data[:,1], label = 'GPS_z')
    plt.title('GPS readings vs. time')
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(loc="upper left")
    plt.pause(1)
    
def listener():

    # Initialize the subscriber node 
    rospy.init_node('listener', anonymous=True)
    
    # Set the topic to subscribe to 
    rospy.Subscriber("GPS_topic", Float64MultiArray, GPS_callback)

    rospy.spin()
    


if __name__ == '__main__':
    listener()
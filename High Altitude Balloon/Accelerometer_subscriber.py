# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:16:29 2020

@author: Bryan
"""

import rospy
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt

def Accel_callback(data):
 
    # plot the data as it comes in
    plt.figure()
    plt.plot(data[:,0], label = 'acceleration_x')
    plt.plot(data[:,1], label = 'acceleration_y')
    plt.title('Acceleration readings vs. time')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend(loc="upper left")
    plt.pause(0.025)
    
def listener():

    # initialize the subscriber node 
    rospy.init_node('listener', anonymous=True)

    # Set the topic the node is subscribing to
    rospy.Subscriber("Accel_topic", Float64MultiArray, Accel_callback)

    rospy.spin()
    

if __name__ == '__main__':
    listener()


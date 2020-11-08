# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:16:29 2020

@author: Bryan
"""

import rospy
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt

def Accel_callback(data):
 
    plt.figure()
    plt.plot(data[:,0], label = 'acceleration_x')
    plt.plot(data[:,1], label = 'acceleration_y')
    plt.title('Acceleration readings vs. time')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend(loc="upper left")
    plt.pause(0.025)
    
def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("GPS_topic", Float64MultiArray, Accel_callback)

    rospy.spin()
    

if __name__ == '__main__':
    listener()


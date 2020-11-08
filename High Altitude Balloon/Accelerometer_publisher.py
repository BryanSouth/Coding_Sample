# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:16:07 2020

@author: Bryan
"""

import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
import Get_Balloon_data
import numpy as np 

def Accel_node():
    
    pub = rospy.Publisher('Accel_topic', Float64MultiArray, queue_size=1)
    rospy.init_node('Accel_node')
    
    rate = rospy.Rate(40) # 40hz
    
    data = Float64MultiArray()
    data.layout.dim[0].label  = "height"
    data.layout.dim[0].size   = 1
    data.layout.dim[0].stride = 2*1= 2
    data.layout.dim[1].label  = "width"
    data.layout.dim[1].size   = 2
    data.layout.dim[1].stride = 2
    data.layout.data_offset = 0 
    data.data = np.zeros((1,2))

      
    while not rospy.is_shutdown():

      for time in range(0,279):
            
            vx_prev = 0
            vz_prev = 0
            ax_prev = 0
            az_prev = 0
            x_prev= 0
            z_prev = 0
            
            GPS_data = Get_Balloon_data.Balloon_sim(vx_prev, vz_prev, ax_prev, az_prev, x_prev, z_prev )
            
            vx_prev = GPS_data[2]
            vz_prev = GPS_data[3]
            ax_prev = GPS_data[4]
            az_prev = GPS_data[5]
            x_prev= GPS_data[0]
            z_prev = GPS_data[1]
            data.data[0,0] = ax_prev
            data.data[0,1] = az_prev
            
            pub.publish(data)
            rate.sleep()

if __name__ == '__main__':
    try:
        Accel_node()
    except rospy.ROSInterruptException:
        pass
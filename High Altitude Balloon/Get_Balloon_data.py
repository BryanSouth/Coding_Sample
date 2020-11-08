# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 01:54:20 2020

@author: Bryan
"""

import numpy as np
import matplotlib.pyplot as plt

height = []
drift = []
accel_x = []
accel_z = []

'''
This script is a demonstration simulation for the High Altitude Balloon project at Western University. It a step in testing the ROS sensor pipeline. 
It simulates the acceleration and position of a high altitude balloon given expected real world parameters and balloon and payload characteristics. 
This script creates the data used in the ROS accelerometer and GPS publisher and subscriber nodes. Guassian noise is applied to the sensor data based on sensor specs. 

'''

def Balloon_sim(v_x, v_z, a_x, a_z, x, z):
    
    m = 120
    C_d = 0.35
    A = 32
    p_a = 1.112
    g = 9.81
    
    
    F_b = 1500
    
    k = 0.5*A*p_a
    dt = 1
    
   
    if z <= 2000:

        a_z = (-m*g - k*C_d*v_z**2 + F_b)/m
        v_z = a_z*dt + v_z
        z = v_z*dt + z
    
    if z <= 1000:
        v_w = 5
        F_w = 0.5*A*p_a*v_w**2*C_d
        a_x = (k*C_d*v_x**2 - F_w )/m
        v_x = a_x*dt + v_x
        x = v_x*dt + x
                     
    if 1000 < z <= 2005:

        if v_x < 0:
                v_w = 5
                F_w = 0.5*A*p_a*v_w**2*C_d
                a_x = (F_w + k*C_d*v_x**2)/m
                v_x = a_x*dt + v_x
                x = v_x*dt + x  

        if v_x > 0:
                v_w = 5
                F_w = 0.5*A*p_a*v_w**2*C_d
                a_x = (F_w - k*C_d*v_x**2)/m
                v_x = a_x*dt + v_x
                x = v_x*dt + x
             
                
    process_noise = np.random.normal(0,5, 1)
    GPS_noise = np.random.normal(0,60, 1)
    accel_noise = np.random.normal(0, 0.1*9.8, 1)
    
    x = x + process_noise + GPS_noise
    z = z + process_noise + GPS_noise
    a_x = a_x + accel_noise
    a_z = a_z + accel_noise

    return x, z, v_x, v_z, a_x, a_z

if __name__ == '__main__':
        
    vx_prev = 0
    vz_prev = 0
    ax_prev = 0
    az_prev = 0
    x_prev= 0
    z_prev = 0

    for time in range(1,279):
        GPS_data = Balloon_sim(vx_prev, vz_prev, ax_prev, az_prev, x_prev, z_prev)
        vx_prev = GPS_data[2]
        vz_prev = GPS_data[3]
        ax_prev = GPS_data[4]
        az_prev = GPS_data[5]
        x_prev= GPS_data[0]
        z_prev = GPS_data[1]
        height.append(z_prev)
        drift.append(x_prev)
        accel_x.append(ax_prev)
        accel_z.append(az_prev)
        
    plt.figure()
    plt.plot(drift, label = 'GPS_x')
    plt.plot(height, label = 'GPS_z')
    plt.title('GPS readings vs. time')
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(loc="upper left")
    
    plt.figure()
    plt.plot(accel_x, label = 'acceleration_x')
    plt.plot(accel_z, label = 'acceleration_y')
    plt.title('Acceleration readings vs. time')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend(loc="upper left")

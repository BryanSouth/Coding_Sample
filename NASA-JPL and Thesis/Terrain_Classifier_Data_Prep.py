#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
# In[]
Created on Thu Jun  6 11:47:18 2019

@author: marchett

My colleage at JPL laid out the framework for this code, and I modified it heavily for my use case. 
Her name is kept as the author name for posterity's sake. The inclusion of this code is meant to show the handling and processing
of large quantities of data from multiple sensors, and to give some background insight into the problem at hand for other scripts
in this project.
"""
import glob
import numpy as np
import tools_pg as tools
import bf_globals


# In[]
datadir = '\\Users\\Bryan\\Documents\\Rover_Data\\'
regex = ['*']

# In[]
imu_contact_bin = []
sinks = []
pressure_new =[]

#initialize empty data bins
slips = []
torques = []
sigmas = []
label = []
sinks = []
sinks2 = []
loads = []
areas = []

#terrain type data folders to include 
terrain_types = ['BST110', 'WED730', 'GRC-01', '2MM', 'MMINTR', 'MMCRSE']

pntr = 0
aTaxel = 0.016**2 # Area of 1 taxel (m)
r = 0.27 # Wheel radius (m)
v2c = 5/1024 # Conversion of motor current sensor output (voltage) to current in amps
eta = 0.70 # transmission efficiency
kappa = 0.213 # torque constant Nm/A
zeta = 195.26 # gear ratio

for b in range(len(terrain_types)):
    '''
    Get all the rover sensor data (sigmas: pressure from the pressure pad, torques: motor torque, 
    slips: wheel slip from string potentiometer) for the specified terrain types
    
    '''
    print(pntr)
    subfolders = [terrain_types[b]]
    files = np.hstack([glob.glob(datadir + a + '\\' + b + '\\') for a, b in zip(subfolders, regex)])
    for j in range(len(files)):
    
        experiment = files[j]
        ftype, data, time = tools.readAlldata(datadir, experiment)
        fdate = experiment.split('_')[-3]
        data_binned, time_binned = tools.alignPGtoIMU(data, time, bf_globals.T, bf_globals.R)
        
        #remove non-stationary 
        rot_mask = tools.rotStartStopMotor(data_binned, graphs = False)
        time_binned = time_binned[~rot_mask]
        
        for k in data_binned.keys():
        
            data_binned[k] = data_binned[k][~rot_mask]
        
        imu_contact_bin = tools.contactIMU(data_binned['imu'])
        
        #contact area 
        contact_data = tools.contactAreaRun(data_binned['pg'], imu_contact_bin)
        #sinkage
        sink = tools.sinkagePG(data_binned, contact_data, imu_contact_bin = imu_contact_bin, return_theta = True)
        
        theta_r = sink[2]
        theta_f = sink[3]
        thetas = [theta_r, theta_f]
        np.save('thetas', thetas)
        
        # calibration curves for pressure pad data
        slopeCal = np.load('slopeCal.npy')
        intCal = np.load('intCal.npy')
        
        ambiguous_idx = [ 0, 1, 2, 3, 4, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50, 
                         70, 71, 72, 73, 74, 94, 95] # these are specific columns of the pressure pad grid wrapped around the wheel that perform poorly
        
        good_idx = [ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32,
                       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 
                       64, 65, 66, 67, 68, 69, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]           
        

        
        forceX = data_binned['ft_xyz'][:,0]
        forceY = data_binned['ft_xyz'][:,1]
        forceZ = data_binned['ft_xyz'][:,2]
        
        torqueX = data_binned['ft_xyz'][:,3]
        torqueY = data_binned['ft_xyz'][:,4]
        torqueZ = data_binned['ft_xyz'][:,5]
        
        theta = theta_r+theta_f
        
        current = data_binned['current'] * v2c # Motor current draw
        torque = current*eta*kappa*zeta*r # Torque at output motor shaft Nm
        
        slip = np.array(data_binned['slip'])
        
        pressure = slopeCal*data_binned['pg'] + intCal
        aGrid = np.asarray(contact_data['all']['npix'])*aTaxel
        contact_idx = contact_data['all']['contact_coords']
        
        contact_mask = np.zeros((len(slip),))
        for i in range(len(slip)):
            '''
            Apply a mask to the data, removing any points that coincide with when an "ambiguous" section of the wheel was in 
            contact with the ground
            '''
            contact_mask[i] = np.isin(contact_idx[i][:,1],ambiguous_idx).sum()
            
        contact_mask = contact_mask == 0 
        pressure = pressure[contact_mask]
        slip = slip[contact_mask]
        aGrid = aGrid[contact_mask]
        theta = theta[contact_mask]
        torque = torque[contact_mask]
        
        load = []
        
        for i in range(len(pressure[:,:,:])):  
            '''
            Sum the pressure data to get total force, so that total load can be divided by contact area 
            to get sigma (total pressure on the wheel-terrain interface)
            
            '''
            pressure_mask = (pressure[i,:,:] > 0)
            force = np.sum(pressure[i,:,:][pressure_mask])
            load.append(force)
        
        sigma = load/aGrid
        
        loads.append(load)
        slips.append(slip)
        torques.append(torque)
        sigmas.append(sigma)
        slipsexp = np.hstack((slip))
        labelexp = np.ones(slipsexp.shape)*pntr
        label.append(labelexp)
        areas.append(aGrid)
        
        sinks.append(sink)
        sinks2.append(sink[0])
    
    pntr += 1 #class label

slips = np.hstack((slips))
torques = np.hstack((torques))
sigmas = np.hstack((sigmas))
label = np.hstack((label))

ML_data = np.column_stack((slips, torques, sigmas, label))
ML_data = ML_data[~np.isnan(ML_data).any(axis=1)]

sink_exp_data = sum([loads, areas], [])

np.save('ML_data', ML_data)


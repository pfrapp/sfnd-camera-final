#!python3

import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

print('Hello from Python, current Anaconda version is 3.7.4')
print(sys.version)

lidar_distances = np.array([8.015, 7.945, 7.9, 7.843, 7.774, 7.724, 7.653, 7.606, 7.565, 7.51, 7.436, 7.37, 7.29, 7.229, 7.14, 7.062, 6.974, 6.91], dtype='float64')
lidar_velocities = np.array([0.549994, 0.709996, 0.489998, 0.59, 0.689998, 0.66, 0.73, 0.469999, 0.419998, 0.579996, 0.710001, 0.66, 0.809999, 0.619998, 0.890002, 0.730004, 0.869999, 0.500002], dtype='float64')
lidar_ttc = np.array([14.5729, 11.1902, 16.1225, 13.2932, 11.2667, 11.703, 10.4836, 16.183, 18.012, 12.9484, 10.4732, 11.1667, 9.00002, 11.6597, 8.02245, 9.67392, 8.0161, 13.8199], dtype='float64')
time_steps = list(range(len(lidar_distances)))

# Compute the expected collision time (ECT)
framerate = 10.0    # Hertz
dT = 1.0 / framerate
lidar_ect = np.zeros(lidar_ttc.shape)
for n, ttc in enumerate(lidar_ttc):
    lidar_ect[n] = 0.1 * n + ttc

mean_velocity = np.mean(lidar_velocities)
mean_ttc = np.mean(lidar_ttc)
mean_ect = np.mean(lidar_ect)
# Make sure to set ddof = 1 in order to get the same behavior
# as in Matlab (by dividing by (N-1) instead of N)
std_ect = np.std(lidar_ect, ddof=1)
print('Average velocity = ' + str(mean_velocity) + ' m/s = ' + str(mean_velocity*3.6) + ' km/h')
print('Average TTC = ' + str(mean_ttc) + ' s (does not make too much sense, as the TTC is not expected to be constant)')
print('Average ECT = ' + str(mean_ect) + ' +- ' + str(std_ect) + \
    ' s = [' + str(mean_ect-std_ect) + ', ' + str(mean_ect+std_ect) + '] s')



img_path = './../img/'

with PdfPages(img_path + 'lidar_distances.pdf') as pdf:
    fig = plt.figure()
    plt.scatter(time_steps, lidar_distances)
    plt.grid(True)
    plt.xlabel('Time step')
    plt.ylabel('Lidar distance (m)')
    pdf.savefig(fig)

with PdfPages(img_path + 'lidar_velocities.pdf') as pdf:
    fig = plt.figure()
    #plt.hist(lidar_velocities, bins=20)
    plt.scatter(time_steps, lidar_velocities)
    plt.grid(True)
    plt.xlabel('Time step')
    plt.ylabel('Lidar velocity (m/s)')
    plt.ylim(0,2)
    pdf.savefig(fig)

with PdfPages(img_path + 'lidar_ttc.pdf') as pdf:
    fig = plt.figure()
    plt.scatter(time_steps, lidar_ttc)
    plt.grid(True)
    plt.xlabel('Time step')
    plt.ylabel('Lidar time-to-collision (s)')
    plt.ylim(5,20)
    pdf.savefig(fig)

with PdfPages(img_path + 'lidar_ect.pdf') as pdf:
    fig = plt.figure()
    plt.scatter(time_steps, lidar_ect)
    plt.plot([time_steps[0], time_steps[-1]], [mean_ect, mean_ect], color=[1,0,0])
    plt.plot([time_steps[0], time_steps[-1]], [mean_ect-std_ect, mean_ect-std_ect], color=[0.5,0.5,0.5])
    plt.plot([time_steps[0], time_steps[-1]], [mean_ect+std_ect, mean_ect+std_ect], color=[0.5,0.5,0.5])
    plt.plot([time_steps[0], time_steps[-1]], [mean_ect-2*std_ect, mean_ect-2*std_ect], color=[0.8,0.8,0.8])
    plt.plot([time_steps[0], time_steps[-1]], [mean_ect+2*std_ect, mean_ect+2*std_ect], color=[0.8,0.8,0.8])
    plt.grid(True)
    plt.xlabel('Time step')
    plt.ylabel('Lidar expected collision time (s)')
    plt.ylim(5,20)
    pdf.savefig(fig)


#!python3

import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Check if the python version is the one we want
print('Hello from Python, current Anaconda version is 3.7.4')
print(sys.version)

img_path = './../img/'

# Read data from file
base_path = os.path.abspath(os.curdir + '/../performance_eval_camera')
for file_export in os.listdir(base_path):
    with open(os.path.join(base_path, file_export), 'r') as fid:
        l1 = fid.readline()
        l2 = fid.readline()
        l3 = fid.readline()
        l4 = fid.readline()
        l5 = fid.readline()
        l6 = fid.readline()

    detector = l1[10:-1]
    descriptor = l2[12:-1]
    print('----------------------------------')
    print('Detector: {}'.format(detector))
    print('Descriptor: {}'.format(descriptor))
    camera_ttc = [float(element) for element in l3.strip('TTC: ').strip(', \n').split(', ')]
    camera_ect = [float(element) for element in l4.strip('ECT: ').strip(', \n').split(', ')]
    mean_ect = float(l5[10:-1])
    std_ect = float(l6[9:-1])
    print('Mean ECT: {}'.format(mean_ect))
    print('Standard deviation ECT: {}'.format(std_ect))
    print('----------------------------------')

    time_steps = list(range(len(camera_ttc)))

    with PdfPages(img_path + 'camera_ttc_det_' + detector + '_desc_' + descriptor + '.pdf') as pdf:
        fig = plt.figure()
        plt.scatter(time_steps, camera_ttc)
        plt.grid(True)
        plt.xlabel('Time step')
        plt.ylabel('Camera time-to-collision (s)')
        plt.title('Detector: {}, Descriptor: {}'.format(detector, descriptor))
        # plt.ylim(5,20)
        pdf.savefig(fig)
        plt.close()

    with PdfPages(img_path + 'camera_ect_det_' + detector + '_desc_' + descriptor + '.pdf') as pdf:
        fig = plt.figure()
        plt.scatter(time_steps, camera_ect)
        plt.plot([time_steps[0], time_steps[-1]], [mean_ect, mean_ect], color=[1,0,0])
        plt.plot([time_steps[0], time_steps[-1]], [mean_ect-std_ect, mean_ect-std_ect], color=[0.5,0.5,0.5])
        plt.plot([time_steps[0], time_steps[-1]], [mean_ect+std_ect, mean_ect+std_ect], color=[0.5,0.5,0.5])
        plt.plot([time_steps[0], time_steps[-1]], [mean_ect-2*std_ect, mean_ect-2*std_ect], color=[0.8,0.8,0.8])
        plt.plot([time_steps[0], time_steps[-1]], [mean_ect+2*std_ect, mean_ect+2*std_ect], color=[0.8,0.8,0.8])
        plt.grid(True)
        plt.xlabel('Time step')
        plt.ylabel('Camera expected collision time (s)')
        plt.title('Detector: {}, Descriptor: {}'.format(detector, descriptor))
        # plt.ylim(5,20)
        pdf.savefig(fig)
        plt.close()

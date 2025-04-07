import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import program_init as init
import re

class File_Data:
    def __init__(self, data_frame):
        self.file_name, self.theta_arr, self.r_arr, self.i_arr = get_values(data_frame)
        self.pattern = r"(\w+)\.(\d+)\.(\d+)\.(\w+)X(\d+)Y(\d+)_leddar(\d+)\.lvm"
        self.pattern2 = r"(\w+)\.(\d+)\.(\d+)\.(\w+)\.(\w+)X(\d+)Y(\d+)_leddar(\d+)\.lvm"
        self.obj_x, self.obj_y, self.leddar, self.identifier, self.pair_label = self.split_file_name(self.file_name)
    
    def split_file_name(self, pattern):
        # Use re.match to find the pattern
        match = re.match(self.pattern, self.file_name)
        match2 = re.match(self.pattern2, self.file_name)

        if match:
            shape, day, month, identifier, x, y, leddar = match.groups()
            pair_label = identifier + '.' + x + '.' + y
            if leddar == '1':
                leddar = 'Left'
            elif leddar == '2':
                leddar = 'Right'
            else:
                leddar = 'Unknown'
        elif match2:
            shape, day, month, identifier_a, identifier_b, x, y, leddar = match2.groups()
            identifier = identifier_a + '.' + identifier_b
            pair_label = identifier + '.' + x + '.' + y
            if leddar == '1':
                leddar = 'Left'
            elif leddar == '2':
                leddar = 'Right'
            else:
                leddar = 'Unknown'
        else:
            shape = day = month = identifier = x = y = leddar = pair_label =None
        
        return x, y, leddar, identifier, pair_label










def get_values(data_frame, offset=0):
    df = data_frame
    file_name = df.attrs['filename']
    r_arr = df['Distance (cm)']
    i_arr = df['Intensity (relative)']
    r_arr = r_arr[0:16]
    i_arr = i_arr[0:16]
    theta_arr = np.linspace(90-45/2+offset, 90+45/2+offset, 16)
    theta_arr = np.radians(theta_arr)

    return file_name, theta_arr, r_arr, i_arr

def polar2cartesian(theta, r):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def plot_polar(ax, file_name, theta_arr, r_arr, i_arr):
    ax.plot(theta_arr, r_arr, label='Distance (cm)')
    lidar_plot_polar(ax)
    ax.set_title(file_name)
    ax.legend()

def plot_cartesian(ax, file_name, x, y):
    ax.scatter(x, y, label='Distance (cm)')
    lidar_plot_cartesian(ax)
    ax.set_title(file_name)
    ax.legend()

def lidar_plot_polar(ax):
    theta = np.radians(np.linspace(90-45/2, 90+45/2, 16))
    r = np.full(theta.shape, 10000)
    for t in theta:
        ax.plot([t, t], [0, 10000], linestyle='--', color='gray')

def lidar_plot_cartesian(ax):
    theta = np.radians(np.linspace(90-45/2, 90+45/2, 16))
    r = np.full(theta.shape, 10000)
    x, y = polar2cartesian(theta, r)
    for i in range(len(x)):
        ax.plot([0, x[i]], [0, y[i]], linestyle='--', color='gray')
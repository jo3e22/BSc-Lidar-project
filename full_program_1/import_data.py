import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import program_init as init

class ScanData:
    def __init__(self, file_name, data):
        self.file_name = file_name
        self.data = data
        self.theta, self.r, self.intensity = self.get_data()
    
    def get_data(self):
        theta = []
        r = []
        intensity = []
        for i in range(0, len(data), 16):
            thetai = np.linspace(90-45/2, 90+45/2, 16)
            thetai = np.radians(theta)
            ri = data[i:i+16, 0]
            intensityi = data[i:i+16, 1]

            theta.append(thetai)
            r.append(ri)
            intensity.append(intensity)
        return theta, r, intensity
    
    def plot_data(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        #plot each section of 16 rows as a separate line
        
        ax.set_title(self.file_name)
        ax.legend()
        plt.show()

data_frames = init.get_dataframes("C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder")
df = data_frames[3]
file_name = df.attrs['filename']

r_arr = df['Distance (cm)']
r_arr = r_arr[0:16]
theta_arr = np.radians(np.linspace(90-45/2, 90+45/2, 16))
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#plot each section of 16 rows as a separate line
ax.plot(theta_arr, r_arr, marker = 'o', label='Intensity')


ax.set_title(file_name)
ax.legend()
plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in the data
csv_folder = 'csv_folder'
files = os.listdir(csv_folder)

def correct_zeros(data):
    '''
    This function corrects the zeros in the data. If the intensity is zero, the radial value is replaced with 10000.
    input: data - the data to be corrected
    output: None
    '''
    data.loc[data['i'] == 0, 'r'] = 10000

def cartesian_plot(data):
    '''
    This function plots the data in cartesian coordinates.
    input: data - the data to be plotted
    output: None
    '''
    x = data['x_origin'] + data['r'] * np.cos(data['theta (rad)'])
    y = data['r'] * np.sin(data['theta (rad)'])
    return x, y

def extract_obj(file):
    '''
    This function extracts the object from the file name.
    input: file - the file name
    output: obj - the object
    '''
    obj = str(file).split('.')[-3:-1]
    return obj

def read_file(file):
    fig, ax = plt.subplots(figsize = (10, 10))
    data = pd.read_csv(os.path.join(csv_folder, file))
    correct_zeros(data)

    obj = extract_obj(file)
    x, y = cartesian_plot(data)

    ax.scatter(int(obj[0]), int(obj[1]), c = 'red', s = 50)
    ax.plot([data['x_origin'], x], [data['y_origin'], y], color = 'gray', linewidth = 0.5)
    ax.scatter(x, y, c = data['i'], cmap = 'viridis', s = 10)
    ax.set_title(file)
    ax.set_ylim(0, 1520)
    ax.set_xlim(0, 1545)
    plt.show()

file = files[310]
for file in files[-70:-10]:
    read_file(file)

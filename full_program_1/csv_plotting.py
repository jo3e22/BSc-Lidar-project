import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as math
import walls as walls

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

def normalise_intensities(data):
    data['i'] = data['i'] * ((2*data['r'])**2)
    data['i'] = data['i'] / data['i'].max()
    data['i'] = data['i'] * 255
    #print(f'min > 0: {data[data["i"]>0]["i"].min()}, max: {data["i"].max()}')

def correct_origin(data):
    data['x_origin'] += 50

def add_xy(data):
    '''
    This function plots the data in cartesian coordinates.
    input: data - the data to be plotted
    output: None
    '''
    x = data['x_origin'] + data['r'] * np.cos(data['theta (rad)'])
    y = data['r'] * np.sin(data['theta (rad)'])
    data['x'] = x
    data['y'] = y

def extract_obj(file):
    '''
    This function extracts the object from the file name.
    input: file - the file name
    output: obj - the object
    '''
    obj = str(file).split('.')[-3:-1]
    return obj

def plot_background(ax, data):
    x = [370, 480, 590, 700, 810, 920, 1030, 1140]
    y = [330, 480, 630, 780, 930, 1080, 1305, 1520]
    room = np.zeros((1545, 1520))

    for i in x:
        for j in y:
            ax.plot(i, j, '+', color='grey')
    ax.plot(data['x_origin'][0], 0, 'o', color = 'blue')
    ax.plot(data['x_origin'][17], 0, 'o', color = 'blue')
    ax.axvline(x=223, color = 'grey', linestyle='dotted')
    ax.axvline(x=1543, color = 'grey', linestyle='dotted')
    ax.axhline(y=1520, color = 'grey', linestyle='dotted')

def create_detector_mask(data, height, width):
    theta = data['theta (rad)']
    mask = np.zeros((width, height))
    start_angle = theta-np.deg2rad(1.5)
    end_angle = theta+np.deg2rad(1.5)
    distance = data['r']
    x_origin = data['x_origin']
    y_origin = data['y_origin']
    
    mask1 = np.copy(mask)
    for d in range(0, 16):
        mask1 += create_binary_mask((x_origin[d], y_origin[d]), distance[d], start_angle[d], end_angle[d], (width, height), data['i'][d])
    mask2 = np.copy(mask)
    for d in range(16, 32):
        mask2 += create_binary_mask((x_origin[d], y_origin[d]), distance[d], start_angle[d], end_angle[d], (width, height), data['i'][d])
    #mask[(mask1 > 0) & (mask2 > 0)] = mask1[(mask1 > 0) & (mask2 > 0)] + mask2[(mask1 > 0) & (mask2 > 0)]
    #mask = mask1 + mask2
    #mask = 255-mask
    mask[(mask1 + mask2 > 0)] = 255
    return mask

def create_binary_mask(origin, radius, start_angle, end_angle, image_size, intensity):
    mask = np.zeros(image_size, dtype=np.uint8)
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    distance_from_origin = np.sqrt((x - origin[0])**2 + (y - origin[1])**2)
    angle_from_origin = np.arctan2(y - origin[1], x - origin[0])
    angle_from_origin = (angle_from_origin + 2 * np.pi) % (2 * np.pi)
    '''mask[(distance_from_origin >= radius-10) & 
         (distance_from_origin <= radius+10) &
         (angle_from_origin >= start_angle) & 
         (angle_from_origin <= end_angle)] = (intensity)'''
    mask[(distance_from_origin <= radius) & 
         (angle_from_origin >= start_angle) & 
         (angle_from_origin <= end_angle)] = 1  #(intensity)+10
    return mask

def plot(data, ax):
    mask = create_detector_mask(data, 1545, 1550)
    binary_mask = np.zeros_like(mask)
    binary_mask[mask > 0] = 1 
    obj = extract_obj(file)
    ax[0].imshow(mask, cmap = 'gray', origin = 'lower')
    ax[0].scatter(int(obj[0]), int(obj[1]), c = 'red', s = 50)
    ax[0].set_aspect('equal')
    ax[0].axis('off')
    plot_background(ax[0], data)

    plot_background(ax[1], data)
    x, y = data['x'], data['y']
    ax[1].scatter(int(obj[0]), int(obj[1]), c = 'red', s = 50)
    ax[1].plot([data['x_origin'], x], [data['y_origin'], y], color = 'gray', linewidth = 0.5)
    ax[1].scatter(x, y, c = data['i'], cmap = 'viridis', s = 10)
    ax[1].set_title(file)
    ax[1].set_ylim(0, 1550)
    ax[1].set_xlim(0, 1545)
    ax[1].set_aspect('equal')
    ax[1].axis('off')

def read_file(file):
    data = pd.read_csv(os.path.join(csv_folder, file))
    correct_origin(data)
    correct_zeros(data)
    normalise_intensities(data)
    add_xy(data)

    return data



file = files[310]
for file in files[-14:-1]:
    data = read_file(file)

    fig, ax = plt.subplots(1, 2, figsize = (15, 30))
    plot(data, ax)

    segments = walls.find_segment(data[['x', 'y', 'i']].values, epsilon=20)

    for segment in segments:
        print(f'segment: {segment}')
    connected_segments = walls.join_connected_segments(segments)
    print('\n\n')

    for connected_segment in connected_segments:
        print(f'connected_segment: {connected_segment}')
    print('\n\n')
  
    '''
    for connected_segment in connected_segments:
        x_arr = [x for (x, y) in connected_segment]
        y_arr = [y for (x, y)  in connected_segment]
        a, b = walls.best_fit(x_arr, y_arr)
        y_fit = [a + b * x for x in x_arr]
        ax[1].plot(x_arr, y_fit, color = 'blue')
    
    for segment in segments:
        x_arr = [x for (x, y) in segment]
        y_arr = [y for (x, y) in segment]
        a, b = walls.best_fit(x_arr, y_arr)
        y_fit = [a + b * x for x in x_arr]
        ax[1].plot(x_arr, y_fit, color = 'red')'''


    plt.show()

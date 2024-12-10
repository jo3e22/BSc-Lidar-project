import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_ellipse_mask(width, depth, center_x, center_y, grid_size):
    # Create a grid of the specified size
    y, x = np.ogrid[:grid_size, :grid_size]
    center_y += 7.5
    if center_y > 1519:
        center_y = 1510

    # Calculate the mask
    mask = ((x - center_x)**2 / width**2 + (y - (center_y))**2 / depth**2 <= 1).astype(int)
    
    return mask

def detector_mask(data, width, height):
    theta = data['theta (rad)']
    mask = np.zeros((width, height))
    mask_array = []
    start_angle = theta-np.deg2rad(1)
    end_angle = theta+np.deg2rad(1)
    distance = data['r']
    x_origin = data['x_origin']
    y_origin = data['y_origin']
    
    for d in range(0, 16):
        det_mask = np.zeros((width, height))
        det_mask += create_binary_mask((x_origin[d], y_origin[d]), distance[d], start_angle[d], end_angle[d], (width, height), data['i'][d])
        mask_array.append(det_mask)

    for d in range(16, 32):
        det_mask = np.zeros((width, height))
        det_mask += create_binary_mask((x_origin[d], y_origin[d]), distance[d], start_angle[d], end_angle[d], (width, height), data['i'][d])
        mask_array.append(det_mask)
    
    return mask_array

def create_binary_mask(origin, radius, start_angle, end_angle, image_size, intensity):
    mask = np.zeros(image_size, dtype=np.uint8)
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    distance_from_origin = np.sqrt((x - origin[0])**2 + (y - origin[1])**2)
    angle_from_origin = np.arctan2(y - origin[1], x - origin[0])
    angle_from_origin = (angle_from_origin + 2 * np.pi) % (2 * np.pi)

    mask[(distance_from_origin <= 3000) & 
         (angle_from_origin >= start_angle) & 
         (angle_from_origin <= end_angle)] = 1
    return mask

def distances(data, james, ax):
    r_arr = []
    distances_arr = []
    diff_arr = []
    sensor_num = []
    data_diff = np.zeros_like(data['r'])
    for i in range(len(data['detector_mask'])):
        mask = data['detector_mask'][i] * james

        if mask.sum() > 0 and data['i'][i] > 0:
            r = data['r'][i]

            #find average distnace between origin and the points in the mask
            y, x = np.where(mask > 0)
            distances = np.sqrt((x - data['x_origin'][i])**2 + (y - data['y_origin'][i])**2)
            sorted_distances = np.sort(distances)
            front_faces = sorted_distances[1:11]
            diff = (r - front_faces)


            r_arr.append(r)
            sensor_num.append(i)
            distances_arr.append(np.mean(front_faces))
            diff_arr.append(np.mean(diff))
            data_diff[i] = np.mean(diff)
            '''
            plt.imshow(mask, cmap = 'gray', origin = 'lower')
            plt.plot(data['x_origin'][i], data['y_origin'][i], 'ro')
            x = data['x_origin'][i]
            y = data['y_origin'][i]
            mid_angle_rad = data['theta (rad)'][i]
            distance = data['r'][i]
            plt.plot([x + distance * np.cos(mid_angle_rad-np.deg2rad(1.5)), x + distance * np.cos(mid_angle_rad+np.deg2rad(1.5))], 
                    [y + distance * np.sin(mid_angle_rad-np.deg2rad(1.5)), y + distance * np.sin(mid_angle_rad+np.deg2rad(1.5))], 
                    linestyle='dotted', color='red')
            plt.show()
            '''

    '''
    for i in range(len(r_arr)):
        diff = (r_arr[i] - distances_arr[i])
        print(f'for Sensor{sensor_num[i]}:: r: {r_arr[i]:.2f}, distance: {distances_arr[i]:.2f}, difference: {diff:.2f}')

    print(f'Average r: {np.mean(r_arr):.2f}, average distance: {np.mean(distances_arr):.2f}, difference of averages: {np.mean(diff_arr):.2f}\n')
    '''
    return data_diff

def run(data, object, ax):
    james = create_ellipse_mask(40, 15, int(object[0]), int(object[1]), 1545)
    local_data = data.copy()
    mask_arr = detector_mask(local_data, 1545, 1545)
    data['detector_mask'] = mask_arr

    data_diff = distances(data, james, ax)
    local_data['r_diff'] = data_diff
    #ax.imshow(james, cmap = 'gray', origin = 'lower')

    #print(local_data)
    return local_data['r_diff']
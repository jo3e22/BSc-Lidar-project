import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_ellipse_mask(width, depth, center_x, center_y, grid_size):
    # Create a grid of the specified size
    y, x = np.ogrid[:grid_size[0], :grid_size[1]]
    center_y += 7.5
    if center_y > 1519:
        center_y = 1510

    # Calculate the mask
    mask = ((x - center_x)**2 / width**2 + (y - (center_y))**2 / depth**2 <= 1).astype(int)
    mask[:, 0:center_x-25] = 0
    mask[:, center_x+25:-1] = 0
    
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

def detector_mask2(angle_offset, x_origin, y_origin):
    centre = 90 + angle_offset
    mask_list = []
    mask = np.zeros((1520, 1543))
    theta = np.linspace(centre + 22.5 + 9, centre - 22.5 - 9, (64))
    #print(f'theta: {theta}')
    theta = np.deg2rad(theta)
    start_angle = theta-np.deg2rad(1.5)
    end_angle = theta+np.deg2rad(1.5)

    for i in range(len(theta)):
        det_mask = np.zeros_like(mask)
        det_mask += create_binary_mask((x_origin, y_origin), start_angle[i], end_angle[i], (1520, 1543))
        mask_list.append({'theta (rad)': theta[i], 'mask': det_mask})
    
    mask_data = pd.DataFrame(mask_list)
    return mask_data

def angles_to_origin(obj, x_origin, y_origin):
    x_left = obj[0] - 25
    x_right = obj[0] + 25
    angle_left = np.arctan2(obj[1] - y_origin, x_left - x_origin)
    angle_right = np.arctan2(obj[1] - y_origin, x_right - x_origin)
    angles = [angle_left, angle_right]
    np.sort(angles)
    return angles

def create_binary_mask(origin, start_angle, end_angle, image_size):
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

def initialise_objects(x = [370, 480, 590, 700, 810, 920, 1030, 1140], y = [165, 330, 480, 630, 780, 930, 1080, 1300, 1520]):
    objects = []
    for i in x:
        for j in y:
            object_mask = create_ellipse_mask(40, 15, int(i), int(j), (1520, 1543))
            objects.append({'co-ordinates': (i,j), 'mask': object_mask})
    obj_df = pd.DataFrame(objects)
    return obj_df

def initialise_walls(x = 1543, y = 1520):
    walls = []
    wall_mask = np.ones((y, x))
    wall_mask[0:-2, 1:-1] = 0
    walls.append({'co-ordinates': (0, 0), 'mask': wall_mask})

    walls_df = pd.DataFrame(walls)


    return walls_df

def initialise_detector_masks():
    mask_list = []
    mask = np.zeros((1550, 2000))
    theta = np.linspace(178.5, 1.5, 178)
    theta = np.deg2rad(theta)
    start_angle = theta-np.deg2rad(1.5)
    end_angle = theta+np.deg2rad(1.5)

    for i in range(len(theta)):
        det_mask = np.zeros_like(mask)
        det_mask += create_binary_mask((1000, 0), start_angle[i], end_angle[i], (1550, 2000))
        mask_list.append({'theta (rad)': theta[i], 'mask': det_mask})
    
    mask_df = pd.DataFrame(mask_list)
    mask_df.origin_x = 1000
    mask_df.origin_y = 0

    return mask_df

def generate_distances(object_df: pd.DataFrame, mask_df: pd.DataFrame, origin: tuple, testing_objs = False, walls = False, data_df = None):
    print(f'data_df: {data_df}')
    print(f'test')
    for index in object_df.index:
        object_mask = object_df['mask'][index]

        if walls != True:
            obj = object_df['co-ordinates'][index]

            mask_df[f'r_{obj}'] = np.zeros_like(mask_df['theta (rad)'])
            angles = angles_to_origin(obj, origin[0], origin[1])
            start = min(angles)
            end = max(angles)

        else:
            obj = 'wall'
            mask_df[f'r_{obj}'] = np.zeros_like(mask_df['theta (rad)'])
            start = 0
            end = np.pi

        if testing_objs:
            try:
                x_data = data_df[f'x.{obj[0]}.{obj[1]}']
                y_data = data_df[f'y.{obj[0]}.{obj[1]}']
                plt.plot(x_data, y_data, 'ro')
            except:
                pass
            plt.plot(origin[0], origin[1], 'ro')
            plt.plot([origin[0], origin[0] + 3000*np.cos(start)], [origin[1], origin[1] + 3000*np.sin(start)], 'r')
            plt.plot([origin[0], origin[0] + 3000*np.cos(end)], [origin[1], origin[1] + 3000*np.sin(end)], 'r')
            plt.imshow(object_mask, cmap = 'gray', origin = 'lower')
            plt.show()

        sub_data = mask_df.loc[
            (mask_df['theta (rad)'] >= (start-np.deg2rad(1))) & 
            (mask_df['theta (rad)'] <= (end+np.deg2rad(1)))
            ]

        for index in sub_data.index:
            mask = mask_df['mask'][index] * object_mask
            y, x = np.where(mask > 0)
            distances = np.sqrt((x - origin[0])**2 + (y - origin[1])**2)
            sorted_distances = np.sort(distances)
            mean_distance = np.mean(sorted_distances[1:11])
            mask_df.at[index, f'r_{obj}'] = mean_distance

            if data_df is not None:
                try:
                    diff = data_df[f'r.{obj[0]}.{obj[1]}'][index] - mean_distance
                    print(f'diff: {diff}')
                    print(f'corrected_diff: {data_df[f"corrected_diff_.{obj[0]}.{obj[1]}"][index]}')
                    if data_df[f'corrected_diff_.{obj[0]}.{obj[1]}'][index] is True:
                        data_df[f'corrected_diff_.{obj[0]}.{obj[1]}'] = diff
                    data_df[f'diff.{obj[0]}.{obj[1]}'] = diff
                except:
                    pass

            if testing_objs:
                print(f'object: {obj}, angle: {mask_df["theta (rad)"][index]:.2f}, distance: {mean_distance:.2f}')
    
    if data_df is not None:
        return mask_df, data_df
    else:
        return mask_df

def run2(object_list, data, testing_objs = False):
    left_mask_data = detector_mask2(data.offset_angle, data['x_origin'][0], data['y_origin'][0])
    right_mask_data = detector_mask2(data.offset_angle, data['x_origin'][17], data['y_origin'][17])

    for obj in object_list:
        left_mask_data[f'r_{obj}'] = np.zeros_like(left_mask_data['theta (rad)'])
        right_mask_data[f'r_{obj}'] = np.zeros_like(right_mask_data['theta (rad)'])

        l_angles = angles_to_origin(obj, data['x_origin'][0], data['y_origin'][0])
        l_start = min(l_angles)
        l_end = max(l_angles)
        r_angles = angles_to_origin(obj, data['x_origin'][17], data['y_origin'][17])
        r_start = min(r_angles)
        r_end = max(r_angles)
        james = create_ellipse_mask(40, 15, int(obj[0]), int(obj[1]), 1545)

        if testing_objs:
            plt.plot(data['x_origin'][0], data['y_origin'][0], 'ro')
            plt.plot(data['x_origin'][17], data['y_origin'][17], 'bo')
            plt.plot([data['x_origin'][0], data['x_origin'][0] + 3000*np.cos(l_start)], [data['y_origin'][0], data['y_origin'][0] + 3000*np.sin(l_start)], 'r')
            plt.plot([data['x_origin'][0], data['x_origin'][0] + 3000*np.cos(l_end)], [data['y_origin'][0], data['y_origin'][0] + 3000*np.sin(l_end)], 'r')
            plt.plot([data['x_origin'][17], data['x_origin'][17] + 3000*np.cos(r_start)], [data['y_origin'][17], data['y_origin'][17] + 3000*np.sin(r_start)], 'b')
            plt.plot([data['x_origin'][17], data['x_origin'][17] + 3000*np.cos(r_end)], [data['y_origin'][17], data['y_origin'][17] + 3000*np.sin(r_end)], 'b')

            plt.imshow(james, cmap = 'gray', origin = 'lower')
   
        left_sub_data = left_mask_data.loc[
            (left_mask_data['theta (rad)'] >= (l_start-np.deg2rad(1))) &
            (left_mask_data['theta (rad)'] <= (l_end+np.deg2rad(1)))
            ]
        right_sub_data = right_mask_data.loc[
            (right_mask_data['theta (rad)'] >= (r_start-np.deg2rad(1))) &
            (right_mask_data['theta (rad)'] <= (r_end+np.deg2rad(1)))
            ]

        for index in left_sub_data.index:
            mask = left_mask_data['mask'][index] * james
            y, x = np.where(mask > 0)
            distances = np.sqrt((x - data['x_origin'][0])**2 + (y - data['y_origin'][0])**2)
            sorted_distances = np.sort(distances)
            mean_distance = np.mean(sorted_distances[1:11])
            left_mask_data.at[index, f'r_{obj}'] = mean_distance

    print(left_mask_data)

if __name__ == '__main__':
   initialise_walls()
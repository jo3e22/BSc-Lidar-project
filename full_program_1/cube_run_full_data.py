#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import numpy as np
import math as math
import walls as walls
import error as error
from scipy.stats import norm
from scipy.stats import gaussian_kde
import re 
import time
from typing import Any, Tuple, Union
import inspect

# Read in the data
Timing_enabled = False
#directory = 'C:/Users/james/OneDrive/Desktop/lidar_code/full_data'
directory = r'C:\Users\james\OneDrive - University of Southampton\PHYS part 3\BSc Project\Code\full_data2'
room_x = int(1543)
room_y = int(1520)
stage_x = int(223)

plot_counter = 0
ax_d = {}

class InputTypeError(Exception):
    """Custom exception for input type errors."""
    pass

def validate_inputs(func: Any, *args, **kwargs) -> None:
    sig = inspect.signature(func)
    for name, value in sig.parameters.items():
        expected_type = value.annotation
        if expected_type is not inspect.Parameter.empty:
            if not isinstance(kwargs.get(name, args[list(sig.parameters.keys()).index(name)]), expected_type):
                raise InputTypeError(f"Expected {expected_type} for '{name}', got {type(kwargs.get(name, args[list(sig.parameters.keys()).index(name)]))}")
            
def time_function(func):
    def wrapper(*args, **kwargs):
        try:
            validate_inputs(func, *args, **kwargs)
        except InputTypeError as e:
            print(f"Input validation error: {e}")
            return None
        
        if Timing_enabled:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time for {func.__name__}:{elapsed_time:.4f} seconds")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper

@time_function
def return_attributes(filename: str, data: pd.DataFrame) -> Union[Tuple[int, int], None]:
    try:
        name = filename.strip('.csv')
        parts = name.split('.')
        if len(parts) != 3:
            raise ValueError(f"Filename {filename} is not in the correct format.")
        
        separation = int(parts[0])
        offset_angle = int(parts[1])

        data.separation = separation
        data.offset_angle = offset_angle

        return separation, offset_angle

    except (InputTypeError, ValueError) as e:
            print(f"Input validation error: {e}")
            return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

@time_function
def return_data(data: pd.DataFrame):
    #pattern = r"^(?P<prefix>corrected_diff|r|i|x|y|diff)_X(?P<xvalue>\d+)Y(?P<yvalue>\d+)$"
    pattern = r"^(?P<prefix>corrected_diff_|r|i|x|y|diff_).(?P<xvalue>\d+).(?P<yvalue>\d+)$"
    data_points = []
    for column in data.columns[4::]:
        match = re.match(pattern, column)
        if match:
            prefix = match.group('prefix')
            x = int(match.group('xvalue'))
            y = int(match.group('yvalue'))

            if (x, y) not in data_points and (x, y) != (0, 0):
                data_points.append((x, y))
        else:
            print(f'No match found for {column}')
        
    return data_points

@time_function
def adjust_detector_masks(detector_df_input: pd.DataFrame, origin: Tuple) -> pd.DataFrame:
    try:
        detector_df = detector_df_input.copy()
        detector_df.origin_x = detector_df_input.origin_x
        detector_df.origin_y = detector_df_input.origin_y

        x_bound = int(detector_df.origin_x - origin[0])
        y_bound = int(detector_df.origin_y - origin[1])

        detector_df['mask'] = detector_df['mask'].apply(lambda mask: mask[int(y_bound):int(y_bound+room_y), int(x_bound):int(x_bound+room_x)])

        return detector_df
    except (InputTypeError, ValueError) as e:
        print(f"Input validation error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

@time_function
def compare_data(data: pd.DataFrame, detector_df: pd.DataFrame, walls_df: pd.DataFrame, origin: Tuple, data_points: list, ax) -> None:
    data = data.copy()
    data['theta (deg)'] = data['theta (rad)'].apply(lambda x: np.rad2deg(x))
    detector_df['theta (deg)'] = detector_df['theta (rad)'].apply(lambda x: np.rad2deg(x))
    data.set_index('theta (deg)', inplace=True)
    detector_df.set_index('theta (deg)', inplace=True)

    results_df = pd.DataFrame()
    results_df['theta_deg'] = data['theta (rad)'].apply(lambda x: np.rad2deg(x))
    results_df.set_index('theta_deg', inplace=True)

    useful_data = []

    for (x, y) in data_points:
        try:
            data_r_col = data[f'r.{x}.{y}']
            data_r_col_nonzero = data_r_col[data_r_col != 10000]
            det_col = detector_df[f'r_({x}, {y})']
            det_col_nonzero = det_col[det_col != 0]

            results_df[f'diff_{x}_{y}'] = np.zeros_like(data_r_col)
            results_df[f'diff_{x}_{y}'] = results_df[f'diff_{x}_{y}'].astype('float64')

            for index in det_col_nonzero.index:
                if index in data_r_col_nonzero.index:
                    diff = data_r_col_nonzero[index]-det_col_nonzero[index]
                    distance = np.sqrt((x-origin[0])**2 + (y-origin[1])**2)

                    wall_dis = walls_df.at[index, 'r_wall']
                    wall_diff = wall_dis - data_r_col_nonzero[index]

                    if abs(wall_diff) < abs(diff) and y < 1450:
                        if abs(wall_diff) > 50:
                            colour = 'purple'
                            ax.scatter(distance, diff+wall_diff, marker='_', color='grey')
                            ax.plot([distance, distance], [diff, diff+wall_diff], color='grey', linestyle='--')
                        else:
                            colour = 'red'
                    else:
                        colour = 'black'
                        results_df.at[index, f'diff_{x}_{y}'] = diff
                    ax.scatter(distance, diff, marker='o', color=colour)

        except Exception as e:
            print(f"An unexpected error occurred while comparing data: {e}")

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Closer to wall than object'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Closer to wall than object but more than 50cm away'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Closer to object than wall')
        ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_xlabel('Distance from origin')
    ax.set_ylabel('Difference in distance')
    ax.set_title('Difference in distance between object and detector')
    ax.set_xlim(0, 1600)
    ax.set_ylim(ymax = 1600)
    ax.set_aspect('equal')

    return results_df

@time_function
def plot_background(ax_obj, data, data_points = None, limits = True):
    x = [370, 480, 590, 700, 810, 920, 1030, 1140]
    y = [165, 330, 480, 630, 780, 930, 1080, 1300, 1520]

    for i in x:
        for j in y:
            #if data_points is not None and (i, j) in data_points:
                #ax_obj.plot(i, j, marker='o', fillstyle='none', color='black', markersize=7)
            #else:
            if data_points is not None and (i, j) not in data_points:
                ax_obj.plot(i, j, marker='+', color='grey', markersize=5)

    ax_obj.scatter(data['x_origin'][0], 0, marker='o', color = 'blue')
    ax_obj.scatter(data['x_origin'][17], 0, marker='o', color = 'blue')
    ax_obj.axvline(x=stage_x, color = 'grey', linestyle='--', label='Stage Edge')
    ax_obj.axvline(x=room_x, color = 'grey', linestyle='--')
    ax_obj.axhline(y=room_y, color = 'grey', linestyle='--')  

    if limits:
        ax_obj.set_xlim(0, room_x)
        ax_obj.set_ylim(0, room_y+27)
        ax_obj.set_aspect('equal')
        ax_obj.set_xticks(x)
        ax_obj.set_xticklabels(x)
        for label in ax_obj.get_xticklabels():
            label.set_rotation(-90)
            label.set_horizontalalignment('right')
        ax_obj.set_yticks(y)

@time_function
def plot_diffs(ax_obj, diff_df: pd.DataFrame, data_points: list, l_r: str) -> None:
    for (x, y) in data_points:
        try:
            diff_col = diff_df[f'diff_{x}_{y}']
            diff_col_nonzero = diff_col[diff_col != 0]
            mean_diff = np.mean(diff_col_nonzero)
            if np.abs(mean_diff) < 30:
                mean_diff_val = np.abs(mean_diff)/30
            else:
                mean_diff_val = 1
            cmap = plt.get_cmap('YlOrRd')
            color = cmap(mean_diff_val)
            ax_obj.scatter(x, y, color=color, marker=MarkerStyle('o', fillstyle=f'{l_r}'), edgecolors='k', s=200)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

@time_function
def plot_kde(differences, ax, limit = 100, extras = False):
    differences = differences[differences != 0]
    print(f'data_points: {len(differences)}')

    included_diffs = []
    for diff in differences:
        if diff < limit and diff > -limit and diff != 0:
            included_diffs.append(diff)
    
    if len(included_diffs) == len(differences):
        limit = int(np.max(np.abs(included_diffs)))
    
    x = np.linspace(max(differences), min(differences), 2000)
    print(f'min and max: {min(differences)}, {max(differences)}')

    x_lim_only = np.linspace(limit, -limit, (2*limit +1))

    mu, std = norm.fit(included_diffs)
    p = norm.pdf(x_lim_only, mu, std)
    print(f'y_val of mean: {norm.pdf(mu, mu, std)}')
    
    kde = gaussian_kde(differences)
    kde_vals = kde(x)
    ax.plot(x, kde_vals, 'k', linewidth=1, label = 'fit results: mu = %.2f,  std = %.2f' % (mu, std))
    ax.fill_between(x, kde_vals, alpha=0.4, color='r')

    #ax.plot(x_lim_only, p, 'k', linewidth=2, label = 'fit results: mu = %.2f,  std = %.2f' % (mu, std))

    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    ax.set_title(title)

    if extras == True:
        ax.axvline(mu, color='r', linestyle='dashed', linewidth=1, label = 'mean')
        ax.axvline(mu-std, color='grey', linestyle='dashed', linewidth=1, label = '1 std')
        ax.axvline(mu+std, color='grey', linestyle='dashed', linewidth=1)
        ax.axvline(limit, color='black', linestyle='dashed', linewidth=1, label = 'limit of included values')
        ax.axvline(-limit, color='black', linestyle='dashed', linewidth=1)
        ax.legend()

    return mu, std

@time_function
def custom_plot(differences, ax, limit = 100):
    differences = differences[differences != 0]
    incl_diffs = []
    for dif in differences:
        if type(dif) == float or type(dif) == np.float64:
            incl_diffs.append(dif)
        else:
            print(f'Non-float value: {dif}')
            print(f'Non-float type: {type(dif)}')

    differences = np.array(incl_diffs)
    differences = np.sort(differences)
    q = len(differences)
    point_volume = 1/q
    height_count = 0

    for i in range(1, len(differences)-1):
        point = differences[i]
        width = np.abs(differences[i+1] - differences[i-1])/2
        height = point_volume/width
        #print(f'width: {width}, height: {height}')
        ax.bar(point, height, width, align='center', alpha=0.5, color='g')
        ax.scatter(point, height, color='black', s=5)
        height_count += height

    print(f'height count: {height_count}')

@time_function    
def custom_hist(differences, ax):
    differences = differences[differences != 0]
    differences = differences[~np.isnan(differences)]
    differences = np.sort(differences)

    if len(differences) == 0:
        #print("No differences found, skipping histogram creation")
        return 0, 0, 0

    start = differences[0]
    end = differences[-1]

    if np.isnan(start) or np.isnan(end):
        print("start or end is NaN, skipping histogram creation")
        return 0, 0, 0

    q = len(differences)
    #print(f'q: {q}')

    num_elements = int((end - start) / 2.5) + 1
    x = np.linspace(start, end, num_elements)
    mu, std = norm.fit(differences)
    p = norm.pdf(x, mu, std)

    ax.hist(differences, bins = x, density = True, alpha = 0.4, color='r', label = 'histogram of differences')
    ax.plot(x, p, 'k', linewidth=2, label = 'fit results')
    ax.axvline(mu, color='r', linestyle='dashed', linewidth=1, label = 'mean')
    ax.axvline(mu-std, color='grey', linestyle='dashed', linewidth=1, label = '1 std')
    ax.axvline(mu+std, color='grey', linestyle='dashed', linewidth=1)

    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('Difference in distance')
    ax.set_ylabel('Frequency')

    return mu, std, q

    


'''
#if object csv and detector csv do not exist, create them
if not os.path.exists(os.path.join(directory, 'objects.csv')):
    obj_df = error.initialise_objects()
    obj_df.to_csv(os.path.join(directory, 'objects.csv'))
else:
    obj_df = pd.read_csv(os.path.join(directory, 'objects.csv'))

if not os.path.exists(os.path.join(directory, 'detectors.csv')):
    detector_df = error.initialise_detector_masks()
    detector_df.to_csv(os.path.join(directory, 'detectors.csv'))
else:
    detector_df = pd.read_csv(os.path.join(directory, 'detectors.csv'))
    #hard coded coordinated for the origin of the detector masks because time is of the essence
    detector_df.origin_x = 1000
    detector_df.origin_y = 0
    print(f"Detector mask shape: {detector_df['mask'].shape}")
    print(f'detector mask shape: {np.array(detector_df["mask"][3]).shape}')
'''
obj_df = error.initialise_objects()
detector_df = error.initialise_detector_masks()
walls_df = error.initialise_walls()

#%%


for filename in os.listdir(directory):
    if filename.endswith("data.csv"):
        take_inputs = input(f'Do you want to take inputsfor file {filename}? (y/n): ')
        data = pd.read_csv(os.path.join(directory, filename))
        return_attributes(filename, data)
        data_points = return_data(data)


        left_origin = (data['x_origin'][0], data['y_origin'][0])
        right_origin = (data['x_origin'][17], data['y_origin'][17])

        left_detectors_df = adjust_detector_masks(detector_df, left_origin)
        right_detectors_df = adjust_detector_masks(detector_df, right_origin)

        left_walls_df = error.generate_distances(walls_df, left_detectors_df, left_origin, False, True)
        right_walls_df = error.generate_distances(walls_df, right_detectors_df, right_origin, False, True)

        left_detectors_df = error.generate_distances(obj_df, left_detectors_df, left_origin, False, False, data)
        right_detectors_df = error.generate_distances(obj_df, right_detectors_df, right_origin, False)

        #left_diffs = compare_data(data[0:16], left_detectors_df, left_walls_df, left_origin, data_points, ax2)
        #right_diffs = compare_data(data[16:32], right_detectors_df, right_walls_df, right_origin, data_points, ax2)

        print(f'\nFilename: {filename}')
        array_of_dfs = []
        for row_index in data.index:
            row_data = pd.DataFrame()
            # Create a copy of the row to avoid modifying the original DataFrame
            row_data_copy = data.iloc[row_index].copy()

            for (x, y) in data_points:
                x_data = row_data_copy[f'x.{x}.{y}']
                y_data = row_data_copy[f'y.{x}.{y}']
                r_data = row_data_copy[f'r.{x}.{y}']
                i_data = row_data_copy[f'i.{x}.{y}']

                new_row = pd.DataFrame({
                    'obj_xy': [(x, y)],
                    'sensor_xy': [(x_data, y_data)],
                    'r': [r_data],
                    'intensity': [i_data],
                    't_f': [False]
                })
                row_data = pd.concat([row_data, new_row], ignore_index=True)

            x_values = [coord[0] for coord in row_data['sensor_xy']]
            y_values = [coord[1] for coord in row_data['sensor_xy']]
            xobj_values = [coord[0] for coord in row_data['obj_xy']]
            yobj_values = [coord[1] for coord in row_data['obj_xy']]
            mean_r = row_data['r'].mean()
            row_data.loc[row_data['r'] >3000, 'r'] = 0

            for row in row_data.iterrows():
                x, y = row[1]['sensor_xy']
                r = row[1]['r']
                i = row[1]['intensity']
                obj_x, obj_y = row[1]['obj_xy']
                difference = np.sqrt((x-obj_x)**2 + (y-obj_y)**2)
                obj_distance = np.sqrt((obj_x-left_origin[0])**2 + (obj_y-left_origin[1])**2)
                
                if r != 0 and difference < 300 and take_inputs == 'y':

                    fig, ax = plt.subplots()
                    ax.scatter(x_values, y_values, c='black', marker='+')
                    ax.scatter(xobj_values, yobj_values, c='grey', marker='o')
                    #ax.plot([x_values, xobj_values], [y_values, yobj_values], c='grey', linestyle='--')
                    ax.plot([x, obj_x], [y, obj_y], c='black', linestyle='--')
                    plot_background(ax, data, data_points)
                    ax.scatter(x, y, c='red', marker='o')
                    ax.scatter(obj_x, obj_y, c='green', marker='o')

                    ax.set_title(f'Filename: {filename}, x: {x}, y: {y}, r: {r}, i: {i}, obj_x: {obj_x}, obj_y: {obj_y}')
                    plt.show()

                    if obj_distance < 15 and obj_y > 1300:
                        tf = True
                    else:
                        t_f = input('Is this a true positive? (y/n): ')

                    
                    if t_f == 'y':
                        row_data.at[row[0], 't_f'] = True
                        data.at[row_index, f'corrected_diff_.{obj_x}.{obj_y}'] = True
                    else:
                        row_data.at[row[0], 't_f'] = False
                        data.at[row_index, f'corrected_diff_.{obj_x}.{obj_y}'] = False

            '''if take_inputs == 'y':
                new_filename = filename.strip('.csv') + '_new'
                new_folder = os.path.join(directory, f'new_data3_{filename.strip(".csv")}')
                output_path = os.path.join(new_folder, f'{new_filename}_{row_index}.csv')
                os.makedirs(new_folder, exist_ok=True)  # Ensure the directory exists
                row_data.to_csv(output_path)
                print(f'Saved {new_filename}_{row_index}.csv to {output_path}')'''
    data.to_csv(os.path.join(directory, f'new{filename}'), index=True)
    print(f'Saved {filename}new to {directory}')



















#%%
directory_180_0 = r"C:\Users\james\OneDrive - University of Southampton\PHYS part 3\BSc Project\Code\full_data2\new_data3_180.0.full_data"
directory_180_24 = r"C:\Users\james\OneDrive - University of Southampton\PHYS part 3\BSc Project\Code\full_data2\new_data_180.24.full_data"
directory_90_0 = r"C:\Users\james\OneDrive - University of Southampton\PHYS part 3\BSc Project\Code\full_data2\new_data_90.0.full_data"
directory_90_24 = r"C:\Users\james\OneDrive - University of Southampton\PHYS part 3\BSc Project\Code\full_data2\new_data_90.24.full_data"
directory_180_29 = r"C:\Users\james\OneDrive - University of Southampton\PHYS part 3\BSc Project\Code\full_data2\new_data_180.29.full_data"

directories = [directory_180_0, directory_180_24, directory_90_0, directory_90_24, directory_180_29]

def run_directories(directory):
    parts = directory.split('_')
    separation = int(parts[-2])
    offset_angle = int(parts[-1])

    def coordinates(string_input):
        string_input = string_input[1:-1]
        string_input = string_input.split(',')
        for i in range(0, len(string_input)):
            if 'np.float64' in string_input[i]:
                string_input[i] = string_input[i].strip(' ')
                string_input[i] = string_input[i].strip('np.float64')
                string_input[i] = string_input[i].strip('(')
                string_input[i] = string_input[i].strip(')')
        x, y = string_input
        return float(x), float(y)

    wall_points = []
    for filename in os.listdir(directory):
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        print(f'Filename: {filename}')
        if filename.endswith(".csv"):
            parts = filename.split('_')
            parts[-1] = parts[-1].strip('.csv')
        
            data = pd.read_csv(os.path.join(directory, filename))
            obj_xy = data['obj_xy']
            sensor_xy = data['sensor_xy']

            r = data['r']
            i = data['intensity']
            t_f = data['t_f']

            wall_data = data[(data['t_f'] == False) & (data['r'] != 0)]
            wall_distributions(wall_data, ax[0])
            wall_points.append(wall_data)
        plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    wall_distributions(pd.concat(wall_points), ax)
    plt.show()

    '''
            for row_index in data.index:
                if r[row_index] != 10000:
                    #print(f'Row index: {row_index}, Filename: {filename}, obj_xy: {obj_xy[row_index]}, lenobjxy: {len(obj_xy[row_index])}, sensor_xy: {sensor_xy[row_index]}, r: {r[row_index]}, i: {i[row_index]}, t_f: {t_f[row_index]}')
                    obj_x, obj_y = coordinates(obj_xy[row_index])
                    sensor_x, sensor_y = coordinates(sensor_xy[row_index])

                    tf_i = t_f[row_index]
                    if tf_i:
                        ax[0].scatter(obj_x, obj_y, c='grey', marker='o')
                        ax[0].scatter(sensor_x, sensor_y, c='green', marker='o')
                        ax[0].plot([sensor_x, obj_x], [sensor_y, obj_y], c='black', linestyle='--')
                        plot_background(ax[0], True)
                    else:
                        #ax[1].scatter(obj_x, obj_y, c='grey', marker='o')
                        ax[1].scatter(sensor_x, sensor_y, c='red', marker='o')
                        #ax[1].plot([sensor_x, obj_x], [sensor_y, obj_y], c='black', linestyle='--')
                        plot_background(ax[1], True)
                        wall_points.append((sensor_x, sensor_y, row_index))

        plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    x = [x for x, y, i in wall_points]
    y = [y for x, y, i in wall_points]
    i = [i for x, y, i in wall_points]

    for index, (x, y, i) in enumerate(wall_points):
        print(f'Index: {index}')
        if i > 15:
            color = 'green'
        else:
            color = 'red'
        ax.scatter(x, y, marker='o', color=color)
    plot_background(ax, True)
    plt.show()'''


def plot_background(ax_obj, limits = True):
    x = [370, 480, 590, 700, 810, 920, 1030, 1140]
    y = [165, 330, 480, 630, 780, 930, 1080, 1300, 1520]

    for i in x:
        for j in y:
            ax_obj.plot(i, j, marker='+', color='grey', markersize=5)

    ax_obj.axvline(x=stage_x, color = 'grey', linestyle='--', label='Stage Edge')
    ax_obj.axvline(x=room_x, color = 'grey', linestyle='--')
    ax_obj.axhline(y=room_y, color = 'grey', linestyle='--')  

    if limits:
        ax_obj.set_xlim(0, room_x)
        ax_obj.set_ylim(0, room_y+27)
        ax_obj.set_aspect('equal')
        ax_obj.set_xticks(x)
        ax_obj.set_xticklabels(x)
        for label in ax_obj.get_xticklabels():
            label.set_rotation(-90)
            label.set_horizontalalignment('right')
        ax_obj.set_yticks(y)


#%%
def wall_distributions(wall_data, ax):
    r = wall_data['r']
    mu, std = norm.fit(r)
    #print(f'mu: {mu}, std: {std}')
    x = np.linspace(mu-5*std, mu+5*std, 1000)
    p = norm.pdf(x, mu, std)
    #x = x - mu
    #r = r - mu

    ax.hist(r, bins = 20, density = True, alpha = 0.4, color='r')
    ax.scatter(x, p, c='black')

    ax.title.set_text(f'Wall distribution, mu: {mu:.2f}, std: {std:.2f}')



#%%
for directory in directories:
    run_directories(directory)
#%%
run_directories(directory_180_0)










#%%
'''
        ax0.imshow(walls_df['mask'][0], cmap='grey', origin='lower')
        ax1.imshow(walls_df['mask'][0], cmap='grey', origin='lower')
        plot_background(ax0, data, data_points)
        plot_background(ax1, data, data_points)

        l_data_copy = data[0:16].copy()
        l_data_copy.index = l_data_copy['theta (rad)']
        r_data_copy = data[16:32].copy()
        r_data_copy.index = r_data_copy['theta (rad)']

        for (x, y) in data_points:
            l_data_x = l_data_copy[f'x.{x}.{y}']
            l_data_y = l_data_copy[f'y.{x}.{y}']
            for index in l_data_copy.index:
                rad_index = np.rad2deg(index)

                if l_data_copy[f'r.{x}.{y}'][index] != 10000:
                    if left_diffs[f'diff_{x}_{y}'][rad_index] == 0:
                        ax0.scatter(l_data_x[index], l_data_y[index], c='r', marker='x')
                    else:
                        ax0.scatter(l_data_x[index], l_data_y[index], c='grey', marker='x')

            r_data_x = r_data_copy[f'x.{x}.{y}']
            r_data_y = r_data_copy[f'y.{x}.{y}']
            for index in r_data_copy.index:
                rad_index = np.rad2deg(index)
                if r_data_copy[f'r.{x}.{y}'][index] != 10000:
                    if right_diffs[f'diff_{x}_{y}'][rad_index] == 0:
                        ax1.scatter(r_data_x[index], r_data_y[index], c='b', marker='x')
                    else:
                        ax1.scatter(r_data_x[index], r_data_y[index], c='grey', marker='x')

        
        plt.show()
'''





'''
            
            fig, [ax, ax1, ax2] = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(filename)
            data_copy = data.copy()
            
            data_copy.loc[0:15, 'theta (rad)'] = data_copy.loc[0:15, 'theta (rad)'] + np.deg2rad(offset)
            data_copy.loc[16:31, 'theta (rad)'] = data_copy.loc[16:31, 'theta (rad)'] - np.deg2rad(offset)

            left_diffs = compare_data(data_copy[0:16], left_detectors_df, left_walls_df, left_origin, data_points, ax2)
            right_diffs = compare_data(data_copy[16:32], right_detectors_df, right_walls_df, right_origin, data_points, ax2)
            all_diffs = pd.concat([left_diffs, right_diffs], axis=1)

            plot_background(ax, data_copy, data_points)
            plot_diffs(ax, left_diffs, data_points, 'left')
            plot_diffs(ax, right_diffs, data_points, 'right')

            #mu, std, q = custom_hist(all_diffs.values.flatten(), ax1)
            lmu, lstd, lq = custom_hist(left_diffs.values.flatten(), ax1)
            rmu, rstd, rq = custom_hist(right_diffs.values.flatten(), ax1)
            #custom_plot(all_diffs.values.flatten(), ax1, 100)

            plt.show()

            #print(f'Filename: {filename}, Points: {q}, Offset: {offset}, mu: {mu}, std: {std}')
            print(f'Offset: {offset}  |  Left Points: {lq:.1f}, Left mu: {lmu:.1f}, Left std: {lstd:.1f}  |  Right Points: {rq:.1f}, Right mu: {rmu:.1f}, Right std: {rstd:.1f}')
'''
# %%
def return_data(data: pd.DataFrame):
    #pattern = r"^(?P<prefix>corrected_diff|r|i|x|y|diff)_X(?P<xvalue>\d+)Y(?P<yvalue>\d+)$"
    pattern = r"^(?P<prefix>corrected_diff_|r|i|x|y|diff_).(?P<xvalue>\d+).(?P<yvalue>\d+)$"
    pattern2 = r"^(?P<prefix>corrected_diff).(?P<xvalue>\d+).(?P<yvalue>\d+)$"
    data_points = []
    for column in data.columns[4::]:
        match = re.match(pattern, column)
        match2 = re.match(pattern2, column)
        if match:
            prefix = match.group('prefix')
            x = int(match.group('xvalue'))
            y = int(match.group('yvalue'))
            if (x, y) not in data_points and (x, y) != (0, 0):
                data_points.append((x, y))
        
        elif match2:
            prefix = match2.group('prefix')
            x = int(match2.group('xvalue'))
            y = int(match2.group('yvalue'))

            if (x,y) in data_points:
                data[f'corrected_diff_.{x}.{y}'] = data[f'corrected_diff.{x}.{y}']
                data.drop(columns=[f'corrected_diff.{x}.{y}'], inplace=True)

        else:
            print(f'No match found for {column}')
        
    return data_points


for filename in os.listdir(directory):
    if filename.endswith("data.csv") and filename.startswith('new'):
        data = pd.read_csv(os.path.join(directory, filename))
        data_points = return_data(data)

        data.to_csv(os.path.join(directory, f'{filename}'), index=True)
        print(f'Saved {filename}new to {directory}')

# %%

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
def plot_background(ax_obj, data: pd.DataFrame, data_points = None, limits = True):
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


for filename in os.listdir(directory):
    if filename.endswith("data.csv"):
        #fig, [ax, ax1, ax2] = plt.subplots(1, 3, figsize=(18, 6))
        #fig.suptitle(filename)

        data = pd.read_csv(os.path.join(directory, filename))
        return_attributes(filename, data)
        data_points = return_data(data)


        left_origin = (data['x_origin'][0], data['y_origin'][0])
        right_origin = (data['x_origin'][17], data['y_origin'][17])

        left_detectors_df = adjust_detector_masks(detector_df, left_origin)
        right_detectors_df = adjust_detector_masks(detector_df, right_origin)

        left_walls_df = error.generate_distances(walls_df, left_detectors_df, left_origin, False, True)
        right_walls_df = error.generate_distances(walls_df, right_detectors_df, right_origin, False, True)

        left_detectors_df = error.generate_distances(obj_df, left_detectors_df, left_origin, False)
        right_detectors_df = error.generate_distances(obj_df, right_detectors_df, right_origin, False)

        print(f'\nFilename: {filename}')
        for offset in range(-10, 11, 1):
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
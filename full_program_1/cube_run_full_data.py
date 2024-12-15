import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import math as math
import walls as walls
import error as error
from scipy.stats import norm
import re 
import time
from typing import Any, Tuple, Union
import inspect
from matplotlib.markers import MarkerStyle

# Read in the data
Timing_enabled = False
directory = 'C:/Users/james/OneDrive/Desktop/lidar_code/full_data'
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
    pattern = r"^(?P<prefix>corrected_diff|r|i|x|y|diff)_X(?P<xvalue>\d+)Y(?P<yvalue>\d+)$"
    data_points = []
    for column in data.columns[4::]:
        match = re.match(pattern, column)
        if match:
            prefix = match.group('prefix')
            x = int(match.group('xvalue'))
            y = int(match.group('yvalue'))

            if (x, y) not in data_points:
                data_points.append((x, y))
        else:
            print(f'No match found for {column}')
        
    return data_points

@time_function
def adjust_detector_masks(detector_df_input: pd.DataFrame, origin: Tuple) -> pd.DataFrame:
    try:
        detector_df = detector_df_input.copy()
        x_bound = int(detector_df_input.origin_x - origin[0])
        y_bound = int(detector_df_input.origin_y - origin[1])

        detector_df['mask'] = detector_df['mask'].apply(lambda mask: mask[int(y_bound):int(y_bound+room_y), int(x_bound):int(x_bound+room_x)])

        return detector_df
    except (InputTypeError, ValueError) as e:
        print(f"Input validation error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def compare_data(data: pd.DataFrame, detector_df: pd.DataFrame, origin: Tuple, data_points: list) -> None:
    data = data.copy()
    data['theta (deg)'] = data['theta (rad)'].apply(lambda x: np.rad2deg(x))
    detector_df['theta (deg)'] = detector_df['theta (rad)'].apply(lambda x: np.rad2deg(x))
    data.set_index('theta (deg)', inplace=True)
    detector_df.set_index('theta (deg)', inplace=True)

    results_df = pd.DataFrame()
    results_df['theta_deg'] = data['theta (rad)'].apply(lambda x: np.rad2deg(x))
    results_df.set_index('theta_deg', inplace=True)

    for (x, y) in data_points:
        try:
            data_r_col = data[f'r_X{x}Y{y}']
            data_r_col_nonzero = data_r_col[data_r_col != 10000]
            det_col = detector_df[f'r_({x}, {y})']
            det_col_nonzero = det_col[det_col != 0]

            results_df[f'diff_{x}_{y}'] = np.zeros_like(data_r_col)

            for index in det_col_nonzero.index:
                if index in data_r_col_nonzero.index:
                    diff = data_r_col_nonzero[index]-det_col_nonzero[index]
                    results_df.at[index, f'diff_{x}_{y}'] = diff
                    #print(f"diff: {np.abs(diff):.1f} Data: {data_r_col_nonzero[index]:.0f}  Detector: {det_col_nonzero[index]:.0f}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    return results_df

@time_function
def plot_background(ax_obj, data: pd.DataFrame, data_points = None, limits = True):
    x = [370, 480, 590, 700, 810, 920, 1030, 1140]
    y = [165, 330, 480, 630, 780, 930, 1080, 1300, 1520]

    for i in x:
        for j in y:
            if data_points is not None and (i, j) in data_points:
                ax_obj.plot(i, j, marker='o', fillstyle='none', color='black', markersize=7)
            else:
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
            ax_obj.scatter(x, y, c=color, marker=MarkerStyle('o', fillstyle=f'{l_r}'), edgecolors='k', markersize=7)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

obj_df = error.initialise_objects()
detector_df = error.initialise_detector_masks()

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        data = pd.read_csv(os.path.join(directory, filename))
        return_attributes(filename, data)
        data_points = return_data(data)

        left_origin = (data['x_origin'][0], data['y_origin'][0])
        right_origin = (data['x_origin'][17], data['y_origin'][17])
        left_detectors_df = adjust_detector_masks(detector_df, left_origin)
        right_detectors_df = adjust_detector_masks(detector_df, right_origin)

        left_detectors_df = error.generate_distances(obj_df, left_detectors_df, left_origin, False)
        right_detectors_df = error.generate_distances(obj_df, right_detectors_df, right_origin, False)

        left_diffs = compare_data(data[0:16], left_detectors_df, left_origin, data_points)
        right_diffs = compare_data(data[16:32], right_detectors_df, right_origin, data_points)

        plot_background(ax, data, data_points)
        plot_diffs(ax, left_diffs, data_points, 'left')
        plot_diffs(ax, right_diffs, data_points, 'right')
        plt.show()
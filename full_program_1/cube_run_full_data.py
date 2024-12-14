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

def create_subplot(identifier):
    global plot_counter
    fig, ax = plt.subplots()
    ax_d[identifier] = ax
    plot_counter += 1
    plt.close(fig)

def arrange_subplots(rows, cols):
    fig, axs = plt.subplots(rows, cols)
    for i, (identifier, ax) in enumerate(ax_d.items()):
        row = i//cols
        col = i%cols
        if rows == 1:
            axs_i = axs[col]
        elif cols == 1:
            axs_i = axs[row]
        else:
            axs_i = axs[row, col]
        
        for line in ax.get_lines():
            if line.get_linestyle() == '--' and len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]:
                axs_i.axvline(x=line.get_xdata()[0], color = line.get_color(), linestyle=line.get_linestyle(), label = line.get_label())
            elif line.get_linestyle() == '--' and len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]:
                axs_i.axhline(y=line.get_ydata()[0], color = line.get_color(), linestyle=line.get_linestyle(), label = line.get_label())
            else:
                axs_i.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), color = line.get_color(), label = line.get_label())
        
        for collection in ax.collections:
            if isinstance(collection, LineCollection):
                axs_i.add_collection(collection)
            else:
                offsets = collection.get_offsets()
                paths = collection.get_paths()
                if len(paths) > 0:
                    marker = paths[0].vertices
                else:
                    marker = 'o'
                color = collection.get_facecolors()[0]
                if  np.array_equal(color, [0, 0, 0, 1]):
                    marker = 'x'
                elif np.array_equal(color, [0, 0, 1, 1]):
                    marker = 'o'
                elif np.array_equal(color, [0.50196078, 0.50196078, 0.50196078, 1.]):
                    marker = '+'

                axs_i.scatter(offsets[:, 0], offsets[:, 1], color=collection.get_facecolors()[0], marker=marker, label=collection.get_label())
        
        #for bar in ax.patches:
            #axs_i.bar(bar.get_x(), bar.get_height(), color = bar.get_facecolor(), label = bar.get_label())
        
        axs_i.set_xticks(ax.get_xticks())
        for label in axs_i.get_xticklabels():
            label.set_rotation(-90)
            label.set_horizontalalignment('right')
        axs_i.set_yticks(ax.get_yticks())
        axs_i.set_title(ax.get_title())
        axs_i.set_xlabel(ax.get_xlabel())
        axs_i.set_ylabel(ax.get_ylabel())
        axs_i.set_xlim(ax.get_xlim())
        axs_i.set_ylim(ax.get_ylim())
        axs_i.legend()
    plt.tight_layout()
    plt.show()

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
def plot_background(identifier, data: pd.DataFrame, data_points = None, limits = True):
    if identifier not in ax_d:
        raise KeyError(f"Identifier {identifier} not found in ax_d.")
    
    ax_obj = ax_d[identifier]
    x = [370, 480, 590, 700, 810, 920, 1030, 1140]
    y = [165, 330, 480, 630, 780, 930, 1080, 1300, 1520]

    for i in x:
        for j in y:
            if data_points is not None and (i, j) in data_points:
                ax_obj.scatter(i, j, marker='x', color='black', s=7)
            else:
                ax_obj.scatter(i, j, marker='+', color='grey', s=5)

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


for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        data = pd.read_csv(os.path.join(directory, filename))
        return_attributes(filename, data)

        data_points = return_data(data)

        identifier = f'{filename}plot'
        create_subplot(identifier)
        plot_background(identifier, data, data_points, limits = bool)


arrange_subplots(2, 3)
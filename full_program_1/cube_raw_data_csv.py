import program_init as init
import data_processing as dp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os as os

directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
#folder_path = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/Code/csv_folder"
folder_path = "C:/Users/james/OneDrive/Desktop/lidar_code/csv_folder"
data_frames = init.get_dataframes(directory, 'cube', False)
centre_of_lidars = 685

def identify(Data, identifier):
    if identifier == 'grid':
        Data.offset = 0
        Data.separation = 180
    elif identifier == '90sep':
        Data.offset = 0
        Data.separation = 90
    elif identifier == '90sep.24offset':
        Data.offset = 24
        Data.separation = 90
    elif identifier == '29offset':
        Data.offset = 29
        Data.separation = 180
    elif identifier == '24offset':
        Data.offset = 24
        Data.separation = 180
    elif identifier == '90sep.29offset':
        Data.offset = 29
        Data.separation = 90
    else:
        Data.offset = 0
        Data.separation = 180

def calc_x(Data):
    if Data.leddar == 'Left':
        x = centre_of_lidars - (Data.separation/2)
    elif Data.leddar == 'Right':
        x = centre_of_lidars + (Data.separation/2)
        
    return x

for i in range(len(data_frames)):
    df = data_frames[i]
    Data = dp.File_Data(df)
    identify(Data, Data.identifier)
    for j in range(i, len(data_frames)):
        if i != j:
            df2 = data_frames[j]
            Data2 = dp.File_Data(df2)
            identify(Data2, Data2.identifier)
            if Data.pair_label == Data2.pair_label:
                left_r_arr = Data.r_arr
                left_i_arr = Data.i_arr
                left_theta_arr = Data.theta_arr[::-1] - np.deg2rad(Data.offset)
                left_x = calc_x(Data)
                left_x = np.full_like(left_r_arr, int(left_x))
                right_r_arr = Data2.r_arr
                right_i_arr = Data2.i_arr
                right_theta_arr = Data2.theta_arr[::-1] + np.deg2rad(Data2.offset)
                right_x = calc_x(Data2)
                right_x = np.full_like(right_r_arr, int(right_x))

                # Combine all arrays into a dictionary
                data = {
                    'r': np.concatenate((left_r_arr, right_r_arr)),
                    'i': np.concatenate((left_i_arr, right_i_arr)),
                    'theta (rad)': np.concatenate((left_theta_arr, right_theta_arr)),
                    'x_origin': np.concatenate((left_x, right_x)),
                    'y_origin': np.zeros(32)
                }

                # Create DataFrame
                df = pd.DataFrame(data)
                df.attrs['filename'] = Data.file_name
                df.attrs['identifier'] = Data.identifier

                # Uncomment to save to CSV
                full_path = os.path.join(folder_path, Data.pair_label)
                df.to_csv(f'{full_path}.csv', index=False)
                print(f'Saved {Data.pair_label}.csv')



import program_init as init
import data_processing as dp
import matplotlib.pyplot as plt
import simulated_data_v2 as sim
import numpy as np

directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
data_frames = init.get_dataframes(directory)
#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
fig, ax = plt.subplots(figsize=(10, 10))

def main(df, ax):
    file_name, theta_arr, r_arr, i_arr = dp.get_values(df, +3)
    #dp.plot_polar(ax, file_name, theta_arr, r_arr, i_arr)
    x, y = dp.polar2cartesian(theta_arr, r_arr)
    #dp.plot_cartesian(ax, file_name, x, y)


    for i in range(0, 16):
        distance = r_arr[i]
        start_angle = (90 - (48/2)) + i*3
        end_angle = start_angle + 3
        mid_angle = (start_angle + end_angle) / 2
        mid_angle_rad = np.deg2rad(mid_angle)

        # Plot the dotted line in the middle of each detector's view
        ax.plot([0, 0 + distance * np.cos(mid_angle_rad)], 
                [0, 0 + distance * np.sin(mid_angle_rad)], 
                 linestyle='dotted', color='blue')


        ax.plot([distance * np.cos(mid_angle_rad-np.deg2rad(1.5)), distance * np.cos(mid_angle_rad+np.deg2rad(1.5))], 
                [distance * np.sin(mid_angle_rad-np.deg2rad(1.5)), distance * np.sin(mid_angle_rad+np.deg2rad(1.5))], 
                linestyle='dotted', color='red')


main(data_frames[3], ax)

ax.set_ylim(0, 200)
ax.set_xlim(-100, 100)
plt.show()








'''
for i, df in enumerate(data_frames):
    main()
    plt.show()'''
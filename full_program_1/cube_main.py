import program_init as init
import data_processing as dp
import matplotlib.pyplot as plt
import simulated_data_v2 as sim
import numpy as np

directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
data_frames = init.get_dataframes(directory, 'cube', False)
#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#fig, ax = plt.subplots(figsize=(10, 10))

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




for i in range(len(data_frames)):
    df = data_frames[i]
    data = dp.File_Data(df)
    print(data.file_name)
    print(data.obj_x)
    print(data.obj_y)
    print(data.leddar)
    print(data.identifier)
    x_arr, y_arr = dp.polar2cartesian(theta_arr, r_arr)

print((x_arr))



# Setup measurment grid
x = [370, 480, 590, 700, 810, 920, 1030, 1140]
y = [330, 480, 630, 780, 930, 1080, 1305, 1520]
room = np.zeros((1545, 1520))

#setup the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
for i in x:
    for j in y:
        ax.plot(i, j, '+', color='grey')
ax.plot(775, 0, 'o', color = 'blue')
ax.plot(595, 0, 'o', color = 'blue')
ax.axvline(x=223, color = 'grey', linestyle='dotted')

ax.scatter(x_arr+775, y_arr, color = 'red', label='Distance (cm)')

#leftLidar = sim.Sensor(595, 0, 0, room)
#rightLidar = sim.Sensor(775, 0, 0, room)


#ax.set_xlim(0, 1545)
#ax.set_ylim(0, 1520)
ax.set_title(file_name)
plt.show()
import program_init as init
import data_processing as dp
import matplotlib.pyplot as plt
import simulated_data_v2 as sim
import numpy as np

directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
data_frames = init.get_dataframes(directory, '24offset', False)
centre_of_lidars = 685

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

def plot_background(ax, sensor1, sensor2, Data):
    x = [370, 480, 590, 700, 810, 920, 1030, 1140]
    y = [330, 480, 630, 780, 930, 1080, 1305, 1520]
    room = np.zeros((1545, 1520))

    for i in x:
        for j in y:
            ax.plot(i, j, '+', color='grey')
    ax.plot(sensor2.x, 0, 'o', color = 'blue')
    ax.plot(sensor1.x, 0, 'o', color = 'blue')
    ax.plot(int(Data.obj_x), int(Data.obj_y), 'o', color = 'green', markersize=10)
    ax.axvline(x=223, color = 'grey', linestyle='dotted')

    ax.set_xlim(0, 1545)
    ax.set_ylim(0, 1520)
    plt.show()

class Sensor:
    def __init__(self, Data):
        self.y = 0
        self.x = self.calc_x(Data)
        self.polar = self.calc_sep(Data)
        self.cartesian = self.cartesian()
    
    def calc_x(self, Data):
        if Data.leddar == 'Left':
            x = centre_of_lidars - (Data.separation/2)
        elif Data.leddar == 'Right':
            x = centre_of_lidars + (Data.separation/2)
        
        return x

    def calc_sep(self, Data):
        if Data.leddar == 'Left':
            sep = -Data.offset
        elif Data.leddar == 'Right':
            sep = Data.offset
        angles = Data.theta_arr[::-1] + np.deg2rad(sep)
        #angles = angles[-16:]
        polar = (angles, Data.r_arr)

        return polar

    def cartesian(self):
        theta = (self.polar[0])
        r = self.polar[1]
        x = y = np.zeros(len(r))
        for i in range(len(r)):
            if r[i] == 0:
                x[i] = self.x
                y[i] = 0
            else:
                x[i] = self.x + r[i] * np.cos(theta[i])
                y[i] = self.y + r[i] * np.sin(theta[i])
        cartesian = (x, y)

        return cartesian
    
    def plot_cartesian(self, ax):
        for i in range(len(self.polar[0])):
            ax.plot([self.x, self.x + self.polar[1][i] * np.cos(self.polar[0][i])], 
                    [0, self.polar[1][i] * np.sin(self.polar[0][i])], 
                    linestyle='dotted', color='blue')
            ax.scatter(self.x + self.polar[1][i] * np.cos(self.polar[0][i]), self.polar[1][i] * np.sin(self.polar[0][i]), color='red')

    def plot_polar(self, ax):
        ax.plot(self.polar[0], self.polar[1], label='Distance (cm)')
        for i in range(0, 16):
            distance = self.polar[1][i]
            start_angle = (90 - (48/2)) + i*3
            end_angle = start_angle + 3
            mid_angle = (start_angle + end_angle) / 2
            mid_angle_rad = np.deg2rad(mid_angle)

            # Plot the dotted line in the middle of each detector's view
            ax.plot([self.x, self.x + distance * np.cos(mid_angle_rad)], 
                    [0, 0 + distance * np.sin(mid_angle_rad)], 
                    linestyle='dotted', color='blue')


            ax.plot([distance * np.cos(mid_angle_rad-np.deg2rad(1.5)) + self.x, distance * np.cos(mid_angle_rad+np.deg2rad(1.5)) + self.x], 
                    [distance * np.sin(mid_angle_rad-np.deg2rad(1.5)), distance * np.sin(mid_angle_rad+np.deg2rad(1.5))], 
                    linestyle='dotted', color='red')
        ax.set_title(Data.file_name)
        ax.legend()

    def plot_blank(self, ax):
        length = 10000
        for i in range(len(self.polar[0])):
            ax.plot([self.x, self.x + length * np.cos(self.polar[0][i])], 
                    [0, length * np.sin(self.polar[0][i])], 
                    linestyle='dotted', color='gray')


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
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_title(str(Data.pair_label))
                sensor = Sensor(Data)
                sensor.plot_blank(ax)
                sensor.plot_cartesian(ax)
                sensor2 = Sensor(Data2)
                sensor2.plot_blank(ax)
                sensor2.plot_cartesian(ax)
                plot_background(ax, sensor, sensor2, Data)
                ax.legend()
                plt.show()

'''
for i in range(len(data_frames)):
    fig, ax = plt.subplots(figsize=(10, 10))
    df = data_frames[i]
    Data = dp.File_Data(df)
    identify(Data, Data.identifier)
    print(Data.pair_label)

    sensor = Sensor(Data)
    sensor.plot_blank(ax)
    sensor.plot_cartesian(ax)
    ax.set_xlim(0, 1545)
    ax.set_ylim(0, 1520)
    #plt.show()

'''









'''
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
'''

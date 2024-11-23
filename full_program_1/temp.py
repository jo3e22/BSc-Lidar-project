import program_init as init
import data_processing as dp
import matplotlib.pyplot as plt
import simulated_data_v2 as sim
import numpy as np
import math

directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
data_frames = init.get_dataframes(directory, 'grid', False)
centre_of_lidars = 685

class Sensor:
    def __init__(self, Data):
        self.x = self.calc_x(Data)
        self.y = 0
        self.angle_offset = self.calc_sep(Data)
        self.fov = 48
        self.num_detectors = 16
        self.detector_angle = self.fov / self.num_detectors
        self.pixel_map = np.zeros((1545, 1520))
        self.detectors = [self.create_detector_mask(90 - self.fov/2 + i*self.detector_angle - self.angle_offset, 
                                                    90 - self.fov/2 + (i+1)*self.detector_angle - self.angle_offset) 
                          for i in range(self.num_detectors)]
        self.data_arr = np.zeros((2, 16))  # will change def scan to use this array that can be called in new function locate_object.

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
        polar = (angles, Data.r_arr)
        return sep

    def create_detector_mask(self, start_angle, end_angle):
        height, width = np.shape(self.pixel_map)
        mask = np.zeros((height, width))
        start_angle_rad = np.deg2rad(start_angle)
        end_angle_rad = np.deg2rad(end_angle)
        
        for i in range(height):
            for j in range(width):
                angle = math.atan2(i - self.y, j - self.x)
                if start_angle_rad <= angle <= end_angle_rad:
                    mask[i, j] = 1
        return mask

    def get_points(self):
        points = []
        for i in range(self.num_detectors):
            distance = self.data_arr[0, i]
            angle = 90 - self.fov/2 + i*self.detector_angle - self.angle_offset
            angle_rad = np.deg2rad(angle)
            points.append((self.x + distance * np.cos(angle_rad), self.y + distance * np.sin(angle_rad)))
        return points

    def plot_detector(self, ax, data_arr):
        for i in range(self.num_detectors):
            distance = self.data_arr[0, i]
            start_angle = 90 - self.fov/2 + i*self.detector_angle - self.angle_offset
            end_angle = start_angle + self.detector_angle
            mid_angle = (start_angle + end_angle) / 2
            mid_angle_rad = np.deg2rad(mid_angle)
            
            # Plot the dotted line in the middle of each detector's view
            ax.plot([self.x, self.x + distance * np.cos(mid_angle_rad)], 
                    [self.y, self.y + distance * np.sin(mid_angle_rad)], 
                    linestyle='dotted', color='blue')
            
            '''
            # Highlight the bar at each distance the detector returns
            ax.plot([self.x + distance * np.cos(mid_angle_rad)], 
                    [self.y + distance * np.sin(mid_angle_rad)], 
                    marker='o', color='red')
            '''

            ax.plot([self.x + distance * np.cos(mid_angle_rad-np.deg2rad(1.5)), self.x + distance * np.cos(mid_angle_rad+np.deg2rad(1.5))], 
                    [self.y + distance * np.sin(mid_angle_rad-np.deg2rad(1.5)), self.y + distance * np.sin(mid_angle_rad+np.deg2rad(1.5))], 
                    linestyle='dotted', color='red')

    def blind_plot(self, r_arr, mask):
        for i in range(self.num_detectors):
            detector_mask = self.detectors[i]
            distance = r_arr[i]
            if distance == 0:
                distance = 10000
            height, width = np.shape(self.pixel_map)

            for y in range(height):
                for x in range(width):
                    if detector_mask[y, x] == 1:
                        if np.sqrt((y - self.y)**2 + (x - self.x)**2) <= distance:
                            mask[y, x] = 1
        return mask



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
                #fig, ax = plt.subplots(figsize=(10, 10))
                #ax.set_title(str(Data.pair_label))
                leftLidar = Sensor(Data)
                rightLidar = Sensor(Data2)
                leftLidar.mask = leftLidar.blind_plot(r_arr=Data.r_arr[::-1], mask=np.zeros((1545, 1520)))
                rightLidar.mask = rightLidar.blind_plot(r_arr=Data2.r_arr[::-1], mask=leftLidar.mask)

                reversed_mask = rightLidar.mask[:, ::-1]
                plt.imshow(reversed_mask, cmap='gray', interpolation='none', origin='lower')
                plt.scatter(int(Data.obj_x), int(Data.obj_y), color='red')
                plt.title(str(Data.pair_label))
                plt.xlim(0, 1545)
                plt.ylim(0, 1520)
                plt.show()


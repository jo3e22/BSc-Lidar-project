import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class Environment:
    """
    Environment Class created an array of the input dimensions + a surrounding border of set reflectivity.
    Inputs:
        Required:
            x_length: determines the x axis length
            y_length: determines the y axis height
        Optional:
            border_reflectivity: determines how reflective the border will be
    """
    def __init__(self, x_length, y_length, border_reflectivity = 0.5):
        self.x_length = int(x_length)  # Dimensions in m
        self.y_length = int(y_length)  # Dimensions in m
        self.border_intensity = int(255*border_reflectivity)
        self.border_width = 2

        self.map = np.zeros((  (self.x_length)+self.border_width, (self.y_length)+self.border_width  ), dtype = np.uint8)
        self.map[:self.border_width, :] = self.border_intensity  # Top border
        self.map[-self.border_width:, :] = self.border_intensity  # Bottom border
        self.map[:, :self.border_width] = self.border_intensity  # Left border
        self.map[:, -self.border_width:] = self.border_intensity  # Right border

    def add_object(self, object_mask):
        if np.shape(self.map) == np.shape(object_mask):
            self.map = self.map + object_mask
        else:
            self.map = self.map
            print('Error adding object to environment')

class Sensor:
    def __init__(self, x, y, angle_offset, pixel_map):
        self.x = x
        self.y = y
        self.angle_offset = angle_offset
        self.fov = 48
        self.num_detectors = 16
        self.detector_angle = self.fov / self.num_detectors
        self.pixel_map = pixel_map
        self.detectors = [self.create_detector_mask(90 - self.fov/2 + i*self.detector_angle - self.angle_offset, 
                                                    90 - self.fov/2 + (i+1)*self.detector_angle - self.angle_offset) 
                          for i in range(self.num_detectors)]
        self.distances_arr = np.zeros((16, 1))  # will change def scan to use this array that can be called in new function locate_object.

    def create_detector_mask(self, start_angle, end_angle):
        height, width = np.shape(self.pixel_map)
        mask = np.zeros((height, width))
        start_angle_rad = np.deg2rad(start_angle)
        end_angle_rad = np.deg2rad(end_angle)
        
        for i in range(height):
            for j in range(width):
                angle = np.atan2(i - self.y, j - self.x)
                if start_angle_rad <= angle <= end_angle_rad:
                    mask[i, j] = 1
        return mask

    def scan(self, pixel_map):
        distances_arr = []
        for detector in self.detectors:
            masked_array = pixel_map * detector
            non_zero_points = np.argwhere(masked_array > 0)
            distances = np.sqrt((non_zero_points[:, 0] - self.y)**2 + (non_zero_points[:, 1] - self.x)**2)
            if len(distances) > 0:
                min_distance = np.min(distances)
                distances_arr.append(min_distance)
            else:
                distances_arr.append(1000)
        return distances_arr

    def plot_detector(self, ax, distances_arr):
        for i, distance in enumerate(distances_arr):
            start_angle = 90 - self.fov/2 + i*self.detector_angle - self.angle_offset
            end_angle = start_angle + self.detector_angle
            mid_angle = (start_angle + end_angle) / 2
            mid_angle_rad = np.deg2rad(mid_angle)
            
            # Plot the dotted line in the middle of each detector's view
            ax.plot([self.x, self.x + distance * np.cos(mid_angle_rad)], 
                    [self.y, self.y + distance * np.sin(mid_angle_rad)], 
                    linestyle='dotted', color='blue')
            
            # Highlight the bar at each distance the detector returns
            ax.plot([self.x + distance * np.cos(mid_angle_rad)], 
                    [self.y + distance * np.sin(mid_angle_rad)], 
                    marker='o', color='red')

class Object:
    def __init__(self, x_start, y_start, x_vel, y_vel, shape, reflectivity = 0.75):
        self.x_start = x_start
        self.y_start = y_start
        self.x_vel = x_vel
        self.y_vel = y_vel
        
        self.shape = shape
        self.reflectivity = reflectivity
        self.mask = self.create_mask()

    def create_mask(self):
        '''
        if type(self.shape) == np.array:
            return self.shape
        elif shape == 'square':
            mask = np.full((5, 5), (self.reflectivity*255), dtype=uint8) 
            return mask
        else:
            return np.zeros((5,5))
        '''
        return np.full((5, 5), (self.reflectivity*255), dtype=np.uint8)

def update(frame, object_name, environment_name, sensors, ax):
    def add_small_mask_to_large_mask(large_mask, small_mask, y, x):
        # Ensure x and y are integers
        x = int(x)
        y = int(y)

        # Calculate the top-left corner of the small mask in the large mask
        top_left_x = x - small_mask.shape[0] // 2
        top_left_y = y - small_mask.shape[1] // 2

        # Wrap around if the object goes beyond the boundaries
        top_left_x = top_left_x % large_mask.shape[0]
        top_left_y = top_left_y % large_mask.shape[1]

        # Add the small mask to the large mask
        for i in range(small_mask.shape[0]):
            for j in range(small_mask.shape[1]):
                large_mask[(top_left_x + i) % large_mask.shape[0], (top_left_y + j) % large_mask.shape[1]] = small_mask[i, j]

        return large_mask

    large_mask = np.copy(environment_name.map)
    small_mask = object_name.mask
    object_x = (frame * object_name.x_vel) % (environment_name.x_length * 2) - environment_name.x_length + object_name.x_start
    object_y = (frame * object_name.y_vel) % (environment_name.y_length * 2) - environment_name.y_length + object_name.y_start

    environment_map = add_small_mask_to_large_mask(large_mask, small_mask, object_x, object_y)

    # Clear the previous plot
    ax.clear()
    ax.imshow(environment_map, cmap='gray', interpolation='none', origin='lower')
    ax.axis('off')
    ax.set_title('Environment Map')

    # Update sensors
    for Sensor in sensors:
        min_distances = Sensor.scan(environment_map)
        Sensor.plot_detector(ax, min_distances)
    
    # Return the updated artists
    return ax.images +ax.lines

def locate_object():
    '''
    will use this function to take the detector readings and locate an object.
    '''

if __name__ == "__main__":
    # Create environment
    Room = Environment(200, 200)

    Target = Object(00, 170, 5, -1, 'square')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10)) 

    # Create sensors
    Sensor1 = Sensor(110, Room.border_width, -10, Room.map)
    Sensor2 = Sensor(90, Room.border_width, 10, Room.map)

    # Keep a reference to the animation object
    anim = FuncAnimation(fig, update, fargs=(Target, Room, [Sensor1, Sensor2], ax), frames=300, interval=1, blit=True)
    plt.show()

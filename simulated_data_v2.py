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

        self.map = np.zeros((  (self.x_length*10)+self.border_width, (self.y_length*10)+self.border_width  ), dtype = np.uint8)
        self.map[:self.border_width, :] = self.border_intensity  # Top border
        self.map[-self.border_width:, :] = self.border_intensity  # Bottom border
        self.map[:, :self.border_width] = self.border_intensity  # Left border
        self.map[:, -self.border_width:] = self.border_intensity  # Right border

        self.fig, self.ax = plt.subplots()

    def plot_map(self):
        self.ax.imshow(self.map, cmap='gray', interpolation='none', origin='lower')
        plt.axis('off')
        plt.title('Environment Map')
        plt.show()

    def add_object(self, object_mask):
        self.map = self.map + object_mask

class Sensor:
    def __init__(self, x, y, angle_offset, color):
        self.x = x
        self.y = y
        self.angle_offset = angle_offset
        self.fov = 48
        self.num_detectors = 16
        self.detector_angle = fov / num_detectors
        self.color = color

        self.detectors = [Wedge((x, y), 100, 90 - fov/2 + i*self.detector_angle + angle_offset, 90 - fov/2 + (i+1)*self.detector_angle + angle_offset, color='gray', alpha=0.5) for i in range(num_detectors)]
        self.highlighted = [Wedge((x, y), 0, 90 - fov/2 + i*self.detector_angle + angle_offset, 90 - fov/2 + (i+1)*self.detector_angle + angle_offset, color=color, alpha=0.5) for i in range(num_detectors)]

    def add_to_plot(self, ax):
        for wedge in self.detectors + self.highlighted:
            ax.add_patch(wedge)

    def update(self, object_position, object_distance):
        for i, (detector, highlight) in enumerate(zip(self.detectors, self.highlighted)):
            start_angle = 90 - self.fov/2 + i * self.detector_angle + self.angle_offset
            end_angle = start_angle + self.detector_angle

            angle_to_object = np.degrees(np.arctan2(object_distance, object_position - self.x))
            if start_angle <= angle_to_object <= end_angle:
                highlight.set_radius(np.sqrt(object_distance**2 + (object_position - self.x)**2))
                highlight.set_color(self.color)
            else:
                highlight.set_radius(0)
                highlight.set_color('gray')

    def reset(self):
        for highlight in self.highlighted:
            highlight.set_radius(0)
            highlight.set_color('gray')

def init():
    obj_sc.set_offsets(np.c_[[], []])
    sensor_1.reset()
    sensor_2.reset()
    return sensor_1.detectors + sensor_1.highlighted + sensor_2.detectors + sensor_2.highlighted + [obj_sc]

def update(frame):
    global object_position
    object_position = (frame * object_speed) % (room_size * 2) - room_size

    # Object's position (moving side to side)
    object_positions = [[object_position, object_distance]]

    # Update sensors
    sensor_1.update(object_position, object_distance)
    sensor_2.update(object_position, object_distance)

    # Update plot
    obj_sc.set_offsets(object_positions)
    return sensor_1.detectors + sensor_1.highlighted + sensor_2.detectors + sensor_2.highlighted + [obj_sc]


if __name__ == "__main__":
    # Create environment
    room = Environment(20, 20)

    target_mask = np.zeros_like(room.map)
    target_mask[50:60, 50:60] = 255
    room.add_object(target_mask)

    room.plot_map()

    # Create sensors
    sensor_1 = Sensor(0, 0, 0, 'red')


    # Add sensors to plot
    sensor_1.add_to_plot(room.ax)
    sensor_2.add_to_plot(room.ax)

    '''
    # Keep a reference to the animation object
    anim = FuncAnimation(room.fig, update, init_func=init, frames=300, interval=100)
    plt.show()
    '''

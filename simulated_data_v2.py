import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.animation import FuncAnimation

class Sensor:
    def __init__(self, x, y, angle_offset, color):
        self.x = x
        self.y = y
        self.angle_offset = angle_offset
        self.fov = 48
        self.num_detectors = 16
        self.detector_angle = fov / num_detectors
        self.color = color

        self.detectors = [Wedge((x, y), 200, 90 - fov/2 + i*self.detector_angle + angle_offset, 90 - fov/2 + (i+1)*self.detector_angle + angle_offset, color='gray', alpha=0.5) for i in range(num_detectors)]
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

# Simulation parameters
fov = 48  # Field of view in degrees
num_detectors = 16  # Number of detectors
room_size = 20  # Room size in meters
object_speed = 0.5  # meters per second
object_distance = 10.0  # meters from the sensors
object_width = 1.0  # meters

# Create sensors
sensor_1 = Sensor(1.75/2, 0, 5, 'red')
sensor_2 = Sensor(-1.75/2, 0, -5, 'blue')

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-room_size / 2, room_size / 2)
ax.set_ylim(0, room_size)
ax.set_aspect('equal')

# Add sensors to plot
sensor_1.add_to_plot(ax)
sensor_2.add_to_plot(ax)

obj_sc = ax.scatter([], [], c='red', s=100, label='Object')

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

# Keep a reference to the animation object
anim = FuncAnimation(fig, update, init_func=init, frames=300, interval=100)
plt.show()

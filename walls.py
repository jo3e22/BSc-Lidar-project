import numpy as np
from sklearn.linear_model import LinearRegression

def find_segment(points, epsilon=10):
    # Find the segments in the points
    segments = []
    segment = []
    for i in range(len(points) - 2):
        x1, y1 = points[i]
        x2, y2 = points[i + 2]
        x0, y0 = points[i + 1]

        # Calculate the distance from point (x0, y0) to the line defined by (x1, y1) and (x2, y2)
        distance = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if distance < epsilon:
            segment.append((x1, y1))
            segment.append((x0, y0))
            segment.append((x2, y2))
            segments.append(segment)
            segment = []

    return segments

def grow_segment(points, segment, epsilon=10):
    while True:
        # Fit a line to the current segment
        segment_array = np.array(segment)
        X = segment_array[:, 0].reshape(-1, 1)
        y = segment_array[:, 1]
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        # Find points that are close to the line and within 3 * epsilon of any point in the segment
        new_points = []
        for point in points:
            x, y = point
            distance_to_line = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
            if distance_to_line < epsilon:
                for seg_point in segment:
                    seg_x, seg_y = seg_point
                    distance_to_segment_point = np.sqrt((x - seg_x)**2 + (y - seg_y)**2)
                    if distance_to_segment_point < 3 * epsilon:
                        new_points.append(point)
                        break

        if not new_points:
            break

        segment.extend(new_points)

    return segment

# Example usage
points = [(0, 0), (1, 1), (2, 2), (3, 5), (4, 4), (5, 5), (6, 6)]
initial_segment = find_segment(points)[0]
grown_segment = grow_segment(points, initial_segment)
print(grown_segment)


def check_arcs(point1, point1):
    d1, r1 = point1
    d2, r2 = point2
    distances = [d1, d2]

    min_distance_idnex = distances.index(min(distances))
    max_distance_index = distances.index(max(distances))
    
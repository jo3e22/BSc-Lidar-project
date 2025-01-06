import numpy as np

def find_segment(points, epsilon=10):
    # Find the segments in the points
    segments = []
    segment = []
    points = np.array(points)
    points = points[points[:, 2] > 0]
    for i in range(len(points) - 2):
        x1, y1, i1 = points[i]
        x2, y2, i2 = points[i + 2]
        x0, y0, i0 = points[i + 1]

        if min(i0, i1, i2) > max(i0, i1, i2)/3:

            # Calculate the distance from point (x0, y0) to the line defined by (x1, y1) and (x2, y2)
            distance = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            if distance < epsilon:
                segment.append((x1, y1))
                segment.append((x0, y0))
                segment.append((x2, y2))
                segments.append(segment)
                segment = []

    return segments

def join_connected_segments(segments_input):
    segments = np.copy(segments_input)
    connected_segments = []

    for i in range(len(segments)-1):
        connected_segment = segments[i]
        for j in range(i+1, len(segments)):
            if segments[j] not in connected_segment:
                test = []
                combined = np.concatenate((segments[i], segments[j]))
                #print(f'Combined test: {combined}')
                for (x, y) in combined:
                    if (x, y) not in test:
                        test.append((x, y))
                if len(test) < len(combined):
                    connected_segment += segments[j]
                    #print(f'Connected segment test: {segments[j]}')
        connected_segments.append(connected_segment)
        connected_segment = []
    
    return connected_segments

def grow_segment(segments, points, epsilon=10):
    segments = np.copy(segments)
    for i, segment in enumerate(segments):
        # Grow the segment by adding points to it
        x1, y1 = segment[0]
        x2, y2 = segment[-1]
        for point in points:
            x0, y0, i0 = point
            if min(i0, x0, y0) > max(i0, x0, y0)/3:
                distance = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if distance < epsilon:
                    segments[i].append((x0, y0))

    return segments



def segment_lines(seg):
    x = [point[0] for point in seg]
    y = [point[1] for point in seg]
    b, a = best_fit(x, y)
    x_fit = np.linspace(min(x), max(x), 20)
    y_fit = a*x_fit + b
    seg_lines =(x_fit, y_fit)
    
    return seg_lines


def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    #print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b
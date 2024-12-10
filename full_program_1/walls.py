import numpy as np

def find_segment(points, epsilon=10):
    # Find the segments in the points
    segments = []
    segment = []
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

def join_connected_segments0(segments_input, b_input):
    segments = np.copy(segments_input)
    b = np.copy(b_input)
    connected_segments = []

    for i in range(len(segments)-1):
        connected_segment = np.copy(segments[i])
        for j in range(i+1, len(segments)):
            if segments[j] not in connected_segment and np.abs(b[i] - b[j]) < 0.3:
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

def join_connected_segments(segments_input, b_input):
    segments = np.copy(segments_input)
    b = np.copy(b_input)
    connected_segments = []
    
    for i in range(len(segments)):
        connected_segment = [segments[i]]
        for j in range(i+1, len(segments)):
            if any(np.array_equal(segments[j], seg) for seg in connected_segment):
                continue
            if np.abs(b[i] - b[j]) < 0.3:
                connected_segment.append(segments[j])
            connected_segments.append(connected_segment)
        
        # Merge segments with shared elements
        i = 0
        while i < len(connected_segments):
            j = i + 1
            while j < len(connected_segments):
                if any(np.array_equal(seg, test) for seg in connected_segments[i] for test in connected_segments[j]):
                    combined = np.array(connected_segments[i] + connected_segments[j])
                    unique_elements = []
                    for element in combined:
                        if not any(np.array_equal(element, e) for e in unique_elements):
                            unique_elements.append(element)
                    connected_segments[i] = unique_elements
                    connected_segments.pop(j)
                else:
                    j += 1
            i += 1

        return connected_segments

def combine_lists_of_tuples(lists):
    def find_common_groups(lists):
        groups = []
        for sublist in lists:
            found = False
            for i, group in enumerate(groups):
                if any(item in group for item in sublist):
                    group.update(sublist)
                    found = True
                    break
            if not found:
                groups.append(set(sublist))
        return groups
    combined_groups = find_common_groups(lists)
    return [list(group) for group in combined_groups]

def combine_lists_of_tuples_b(lists, b):
    def find_common_groups(lists, b):
        groups = []
        for i, sublist in enumerate(lists):
            found = False
            for j, group in enumerate(groups):
                if any(item in group for item in sublist) and np.abs(b[i] - b[j]) < 0.3:
                    group.update(sublist)
                    found = True
                    break
            if not found:
                groups.append(set(sublist))
        return groups
    combined_groups = find_common_groups(lists, b)
    return [list(group) for group in combined_groups]

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
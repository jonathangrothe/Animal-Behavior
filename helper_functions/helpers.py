import numpy as np

def get_estimated_angle(a1,a2):
    qa1 = 1
    qa2 = 1    
    adj_a1 = a1
    adj_a2 = a2
    if adj_a1 > np.pi/2: 
        adj_a1 = np.pi - a1
        qa1 = 2
    if adj_a1 < 0:
        if adj_a1 < -np.pi/2:
            adj_a1 = adj_a1 + np.pi
            qa1 = 3
        else: 
            adj_a1 = -a1
            qa1 = 4
    if adj_a2 > np.pi/2: 
        adj_a2 = np.pi - a2
        qa2 = 2
    if adj_a2 < 0:
        if adj_a2 < -np.pi/2:
            adj_a2 = adj_a2 + np.pi
            qa2 = 3
        else: 
            adj_a2 = -a2
            qa2 = 4
    t = 0
    quad_diff = qa2-qa1
    if quad_diff == 0:
        t = 1
    if np.abs(quad_diff) == 2:
        t = 3
    if np.abs(quad_diff) == 3:
        t = 4
    if quad_diff == -1:
        if qa1 == 3:
            t = 4
        else:
            t = 2
    if quad_diff == 1:
        if qa1 == 2:
            t = 4
        else:
            t = 2

    a_bigger = max(adj_a1,adj_a2)
    a_smaller = min(adj_a1,adj_a2)
    b_bigger = np.pi/2-a_bigger
    b_smaller = np.pi/2-a_smaller

    aprox_angle = a_bigger - a_smaller
    if t == 1:
        aprox_angle = a_smaller + b_bigger + np.pi/2
    if t == 2:
        aprox_angle = a_smaller + a_bigger
    if t == 4:
        aprox_angle = b_bigger + b_smaller

    return aprox_angle
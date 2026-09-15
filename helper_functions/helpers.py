'''
A few helper functions that took more lines of code then I wanted to put in the other files
'''
import numpy as np
import math

def get_estimated_angle(a1,a2):
    # Mostly a relic of the bifurcation angle saga
    '''
    A function that takes the angles of two lines and gets the estimated bifurcation angle,
    Is meant to be able to handle negatives and angles that are more than 2pi away from each other
    Parameters:
    a1: the angle of line 1 (radians)
    a2: the angle of line 2 (radians)
    Returns:
    aprox_angle: the approximate bifurcation angle
    '''
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


def trajectory_grid(xpoints,ypoints,N,agg=1,indi=0):
    '''
    A function which produces a 101x101 grid of angles corresponding to the expected direction of the agent at each whole number point (in the 100x100 space)
    This is done by considering two situations: the complete aggregation situation and the nearest target situation and then combining them
    parameters:
        xpoints: the x positions of the targets
        ypoints: the y positions of the targets
        N: the number of points per side
        agg: the proportion for the completely aggregated solution
        indi: the proportion for the nearest target solution
    returns:
        angles: a list of size 101*101 with the angles for the vector field arranged from (0,0) -> (100,0) -> ... -> (100,100)
    '''
    
    angles = []
    factor = 100/(N-1)
    ntarg = len(xpoints)
    for p in range(N**2):
        new_x = p % N * factor
        new_y = p // N * factor
        #print(f"p: {p}, x: {new_x}, y: {new_y}")
        new_x_sum = 0
        new_y_sum = 0
        best_angle = 0
        best_dist = 100
        for t in range(ntarg):
            x_diff = xpoints[t]- new_x
            y_diff = ypoints[t] - new_y
            new_dist = np.sqrt((x_diff)**2+(y_diff)**2)
            #new_weight = np.exp(-1*new_dist/100)
            new_angle = np.atan2(y_diff,x_diff)
            new_x_sum += 1/ntarg * np.cos(new_angle)
            new_y_sum += 1/ntarg * np.sin(new_angle)
            if new_dist < best_dist:
                best_dist = new_dist
                best_angle = new_angle
        agg_angle = np.atan2(new_y_sum,new_x_sum)
        angles_unwrapped = np.unwrap([agg_angle,best_angle])
        combined_angle = agg*angles_unwrapped[0]+indi*angles_unwrapped[1]
        angles.append(combined_angle)
        #print(f"angle at {new_x,new_y}: {new_overall_angle}")
    return(angles)

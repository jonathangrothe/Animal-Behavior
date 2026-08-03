import numpy as np
import math

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

def get_estimated_sigma(p0,p1,p2,h0,h1,h2,N):
    # we assume we are at p0 and want to find how attractive the aggregated bump between p1 and p2 is
    x = p0[0]
    y = p0[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    angle_1 = np.atan2(y1-y,x1-x)
    angle_2 = np.atan2(y2-y,x2-x)
    theor_neuron1 = angle_1*N/(2*np.pi)
    theor_neuron2 = angle_2*N/(2*np.pi)
    delta_ang1 =  np.abs(theor_neuron1-round(theor_neuron1))
    delta_ang2 = np.abs(theor_neuron2-round(theor_neuron2))

    for item in range(int(theor_neuron2),math.ceil(theor_neuron1)):
        # calculate the theoretical sigma needed for the total attraction at this neuron to be greater than the total attraction at the other neuron
        # Note: we assume that we are at p0, but in reality we are almost there and if sigma is small then we are pretty sensitive to where we are in the ring attractor angle-wise,
        # ie: if the angle to p0 lands at neuron 30.5, then that is a much lower attraction than if it was at neuron 30 (but also we need to account for total area under the curve...)
        # to account for total area under the curve, we should probably just do the same thing (but it will be much more complicated to calculate sigma...)
        # Come back to this math, I think its not quite right
        neuron_angle = item*2*np.pi/100
        delta_ang1 = np.abs(theor_neuron1-neuron_angle)
        delta_ang2 = np.abs(theor_neuron2-neuron_angle)
        log_h0 = math.log(h0)
        log_h1 = math.log(h1)
        log_h2 = math.log(h2)
        print(f"log h0: {log_h0}, log h1: {log_h1}, log h2: {log_h2}, num: {-0.5*(delta_ang1**2+delta_ang2**2)}")
        sigma_est = np.sqrt((-0.5*(delta_ang1**2+delta_ang2**2))/(log_h0-log_h1-log_h2))
        print(f"angle: {neuron_angle}, neuron: {item}, sigma: {sigma_est}")

    theor_ang_delta = np.abs(angle_1-angle_2)
    j = np.log(h0/(h1+h2))
    theor_sigma = np.sqrt((-0.5*(theor_ang_delta/2)**2)/j) # instead of just calculating it at the theor_ang_delta, we need to calculate it everywhere
    print(f"theoretical sigma: {theor_sigma}, difference from neuron and true angles: 1: {delta_ang1}, 2: {delta_ang2}")
    return theor_sigma

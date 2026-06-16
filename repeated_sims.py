# file which contains a function for running the simulation many times (with the option to alter the settings each time)
# and returns data on 'success' and trajectories
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
from scipy.stats import binom

def sample_sims(bp, changing_params, n_samples, include_trajs=False, include_activity=False):
    '''
    A function for running a variety of consecutive simulations over changing values of specified parameters

    parameters:
    bp: (base parameters) a dictionary which contains the base values to run the simulation on. It will contain an entry for every parameter in simulate_ring_attractor in sim_ra
    changing_params: a dictionary which contains the parameters that are going to be changed throughout the simulations as keys, 
                     and a list of values those parameters will take as values.
    n_samples: the number of samples to run for each specific set of parameters
    include_trajs: whether or not to return the trajectories
    include_activity: whether or not to return the sum of activity

    returns: 
    target_list: a list of size n_samples * len(changing_params) which target the agent reaches each time, if the agent fails to reach a target it is -1
    time_list: a list of size n_samples * len(changing_params) which contains the time the agent reaches the target, if no target is reached it is the number of timesteps + 1
    decision_points: a list of lists where each list contains all the time steps when a decision is made, as according to the get_bifurcation_times function in sim_met
    decision_pos: a list of list of tuples in which each list contains the positions (as x,y tuples) of the agent at the bifurcation time, as found using get_bifurcation_times in sim_met
    sum_activity_list: TO BE DELETED, the sum of all neuron activity, I thought this would be interesting but I haven't found a great use for it
    activity_df: a dataframe which contains all the neuron activity merged, each row is a neuron, simulations are differentiated by sets of 100 rows
                 (ie: rows 0-99 is one simulation, 100-199 is the next, etc.). Each column corresponds to a time step, and each entry is the corresponding neuron's
                 activity at that time. NaN entrys are present when a simulation ends before that time step. 
    x_list: a list of size n_samples * len(changing_params) which contains numpy arrays which contain all the x positions for that simulation, 
            each array is one dimensional and will have entries equal to the number of time steps that simulation runs for
    y_list: a list of size n_samples * len(changing_params) designed the same as x_list but containing y positions instead of x positions
    headings_list: a list of size n_samples * len(changing_params) designed the same as x_list and y_list but containing headings (in polar coordinates)
    '''
    time_list = []
    target_list = []
    activity_list = [] 
    decision_points = []
    decision_pos = []
    activity_df = None
    x_list = []
    y_list = []
    headings_list = []
    n_sweeps = -1
    for key in changing_params.keys():
        curr_len = len(changing_params[key])
        if n_sweeps > 0 and curr_len != n_sweeps:
            raise ValueError("Changing parameters are different dimensions")
        n_sweeps = curr_len
    for index in range(n_sweeps):
        for param in changing_params.keys():
            bp[param] = changing_params[param][index]
            print(f"param: {param}, value: {changing_params[param][index]}")
        for sample in range(n_samples):
                print(f"sample: {sample}")
                headings, xPos, yPos, targetsx, targetsy, activity = sim_ra.simulate_ring_attractor(bp['N'],bp['L'],bp['T'],bp['ntargets'],bp['nagents'],bp['allocentricFlag'],
                                                                                 bp['periodicFlag'],bp['rEgo'],bp['rEgoTarget'],bp['Egonumber'],bp['distf'],
                                                                                 bp['adistf'],bp['J'],bp['beta'],bp['h0'],bp['h_b'],bp['dt'],bp['v0'],bp['v0t'],
                                                                                 bp['sigma'],bp['hColl'],bp['rColl'],bp['initialx'],bp['initialy'],bp['initialxt'],bp['initialyt'],
                                                                                 False,True)
                
                if bp['ntargets'] > 0:
                    target_reached, time_reached = sim_met.get_destination_metrics(xPos,yPos,targetsx,targetsy)
                    target_list.append(target_reached)
                    time_list.append(time_reached)

                if include_trajs:
                    xpos_1d = xPos.ravel()
                    ypos_1d = yPos.ravel()
                    x_list.append(xpos_1d)
                    y_list.append(ypos_1d)
                    headings_list.append(headings)

                if include_activity: # inefficient I think 
                    for neuron in range(np.shape(activity)[0]):
                        activity_list.append(activity[neuron,0,:])

    if include_activity:
        activity_df = pd.DataFrame(activity_list)
    
    return target_list, time_list, decision_points, decision_pos, activity_df, x_list, y_list, headings_list


def boundary_search(bp,base_min,base_max,sample_size,min_search,param,boundary_prob):
    '''
    A function that performs a binary search to find a boundary value for a given parameter. 
    Boundary value is defined as a value in which the agent reaches a target with between boundary_prob and 1-boundary_prob probability

    params:
    bp: set of base parameters for the simulate_ring_attractor function
    base_min: the absolute minimum for the parameter we are searching over, 
              it is important that this (and the base_max) is a decent guess with a little bit of room so the search doesn't get stuck in an extreme
    base_max: the absolute maximum for the parameter we are searching over
    sample_size: number of samples to take
    min_search: whether we are searching for a minimum or a maximum, tells us which direction to move in
    param: the parameter we are searching over
    boundary_prob: the probability of success we are looking for, because the simulation has stochasticity we don't want it to be too high, as that could lead to getting stuck
                  
    returns:
    boundary: the boundary value we found after our search
    boundary_range: the range of the region that was being searched when the desired value was found
    '''
    ntargets = bp['ntargets']
    h0_bool = param == 'h0'
    boundary = 0
    found_range = 0
    found = False
    min_val = base_min
    max_val = base_max
    value = (min_val+max_val)/2
    bp[param] = search_helper(ntargets, h0_bool, value)
    while not found:
        target_list = []
        print(f"{param}: {bp[param]}")
        print(f"min: {min_val}")
        print(f"max: {max_val}")
        for s in range(sample_size):
            headings, xPos, yPos, targetsx, targetsy, activity = sim_ra.simulate_ring_attractor(bp['N'],bp['L'],bp['T'],bp['ntargets'],bp['nagents'],bp['allocentricFlag'],
                                                                                                bp['periodicFlag'],bp['rEgo'],bp['rEgoTarget'],bp['Egonumber'],bp['distf'],
                                                                                                bp['adistf'],bp['J'],bp['beta'],bp['h0'],bp['h_b'],bp['dt'],bp['v0'],bp['v0t'],
                                                                                                bp['sigma'],bp['hColl'],bp['rColl'],bp['initialx'],bp['initialy'],bp['initialxt'],bp['initialyt'],
                                                                                                False,True)
            target_reached, time_reached = sim_met.get_destination_metrics(xPos,yPos,targetsx,targetsy) # another vote to rework this function
            target_list.append(target_reached)
        n_reached = 0
        for item in target_list: # could try and make this into a more general loop which loops for a different metric
            if item != -1:
                n_reached += 1
        print(f"n reached: {n_reached}")
        if n_reached < int(boundary_prob*sample_size):
            if min_search:
                min_val = value
            else:
                max_val = value
            value = (min_val + max_val)/2
            bp[param] = search_helper(ntargets, h0_bool, value)
        if n_reached > int((1-boundary_prob)*sample_size):
            if min_search:
                max_val = value
            else:
                min_val = value
            value = (min_val + max_val)/2
            bp[param] = search_helper(ntargets, h0_bool, value)
        if int(boundary_prob*sample_size) < n_reached < int((1-boundary_prob)*sample_size):
            curr_range = max_val - min_val
            value = (min_val+max_val)/2
            boundary = value
            found_range = curr_range
            break
    return boundary, found_range


def search_helper(nt,h0_bool,val):
    '''
    A helper function for the binary search function that handles the fact that we need a list of h0 values when searching over that parameter
    could be expanded to encode an uneven parameter in h0 ? 

    params: 
    nt: number of targets
    h0_bool: if the parameter we are interested in is h0
    val: the value we will be using in the search

    returns: 
    h0_list: the list of h0s we need for the simulation
    val: the value of the parameter when we aren't using h0
    '''
    if h0_bool:
        h0_list = []
        for i in range(nt):
            h0_list.append(val)
        return h0_list
    return val

def target_dist_test(min_val,max_val,nt,target_list,direction):
    '''
    a helper function for doing boundary search with three (+ ?) targets that sees how close to evenly distributed they are
    if they are close to evenly distributed we're good; good enough for now is just reaching every target
    '''
    sample_size = len(target_list)
    for i in range(nt):
        times_reached = target_list.count(i)
        if times_reached <= sample_size*0.1:
            if direction=='min':
                x=0
    return (min_val + max_val)/2

def target_range_test(min_val,max_val,target_list,direction,boundary_prob):
    sample_size = len(target_list)
    n_reached = 0
    for item in target_list:
        if item != -1:
            n_reached += 1
    print(f"n reached: {n_reached}")
    if n_reached < int(boundary_prob*sample_size):
        if direction=='min':
            min_val = value
        else:
            max_val = value
        return min_val, max_val
    if n_reached > int((1-boundary_prob)*sample_size):
        if direction=='max':
            max_val = value
        else:
            min_val = value
        return min_val, max_val
    if int(boundary_prob*sample_size) < n_reached < int((1-boundary_prob)*sample_size):
        curr_range = max_val - min_val
        value = (min_val+max_val)/2
        return value
    
def area_helper(area, param, param_list):
    if param == 'sigma':
        h0_list = []
        for item in param_list: 
            expon = -(item**2*2)
            inv = 1/expon
            integral = inv*(np.exp(np.pi/expon)-1)
            h0 = area/2*integral
            h0_list.append(h0)
        return h0_list
    if param == 'h0':
        s = 1
        tol=1e-10
        sigma_list = []
        for item in param_list:
            for i in range(10):
                fs   = f(area, s, item)
                fps  = fprime(s, item)
                if abs(fs) < tol:
                    if s > 1:
                        print(f"warning, h0 is too small to support this area")
                        sigma_list.append(1)
                        break
                    sigma_list.append(s)
                    break
                if fps == 0:
                    raise ValueError("Derivative is zero — Newton's method failed. Try a different starting guess.")
                s = s - fs / fps
                if s <= 0:
                    s = 0.25
        return sigma_list

def f(a,s,h):
    return 4*h*s**2-4*h*s**2*np.exp(-np.pi/(2*s**2))-a

def fprime(s,h):
    return 4 * h * (2 * s * (1 - np.exp(-np.pi/(2*s**2))) - (np.pi / s) * np.exp(-np.pi/(2*s**2)))

        




    


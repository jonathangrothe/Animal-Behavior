# file which contains a function for running the simulation many times (with the option to alter the settings each time)
# and returns data on 'success' and trajectories
import numpy as np
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met

def sample_sims(bp,changing_params,n_samples,include_trajs,include_activity):
    '''
    A function for running a variety of consecutive simulations over changing values of specified parameters

    Parameters:
        bp: (base parameters) a dictionary which contains every parameter in the simulate_ring_attractor function except plot, stop, and stopping distance as a key,
            and an approriate value for that parameter as the corresponding value
        changing_params: a dictionary which contains the parameters that are going to be changed throughout the simulations as keys, 
                        and a list of values those parameters will take as values. The lists of values must all be the same size.
        n_samples: an integer which represents the number of samples to run for each unique set of parameters
        include_trajs: a boolean which controls whether or not to return the trajectories
        include_activity: a boolean which controls whether or not to return the sum of activity

    Returns: 
        target_list: a list of size (n_samples * len(changing_params)) which contains which target the agent reaches each during each individual simulation, 
                     if the agent fails to reach a target it is -1
        time_list: a list of size (n_samples * len(changing_params)) which contains the time the agent takes to reach the target, 
                   if no target is reached it is the number of timesteps + 1
        activity_list: a list of size (n_samples * len(changing_params)) which contains a numpy array of size (bp['N'] x tsteps) 
                       which contains the activity for each neuron at each timestep for each individual simulation
        x_list: a list of size (n_samples * len(changing_params)) which contains 1d numpy arrays which contain all the x positions for the agent in each individual simulation
        y_list: a list of size (n_samples * len(changing_params)) which contains 1d numpy arrays which contain all the y positions for the agent in each individual simulation
        headings_list: a list of size (n_samples * len(changing_params)) which contains 1d numpy arrays which contain all the headings for the agent in each individual simulation
    '''
    time_list = []
    target_list = []
    activity_list = [] 
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

                if include_activity:
                    activity_list.append(activity[:,0,:])

    return target_list, time_list, activity_list, x_list, y_list, headings_list


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

    def search_helper(nt,h0_bool,val):
        '''
        A helper function that handles the fact that we need a list of h0s to run the simulation
        '''
        if h0_bool:
            h0_list = []
            for i in range(nt):
                h0_list.append(val)
            return h0_list
        return val
    
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
            target_reached, time_reached = sim_met.get_destination_metrics(xPos,yPos,targetsx,targetsy)
            target_list.append(target_reached)
        n_reached = 0
        for item in target_list:
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


def area_helper(area, param, param_list):
    '''
    A function that when given a certain area and either sigma or h0, calculates the value of the other parameter that will give the desired area
    parameters:
    area: the area of the input bump
    param: the name of the parameter that we are given
    param_list: a list of param values
    returns: 
    h0_list: returns a list of h0 values if the input list was of sigma values
    sigma_list: returns a list of sigma values if the input list was of h0 values
    '''
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

        




    


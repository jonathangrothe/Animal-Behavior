# file which contains a function for running the simulation many times (with the option to alter the settings each time)
# and returns data on 'success' and trajectories
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
from scipy.stats import binom

def sample_sims(bp, changing_params, n_samples, include_trajs=[False, "scatter"], include_activity=False):
    '''
    bp: (base parameters) a dictionary which contains the base values to run the simulation on. It will contain an entry for every parameter in the simulation
    changing_params: a dictionary which contains the parameters that are going to be changed throughout the simulations as keys, 
                    and a list of values those parameters will take as values.
    n_samples: the number of samples to run for each specific set of parameters
    PARAMETERS TO ADD: 
    success_metric: list of success metrics we can get (min/sum, correct/reach percentage, time to target, time to decision)
    track_trajectories: boolean that if True collects data on all the trajectories
    returns:
    metrics: dictionary of metrics?
    trajectories: returns all the trajectories in two dataframes, x_trajs_df and y_trajs_df for easy indexable access to both rows and columns going forward
                (we need to be able to split samples based on rows, and aggregate the position over each time)
    '''
    # initialize all the stuff I want to collect
    # to do: create a warning if including trajectory and including activity when changing parameters 
    # (b/c they are meant to only aggregate over samples of the same exact simulation settings)
    if include_trajs[0] or include_activity:
        for param in changing_params.keys():
            param_value_list = changing_params[param]
            if len(param_value_list) > 1:
                print("warning: taking trajectories or neuron activity over different simulation settings")
    success_list = []
    time_list = []
    target_list = []
    activity_list = [] 
    sum_activity_list = []
    decision_points = []
    decision_pos = []
    activity_df = None
    x_list = []
    y_list = []
    for param in changing_params.keys():
        param_value_list = changing_params[param]
        for value in param_value_list:
            bp[param] = value
            print(f"param: {value}")
            for sample in range(n_samples):
                print(f"sample: {sample}")
                headings, xPos, yPos, targetsx, targetsy, activity = sim_ra.simulate_ring_attractor(bp['N'],bp['L'],bp['T'],bp['ntargets'],bp['nagents'],bp['allocentricFlag'],
                                                                                 bp['periodicFlag'],bp['rEgo'],bp['rEgoTarget'],bp['Egonumber'],bp['distf'],
                                                                                 bp['adistf'],bp['J'],bp['beta'],bp['h0'],bp['h_b'],bp['dt'],bp['v0'],bp['v0t'],
                                                                                 bp['sigma'],bp['hColl'],bp['rColl'],bp['initialx'],bp['initialy'],bp['initialxt'],bp['initialyt'],
                                                                                 True,True)
                # Basic target and time metrics
                target_reached, time_reached, start = sim_met.get_destination_metrics(xPos,yPos,targetsx,targetsy)
                target_list.append(target_reached)
                time_list.append(time_reached)
                all_fail = target_list.count(-1) == len(target_list)

                # Bifurcation times and locations: 
                dec_points_list, dec_pos = sim_met.get_bifurcation_times(xPos[0,:],yPos[0,:]) #change this when we get more agents
                decision_points.append(dec_points_list)
                decision_pos.append(dec_pos)

                # Neuron activity metrics
                sum_activity, range_activity, var_activity, n_inhib_list, n_active_list, neuron_change_rates = sim_met.get_neuron_info(activity[:,0,:])
                sum_activity_list.append(sum_activity)

                # code for plotting neuron activity: 
                if include_trajs[0]:
                    xpos_1d = xPos.ravel()
                    ypos_1d = yPos.ravel()
                    if include_trajs[1] == 'heat':
                        x_list += list(xpos_1d)
                        y_list += list(ypos_1d)
                    if include_trajs[1] == 'scatter':
                        x_list.append(xpos_1d)
                        y_list.append(ypos_1d)

                if include_activity: # we also want to add a sum of activity list,
                    for neuron in range(np.shape(activity)[0]):
                        activity_list.append(activity[neuron,0,:])


    if include_activity:
        activity_df = pd.DataFrame(activity_list)
    
    if include_trajs[0]:
        if include_trajs[1] == 'heat':
            x_list = np.array(x_list)
            y_list = np.array(y_list)
    # change the return to take out range, add in bifurcation times (as df?)
    return success_list, target_list, time_list, decision_points, decision_pos, sum_activity_list, activity_df, x_list, y_list

def boundary_search(bp,base_min,base_max,sample_size,min_search,boundary_prob=0.05,n_in_bounds=5):
    '''
    A function that performs a modified binary search to find the target attractiveness value which produces a probability close to the boundary_prob
    Binary search until we are out of 0 and 1, because that will be the majority of cases, 
    once it finds a non 0 or 1 probability, assume that the linear range of the function is half the current range and extrapolate where boundary_prob would be on that curve
    then update the max and min around the predicted value, regardless of their comparisons to previous max and min. 
    TO UPDATE: keep range constant? Maybe at the decision point do a quick sweep to try and get an idea of how long the transition period is? 
    params:
    bp: set of base parameters for the simulate_ring_attractor function
    base_min: the absolute minimum for h0
    base_max: the absolute maximum for h0
    KEEP IN MIND THE MEAN OF BASE_MIN AND BASE_MAX MUST BE IN THE SUCCESS RANGE (or find a way to fix this later)
    sample_size: number of samples to take
    boundary_prob: the probability of success we are looking for, defaul to 0.05 to try and find right where it starts to fail
    n_in_bounds: number of times to search once we have reached a non 0 or 1 probability
    returns:
    boundary_list: a list which contains the smallest* and the largest h0 value where we get really close to this boundary_prob
    KEEP IN MIND: THIS DOESN'T NECESSARILY DO GREAT WITH DIPS, WHICH IS DEF A PROBLEM IF WE WANT TO RELY ON IT HEAVILY
    this should probably be used in combination with our sweeps to most effeciently get boundaries. 
    '''
    boundary_list = []
    found_range = []
    found = False
    min_val = base_min
    max_val = base_max
    value = (min_val+max_val)/2
    bp['h0'] = [value,value]
    while not found:
        target_list = []
        print(f"h0: {bp['h0']}")
        print(f"min: {min_val}")
        print(f"max: {max_val}")
        for s in range(sample_size):
            headings, xPos, yPos, targetsx, targetsy, activity = sim_ra.simulate_ring_attractor(bp['N'],bp['L'],bp['T'],bp['ntargets'],bp['nagents'],bp['allocentricFlag'],
                                                                                                bp['periodicFlag'],bp['rEgo'],bp['rEgoTarget'],bp['Egonumber'],bp['distf'],
                                                                                                bp['adistf'],bp['J'],bp['beta'],bp['h0'],bp['h_b'],bp['dt'],bp['v0'],bp['v0t'],
                                                                                                bp['sigma'],bp['hColl'],bp['rColl'],bp['initialx'],bp['initialy'],bp['initialxt'],bp['initialyt'],
                                                                                                False,True)
            target_reached, time_reached, start = sim_met.get_destination_metrics(xPos,yPos,targetsx,targetsy)
            target_list.append(target_reached)
        n_reached = 0
        for item in target_list:
            if item != -1:
                n_reached += 1
        print(f"n reached: {n_reached}")
        if n_reached < int(0.2*sample_size):
            if min_search:
                min_val = value
            else:
                max_val = value
            value = (min_val + max_val)/2
            bp['h0'] = [value,value]
        if n_reached > int(0.8*sample_size):
            if min_search:
                max_val = value
            else:
                min_val = value
            value = (min_val + max_val)/2
            bp['h0'] = [value,value]
        if int(0.2*sample_size) < n_reached < int(0.8*sample_size):
            # we know we're in the range, so if we our current range is appropriately small, we're good? with an error reported?
            curr_range = max_val - min_val
            value = (min_val+max_val)/2
            boundary_list.append(value)
            found_range.append(curr_range)
            break
    print(boundary_list)
    print(found_range)
    return boundary_list, found_range

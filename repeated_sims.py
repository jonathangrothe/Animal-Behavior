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
    range_activity_list = []
    range_argmin_list = []
    decision_points = []
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
                                                                                 False,True)
                # Basic target and time metrics
                target_reached, time_reached, start = sim_met.get_destination_metrics(xPos,yPos,targetsx,targetsy)
                target_list.append(target_reached)
                time_list.append(time_reached)
                all_fail = target_list.count(-1) == len(target_list)
                # avg_dists_timereached = sim_met.get_avg_distance(xPos, yPos, targetsx, targetsy, time_reached)
                # dist = min(avg_dists_timereached)
                # success_list.append(dist)

                # Neuron activity metrics
                sum_activity, range_activity, var_activity, n_inhib_list, n_active_list, neuron_change_rates = sim_met.get_neuron_info(activity[:,0,:])
                dec_start, dec_end = sim_met.get_decision_time(sum_activity)
                decision_points.append([dec_start,dec_end]) 
                sum_activity_list.append(sum_activity)
                range_activity_list.append(range_activity)
                range_argmin_list.append(np.argmin(range_activity[start:])+start)

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

    return success_list, target_list, time_list, decision_points, sum_activity_list, range_activity_list, range_argmin_list, activity_df, x_list, y_list

def boundary_search(bp,base_min,base_max,sample_size,boundary_prob=0.05):
    '''
    A function that does binary search to find the approximate appropriate target attractiveness space for a given set of parameters?
    Right now we need the starting value for the search to be succesful to mean something for both
    '''
    # when looking for mins, the first nonzero after we see a 0 we should lower the 0
    # when looking for maxs, the first nonzero after we see a 0 we should raise the 0
    boundary_list = []
    for i in range(2):
        print(i)
        found = False
        min_val = base_min
        max_val = base_max
        value = (min_val+max_val)/2
        bp['h0'] = [value,value]
        while not found:
            target_list = []
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
            if n_reached == 0:
                if i == 0:
                    min_val = value
                else:
                    max_val = value
                value = (min_val + max_val)/2
                bp['h0'] = [value,value]
                print(f"new h0: {bp['h0']}")
            if n_reached == sample_size:
                if i == 0:
                    max_val = value
                else:
                    min_val = value
                value = (min_val + max_val)/2
                bp['h0'] = [value,value]
                print(f"new h0: {bp['h0']}")
            if 0 < n_reached < sample_size:
                # assume we are in a linear zone between a and b
                if i == 0:
                    min_val = min_val
                found = True
                if i == 0:
                    boundary_list.append(min_val)
                    print(f"found, min is {min_val}")
                if i == 1:
                    boundary_list.append(max_val)
                    print(f"found, max is: {max_val}")             
    return boundary_list
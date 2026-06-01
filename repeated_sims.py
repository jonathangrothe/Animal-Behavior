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
    bp: (base parameters) a dictionary which contains the base values to run the simulation on. It will contain an entry for every parameter in the simulation
    changing_params: a dictionary which contains the parameters that are going to be changed throughout the simulations as keys, 
                    and a list of values those parameters will take as values.
    n_samples: the number of samples to run for each specific set of parameters
    include_trajs: whether or not to return the trajectories
    include_activity: whether or not to return the sum of activity

    returns: Each list that is returned is continuously appended to over the simulation, 
             so there will be n_samples consecutive lists which are results from the same settings, 
             and then the next list will be run with the corresponding settings in changing_params

    success_list: TO BE DELETED, EMPTY LIST RIGHT NOW, meant to be a list of boolean 0 if agent doesn't reach target 1 if agent does
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
    # initialize all the stuff I want to collect
    # to do: create a warning if including trajectory and including activity when changing parameters 
    # (b/c they are meant to only aggregate over samples of the same exact simulation settings)
    if include_trajs or include_activity:
        for param in changing_params.keys():
            param_value_list = changing_params[param]
            if len(param_value_list) > 1:
                print("warning: taking trajectories or neuron activity over different simulation settings")
    time_list = []
    target_list = []
    activity_list = [] 
    decision_points = []
    decision_pos = []
    activity_df = None
    x_list = []
    y_list = []
    headings_list = []
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
                
                # need a skip if there are no targets

                if bp['ntargets'] > 0:
                    # Basic target and time metrics
                    target_reached, time_reached, start = sim_met.get_destination_metrics(xPos,yPos,targetsx,targetsy)
                    target_list.append(target_reached)
                    time_list.append(time_reached)

                    # Bifurcation times and locations: 
                    dec_points_list, dec_pos = sim_met.get_bifurcation_times(xPos[0,:],yPos[0,:]) #change this when we get more agents
                    decision_points.append(dec_points_list)
                    decision_pos.append(dec_pos)

                # code for plotting neuron activity: 
                if include_trajs:
                    xpos_1d = xPos.ravel()
                    ypos_1d = yPos.ravel()
                    x_list.append(xpos_1d)
                    y_list.append(ypos_1d)
                    headings_list.append(headings)

                if include_activity: # we also want to add a sum of activity list,
                    for neuron in range(np.shape(activity)[0]):
                        activity_list.append(activity[neuron,0,:])

    if include_activity:
        activity_df = pd.DataFrame(activity_list)
    
    return target_list, time_list, decision_points, decision_pos, activity_df, x_list, y_list, headings_list

# could be good to get this to be able to search for a certain bifurcation angle ...
def boundary_search(bp,base_min,base_max,sample_size,min_search,param='h0',boundary_prob=0.2):
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
    param: the parameter we are searching over (usually h0)
    boundary_prob: the probability of success we are looking for, defaul to 0.05 to try and find right where it starts to fail
    returns:
    boundary_list: a list which contains the smallest* and the largest h0 value where we get really close to this boundary_prob
    boundary_range: the range of the region that was being searched when the desired value was found
    '''
    boundary_list = []
    found_range = []
    found = False
    min_val = base_min
    max_val = base_max
    value = (min_val+max_val)/2
    if param == 'h0':
        bp['h0'] = [value,value]
    if param == 'sigma':
        bp['sigma'] = value
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
        if n_reached < int(boundary_prob*sample_size):
            if min_search:
                min_val = value
            else:
                max_val = value
            value = (min_val + max_val)/2
            bp['h0'] = [value,value]
        if n_reached > int((1-boundary_prob)*sample_size):
            if min_search:
                max_val = value
            else:
                min_val = value
            value = (min_val + max_val)/2
            bp['h0'] = [value,value]
        if int(boundary_prob*sample_size) < n_reached < int((1-boundary_prob)*sample_size):
            # we know we're in the range, so if we our current range is appropriately small, we're good? with an error reported?
            curr_range = max_val - min_val
            value = (min_val+max_val)/2
            boundary_list.append(value)
            found_range.append(curr_range)
            break
    print(boundary_list)
    print(found_range)
    return boundary_list, found_range

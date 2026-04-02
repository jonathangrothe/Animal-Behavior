# file which contains a function for running the simulation many times (with the option to alter the settings each time)
# and returns data on 'success' and trajectories

import numpy as np
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met

def sample_sims(bp, changing_params, n_samples, slice=100):
    '''
    bp: (base parameters) a dictionary which contains the base values to run the simulation on. It will contain an entry for every parameter in the simulation
    changing_params: a dictionary which contains the parameters that are going to be changed throughout the simulations as keys, 
                    and a list of values those parameters will take as values.
    n_samples: the number of samples to run for each specific set of parameters
    slice: what number to slice positions into for trajectories (value of 100 means every 100th position)
    PARAMETERS TO ADD: 
    success_metric: list of success metrics we can get (min/sum, correct/reach percentage, time to target, time to decision)
    track_trajectories: boolean that if True collects data on all the trajectories
    returns:
    metrics: dictionary of metrics?
    trajectories: probably just return all the trajectories from every sample so that we can fiddle with what ones we want to show after the fact
    '''
    # initialize all the stuff I want to collect
    success_list = []
    x_trajs = []
    y_trajs = []
    for param in changing_params.keys():
        param_value_list = changing_params[param]
        for value in param_value_list:
            bp[param] = value
            for sample in range(n_samples):
                headings, xPos, yPos, targetsx, targetsy = sim_ra.simulate_ring_attractor(bp['N'],bp['L'],bp['T'],bp['ntargets'],bp['nagents'],bp['allocentricFlag'],
                                                                                 bp['periodicFlag'],bp['rEgo'],bp['rEgoTarget'],bp['Egonumber'],bp['distf'],
                                                                                 bp['adistf'],bp['J'],bp['beta'],bp['h0'],bp['h_b'],bp['dt'],bp['v0'],bp['v0t'],
                                                                                 bp['sigma'],bp['hColl'],bp['rColl'],bp['initialx'],bp['initialy'],bp['initialxt'],bp['initialyt'],
                                                                                 True,False)
                even = all(x == bp['h0'][0] for x in bp['h0'])
                best_index = -1
                if not even: 
                    best_index = bp['h0'].index(max(bp['h0']))
                success_measure = sim_met.get_min_distance(xPos, yPos, targetsx, targetsy, even, best_index)
                success_list.append(success_measure)
                x_sliced = xPos[:,::slice]
                y_sliced = yPos[:,::slice]
                x_trajs.append(x_sliced)
                y_trajs.append(y_sliced)
    return success_list, x_trajs, y_trajs
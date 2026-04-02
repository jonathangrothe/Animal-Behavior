# file which contains a function for running the simulation many times (with the option to alter the settings each time)
# and returns data on 'success' and trajectories

import numpy as np
import simulate_ringattractor as sim_ra

def sample_sims(bp, changing_params, n_samples):
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
    trajectories: probably just return all the trajectories from every sample so that we can fiddle with what ones we want to show after the fact
    '''
    # initialize all the stuff I want to collect
    success = []
    x_trajs = []
    y_trajs = []
    for param in changing_params.keys():
        param_value_list = changing_params[param]
        for value in param_value_list:
            bp[param] = value
            for sample in range(n_samples):
                headings, xPos, yPos, targetsx, targetsy = sim_ra.simulate_ring_attractor(bp['N'],bp['L'],bp['T'],bp['ntargets'],bp['nagents'],bp['allocentricFlag'],
                                                                                 bp['periodicflag'],bp['rEgo'],bp['rEgoTarget'],bp['Egonumber'],bp['distf'],
                                                                                 bp['adistf'],bp['J'],bp['beta'],bp['h0'],bp['h_b'],bp['dt'],bp['v0'],bp['v0t'],
                                                                                 bp['sigma'],bp['hColl'],bp['rColl'],bp['initialx'],bp['initialy'],bp['initialxt'],bp['initialyt'],
                                                                                 False,False)
                # get success metric
                # get trajectory metric
    return None
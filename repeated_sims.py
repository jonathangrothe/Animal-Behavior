# file which contains a function for running the simulation many times (with the option to alter the settings each time)
# and returns data on 'success' and trajectories
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met

def sample_sims(bp, changing_params, n_samples, slice=100, include_trajs=False, include_activity=False):
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
    trajectories: returns all the trajectories in two dataframes, x_trajs_df and y_trajs_df for easy indexable access to both rows and columns going forward
                (we need to be able to split samples based on rows, and aggregate the position over each time)
    '''
    # initialize all the stuff I want to collect
    img = None
    activity_df = None
    success_list = []
    time_list = []
    target_list = []
    if include_trajs == True:
        x_trajs = []
        y_trajs = []
    if include_activity == True:
        activity_list = []
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
                target_reached, time_reached, start = sim_met.get_destination_metrics(xPos,yPos,targetsx,targetsy)
                target_list.append(target_reached)
                time_list.append(time_reached)
                avg_dists_timereached = sim_met.get_avg_distance(xPos, yPos, targetsx, targetsy, time_reached)
                dist = min(avg_dists_timereached)
                # for now I think we're just interested in distance to a target, 
                # because we have success contained elsewhere
                '''
                even = all(x == bp['h0'][0] for x in bp['h0']) # REWORK THIS WHEN WE ADD AGENT ATTRACTIONS
                best_index = -1
                if not even: 
                    best_index = bp['h0'].index(max(bp['h0'])) # REWORK THIS FOR UNEVEN GEOMETRIES
                    dist = avg_dists_timereached[best_index]
                '''
                success_list.append(dist)
                if include_trajs == True:
                    xpos_1d = xPos.ravel()
                    ypos_1d = yPos.ravel()
                    img = sim_met.plot_from_density(xpos_1d, ypos_1d, 20)

                if include_activity == True:
                    for neuron in range(np.shape(activity)[0]):
                        activity_list.append(activity[neuron,0,:])
    # ADD IMPLEMENTATION FOR AVERAGING TRAJECTORIES OVER MULTIPLE SAMPLES
    if include_activity == True:
        activity_df = pd.DataFrame(activity_list)

    return success_list, target_list, time_list, activity_df, img
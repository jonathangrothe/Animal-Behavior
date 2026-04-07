import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# -------- Getting metrics --------

#so with what we returned from running the sim we can define a function to get metrics
def get_destination_metrics(xPos, yPos, targetsx, targetsy , stopping_distance = 0.1):
    '''
    A function which takes the outputs from the simulate_ringattractor code and gets a few metrics to quantify what happened in the simulation
    Inputs:
    xPos: all the x positions of the agent(s)
    yPos: all the y positions of the agent(s)
    targetsx: all the x positions of the targets
    targetsy: all the y positions of the targets
    stopping_distance: the maximum distance between the agent and target at which the agent is considered to have reached the target
    Returns:
    final_target: the final target the agent ends up at, -1 if the agent doesn't end up at a target
    time_target_reached: the time the agent got within the decision_precision of the final target
    movement_starts: the time the agent moves at least a tenth of the original smallest distance from the target away from the starting position
    '''
    # finds the first target the agent reaches (if the agent reaches a target)
    # and what time it reaches that target at
    ntargets = len(targetsx[:,0])
    tsteps = len(xPos[0,:])
    time_target_reached = tsteps
    target_reached = -1
    for index in range(len(xPos[0,:])):
        agent_x = xPos[0,index]
        agent_y = yPos[0,index]
        for target in range(ntargets):
            target_x = targetsx[target,index]
            target_y = targetsy[target,index]
            distance = np.sqrt((agent_x - target_x)**2 + (agent_y-target_y)**2)
            if distance < stopping_distance:
                time_target_reached = index
                target_reached = target
                break

    # finding the time the agent starts moving
    init_distance = 10000
    for i in range(ntargets):
        targ_distance = np.sqrt((xPos[0,0] - targetsx[i,0])**2 + (yPos[0,0]-targetsy[i,0])**2)
        if targ_distance < init_distance:
            init_distance = targ_distance
    movement_start = 0
    starting_x = xPos[0,0]
    starting_y = yPos[0,0]
    for index in range(len(xPos[0,:])):
        agent_x = xPos[0,index]
        agent_y = yPos[0,index]
        distance_fromstart = np.sqrt((agent_x-starting_x)**2+(agent_y-starting_y)**2)
        if distance_fromstart > init_distance*0.1:
            movement_start = index
            break
        
    return target_reached, time_target_reached, movement_start

def get_direction_info(headings, n_peaks, start_step=200, dest_step=5000):
    '''
    A function which takes the agent's heading at each time step and determines when it turns towards one target
    This will depend on the initial geometry, so we might need to take that as a parameter
    First we'll try to just detect changes and see if that's adequate
    '''
    headings = headings[0,:]
    peaks, properties = find_peaks(headings, prominence=0)
    prominences = properties['prominences']
    top_n_indices = np.argsort(prominences)[-n_peaks:]
    top_n_peaks = peaks[top_n_indices]
    return top_n_peaks

def get_success_rate(target_list, sample_size):
    '''
    A function which takes a list of targets over a number of samples, 
    and returns the probability of reaching a target for each sample
    Parameters:
    target_list: a list of targets reached by the agent in each simulation, arranged consecutively by sample
    n_samples: the number of samples
    Returns:
    success_prob: a list of size n_samples with each entry being the probability of success for that sample
    '''
    n_samples = len(target_list)//sample_size
    success_prob = []
    for s in range(n_samples):
        sample_list = target_list[s*sample_size:(s+1)*sample_size]
        failures = sample_list.count(-1)
        success_prob.append(1-failures/sample_size)
    return success_prob

def get_avg_distance(xPos, yPos, targetXpos, targetYpos, stop_time):
    tsteps = len(xPos[0,:])
    dists = np.sqrt((xPos[0, :] - targetXpos[:, :tsteps])**2 + 
                (yPos[0, :] - targetYpos[:, :tsteps])**2)
    return(np.mean(dists,axis=1))



def get_min_distance(xPos, yPos, targetXpos, targetYpos, even=True, better_ind=-1):
    '''
    A function that returns the minimum distance over the course of the simulation to the closer/better target over the sum of the distances to the targets
    If the geometry/attraction is uneven, we will designate one of the targets as the best target
    Designed for the two target case, but probably can be expanded to more targets
    Inputs: 
    xPos: the x positions of the agent throughout the simulation
    yPos: the y positions of the agent throughout the simulation
    targetXpos: the x positions of the targets throughout the simulation
    targetYpos: the y positions of the targets throughout the simulation
    even: a boolean which is either true (even geometry and attraction), or false, which implies that one of the targets is better than others
    better_ind: an integer which designates the best target
    '''
    tsteps = len(xPos[0,:])
    dists = np.sqrt((xPos[0, :] - targetXpos[:, :tsteps])**2 + 
                (yPos[0, :] - targetYpos[:, :tsteps])**2)
    sum_dists = dists.sum(axis=0)
    min_dists = dists[better_ind] if not even else dists.min(axis=0)
    min_over_sum = min_dists / sum_dists
    total_min = min_over_sum.min()
    return total_min

def plot_trajectories(xtraj, ytraj, targetsx, targetsy, colors, L, title, n_groups, agg = 'mean'):
    '''
    plots n different trajectories in n different colors
    xtraj: a n x t numpy array of x positions
    ytraj a n x t numpy array of y positions
    colors: a length n list of colors
    L: length of the grid
    '''
   # and since there is a lot of dimensions here we should probably write some code to flag when the dimensions are mismatched
    group_size = len(xtraj)//n_groups # FLAG THIS WHEN UNEVEN IN THE FUTURE
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.scatter(
            targetsx,
            targetsy,
            s = 10,
            marker = 's',
            c = [[0.8, 0, 0.2]], 
        )
    marker = 's'
    if agg == 'median':
        marker = 'o'
    if agg == 'min':
        marker = 'v'
    if agg == 'max':
        marker = '^'
    for group in range(n_groups):
        x_split = xtraj.iloc[group*group_size:(group+1)*group_size,:]
        y_split = ytraj.iloc[group*group_size:(group+1)*group_size,:]
        x_points = []
        y_points = []
        for point in range(x_split.shape[1]):
            if agg == 'mean':
                xloc_mean = np.mean(x_split.iloc[:,point])
                yloc_mean = np.mean(y_split.iloc[:,point])
                x_points.append(xloc_mean)
                y_points.append(yloc_mean)
            if agg == 'median':
                y_median = np.argpartition(y_split.iloc[:,point], group_size // 2)[group_size // 2]
                xloc_median = x_split.iloc[y_median,point]
                yloc_median = y_split.iloc[y_median,point]
                x_points.append(xloc_median)
                y_points.append(yloc_median)
            if agg == 'max':
                y_max = np.argmax(y_split.iloc[:, point])
                xloc_max = x_split.iloc[y_max,point]
                yloc_max = y_split.iloc[y_max,point]
                x_points.append(xloc_max)
                y_points.append(yloc_max)
            if agg == 'min':
                y_min = np.argmin(y_split.iloc[:,point])
                xloc_min = x_split.iloc[y_min,point]
                yloc_min = y_split.iloc[y_min,point]
                x_points.append(xloc_min)
                y_points.append(yloc_min)
        plt.scatter(
            x_points,
            y_points,
            s=10,
            color = colors[group],
            marker = marker
        )
    plt.title(title)
    #plt.show()
    plt.pause(0.001)  

def plot_metric(metric,x,title,xlabel,ylabel):
    '''
    A function which plots an aggregated metric over changing values of a parameter
    metric: a list of metrics with n samples per each value of the parameter we are measuring (the y axis)
    x: the values of the parameter we are measuring success over (the x axis)
    '''
    # first step: get the dimesions to line up
    # second step: plot it
    n_values = len(x)
    sample_size = len(metric)//n_values
    mean_metric = []
    se_metric = []
    for value in range(n_values):
        sample_metric = metric[value*sample_size:(value+1)*sample_size]
        mean_metric.append(np.mean(sample_metric))
        se_metric.append(np.std(sample_metric)/np.sqrt(sample_size))
    print(f"mean list: {mean_metric}")
    print(f"se list: {se_metric}")
    plt.plot(x, mean_metric, color = 'blue', label = 'Mean')
    plt.plot(x, np.add(mean_metric,se_metric), color = 'red', label = 'standard error', linestyle = ':')
    plt.plot(x, np.subtract(mean_metric,se_metric), color = 'red', linestyle = ':')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
 


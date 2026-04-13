import numpy as np
from scipy.signal import find_peaks
import cv2
import matplotlib.pyplot as plt

# -------- Getting metrics --------

#so with what we returned from running the sim we can define a function to get metrics
def get_destination_metrics(xPos, yPos, targetsx, targetsy , stopping_distance = 0.5):
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

def get_success_rate(target_list, sample_size, uneven=False, best_index=-1):
    '''
    A function which takes a list of targets over a number of samples, 
    and returns the probability of reaching a target for each sample
    Parameters:
    target_list: a list of targets reached by the agent in each simulation, arranged consecutively by sample
    sample_size: the number of samples of exactly the same settings run
    Returns:
    success_prob: a list of size sample_size with the probability of reaching a target for each sample
    correct_prob: a list of size sample_size with the probability of reaching the correct target for each sample
                  if the attraction is even, an empty list is returned
    success_se: the standard error of the probability of reaching a target
    correct_se: the standard error of the probability of reaching the correct target,
                if the attraction is even, and empty list is returned
    '''
    n_samples = len(target_list)//sample_size
    success_prob = []
    success_se = []
    correct_prob = []
    correct_se = []
    for s in range(n_samples):
        sample_list = target_list[s*sample_size:(s+1)*sample_size]
        failures = sample_list.count(-1)
        if uneven:
            successes = sample_list.count(best_index)
            cor_prob = successes/sample_size
            cor_se = np.sqrt((cor_prob*(1-cor_prob))/sample_size)
            correct_prob.append(cor_prob)
            correct_se.append(cor_se)
        suc_prob = 1-failures/sample_size
        suc_se = np.sqrt((suc_prob*(1-suc_prob))/sample_size)
        success_prob.append(suc_prob)
        success_se.append(suc_se)
    return success_prob, success_se, correct_prob, correct_se

def get_avg_distance(xPos, yPos, targetsx, targetsy, last_p=0.25):
    '''
    A function that returns a 1d numpy array of length ntargets with the average distance to each target
    over a certain proportion of the simulation
    Parameters:
    xPos: all the x positions of the agent(s)
    yPos: all the y positions of the agent(s)
    targetsx: all the x positions of the targets
    targetsy: all the y positions of the targets
    last_p: the proportion of timesteps of the simulation we will consider, counting backward 
            default is 0.25, which means we will consider the last quarter of the simulation
    '''
    tsteps = len(xPos[0,:])
    quarter = int(tsteps*last_p)
    x_last_quarter = xPos[0,-quarter:]
    y_last_quarter = yPos[0,-quarter:]
    targetsx_last_quarter = targetsx[:, :quarter]
    targetsy_last_quarter = targetsy[:,:quarter]
    dists = np.sqrt((x_last_quarter - targetsx_last_quarter)**2 + 
                    (y_last_quarter - targetsy_last_quarter)**2)
    return(np.mean(dists,axis=1))

def get_neuron_info(activity):
    # NEXT STEPS: use inhibition to accurately categorize a decision interval,
    # once we have the decision interval for each sample, we can look at interesting goings on in that time
    # Since we know that only a few neurons can be activated at a time and the rest are inhibited, 
    # it might be intersting to look at this contrast as a way of classifying the model's behavior (ie: difference between magnitudes of activated and inhibited)
    # definitely would be interesting to know if there are moments when more are activated at lower levels and if there are moments when only one or two are activated
    # or if that depends on simulation settings 
    '''
    A function which gets information about neuron activity over the course of the simulation
    info we want: 
    at each time step: number of neurons activated, max activation, which neuron is most activated
    '''
    n_inhib = 0
    inhib_index = []
    tsteps = np.shape(activity)[1] - 100 # I let the sim run for 100 times after reaching the target, but don't want that to impact analysis
    for t in range(tsteps):
        max_activation = np.max(activity[:,t], axis=0)
        if max_activation < 0:
            n_inhib += 1
            inhib_index.append(t)
    return n_inhib, inhib_index

def get_inhibition_rates(inhib_list,sample_size):
    '''
    A function that expands the neuron info to more samples
    parameters:
    inhib_list: a list of number of times when all neurons were inhibited (each value from get_neuron_info)
    sample_size: sample size for each run with the exact same settings 
    returns: 
    mean_inhib: a list of the mean over each sample of the number of times all neurons were inhibited 
    '''
    mean_inhib = []
    n_samples = len(inhib_list)//sample_size
    for x in range(n_samples):
        mean_n_inhib = np.mean(inhib_list[x*sample_size:(x+1)*sample_size])
        print(f"mean: {mean_n_inhib}")
        mean_inhib.append(mean_n_inhib)
    return mean_inhib



# THIS NEEDS REWORKING
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
    Parameters:
    metric: a list of metrics with n samples per each value of the parameter we are measuring (the y axis)
    x: the values of the parameter we are measuring success over (the x axis)
    title: the title of the plot
    xlabel: the label of the x-axis
    ylabel: the label of the y-axis
    Expected output: 
    A line plot of the metric (y-axis) with standard error lines over x (x-axis)
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
    plt.plot(x, mean_metric, color = 'blue', label = 'Mean')
    plt.plot(x, np.add(mean_metric,se_metric), color = 'red', label = 'standard error', linestyle = ':')
    plt.plot(x, np.subtract(mean_metric,se_metric), color = 'red', linestyle = ':')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    
def plot_neurons(activity_df, activation_cutoff=0.5, tstart=0, tstop=0):
    # implementation right now: iterates through the mean contributions and finds the maximum mean contribution of any one neuron
    # then plots the activity of all neurons that contribute more than half as much as the mean neuron
    # this could be functionalized, but it is also a good tool to help show when decisions are made
    # this can really clearly illustrate the differences between ego and allocentric in this context
    # next: look at the bad decisions and see if anything is different (look at what the mean contributions actually are as well)
    # and look at the noisy data and see if this sheds any light
    # also to do: extend this to mean neuron activity over multiple samples?
    '''
    A function which takes the activity of all the neurons in the ring attractor for one agent and plots them
    parameters: 
    activity_df: a Nxt dataframe where N is the number of neurons and t is the number of timesteps in the simulation
    '''
    if tstop == 0:
        tstop = len(activity_df)
    n_neurons = activity_df.shape[0]
    mean_max = -100
    active_neuron_list = []
    for neuron in range(n_neurons):
        mean_activity = np.mean(activity_df.iloc[neuron,tstart:tstop+1])
        if mean_activity > mean_max:
            mean_max = mean_activity

    print(mean_activity)
    if mean_activity < 0:
        if activation_cutoff == 0:
            activation_cutoff += 0.0001 # avoid division by 0?
        activation_cutoff = 1/activation_cutoff
    for neuron in range(n_neurons):
        mean_activity = np.mean(activity_df.iloc[neuron,tstart:tstop+1])
        if mean_activity > mean_max * activation_cutoff:
            plt.plot(activity_df.iloc[neuron,tstart:tstop+1], label = f"neuron: {neuron}")
            active_neuron_list.append(neuron)
        plt.title("plot of neuron activity")
        plt.legend()
    return active_neuron_list


# heat map plotting from vivek's code - very computationally intensive right now
def density_map(x, y):
    '''
    Takes x and y values and returns a 2d array corresponding to a 2d histogram.
    Designed to be used on a trajectory plot, x and y are both intended to be coordinates
    Parameters:
    x: x values to consider in 2d hist
    y: y values to consider in 2d hist
    Returns:
    tmp_img: a 2d array with x and y values bucketed into rows respectively. 
            Range is pre set to 0,100 in each dimension with 500 bins
    '''
    blur = (11, 11)
    h, xedge, yedge, image = plt.hist2d(x, y, bins = 500, density=True, range = [[0,100],[0,100]])
    tmp_img = np.rot90(cv2.GaussianBlur(h, blur, 0))
    tmp_img /= np.max(tmp_img)
    return tmp_img

def plot_from_density(xPos, yPos, window_size):
    '''
    A function which takes x positions, y positions, and window size and plots a heat map of the trajectory
    Parameters: 
    xPos: an array of x positions. Can include multiple runs as long as x indices are aligned with y indices
    yPos: an array of y positions
    window_size: how many positions to aggregate over at a time. 
                Does not affect how many times we call density_map, just the width of each range of densities
    '''
    tmax = len(xPos)
    for i in range((tmax-window_size)//10):
        # calculate the window
        window_min = i*10
        window_max = i*10 + window_size
        # get the x positions and y positions in the window and map them
        x = xPos[window_min:window_max+1]
        y = yPos[window_min:window_max+1]
        tmp_img = density_map(x, y)
        if i == 0:
            img = tmp_img
        else:
            img = np.fmax(tmp_img, img)
    return img
    

 


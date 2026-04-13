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

    
def plot_neurons(activity_df):
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
    n_neurons = activity_df.shape[0]
    mean_max = -100
    active_neuron_list = []
    for neuron in range(n_neurons):
        mean_activity = np.mean(activity_df.iloc[neuron,:])
        if mean_activity > mean_max:
            mean_max = mean_activity
    for neuron in range(n_neurons):
        mean_activity = np.mean(activity_df.iloc[neuron,:])
        if mean_activity > mean_max * 0.5:
            plt.plot(activity_df.iloc[neuron,:])
            active_neuron_list.append(neuron)
        plt.title("plot of neuron activity")
    return active_neuron_list


# plotting from vivek's code
def density_map(x, y):
    blur = (11, 11)
    h, xedge, yedge, image = plt.hist2d(x, y, bins = 100, density=True, range = [[0,100],[0,100]])
    print(f"shape of h: {np.shape(h)}")
    print(f"h: {h}")
    tmp_img = np.rot90(cv2.GaussianBlur(h, blur, 0))
    tmp_img /= np.max(tmp_img)
    print(f"shape of tmp_img: {np.shape(tmp_img)}")
    print(f"tmp_img: {tmp_img}")
    return tmp_img

def plot_from_density(xPos, yPos, window_size):
    tmax = len(xPos)
    ts = np.arange(1, tmax + 1)

    for i in range(tmax-window_size):
        # calculate the window
        window_min = i
        window_max = i + window_size

        # get the x positions and y positions in the window and map them
        x = xPos[(ts > window_min) & (ts < window_max)]
        y = yPos[(ts > window_min) & (ts < window_max)]
        tmp_img = density_map(x, y)
        if i == 0:
            img = tmp_img
        else:
            img = np.fmax(tmp_img, img)
    return img
    

 


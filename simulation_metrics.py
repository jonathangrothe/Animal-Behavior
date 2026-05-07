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

def get_peaks(data_arr):
    '''
    a function which takes a numpy array and finds the top n peaks 
    designed to be used to find the decision points, could be used with headings, trajectories, or neuron activity
    (as long as we get the data in the right shape)
    '''
    x = np.linspace(0,len(data_arr),num=len(data_arr))
    diff = np.gradient(data_arr,x)
    critical_points = np.where(np.diff(np.sign(diff)))[0]
    print(f"critical points: {critical_points}")
    return critical_points

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
    '''
    A function which gets information about neuron activity over the course of the simulation
    info we want: 
    sum of activity at each step, the range of activity at each step, variance of activity at each step, number inhibited at each step, number activated at each step,
    rate of change of each neuron 
    and maybe not this function, but then we want time step of the minimum range, timesteps the number inhibited/activated changes
    '''
    sum_activity = []
    range_activity = []
    var_activity = []
    n_inhib_list = []
    n_active_list = []
    neuron_change_rates = []
    tsteps = np.shape(activity)[1]
    for t in range(tsteps):
        sum_at_t = np.sum(activity[:,t], axis=0)
        max_activation = np.max(activity[:,t], axis=0)
        min_activation = np.min(activity[:,t], axis=0)
        range_at_t = max_activation-min_activation
        variance_at_t = np.var(activity[:,t])
        n_inhib = np.count_nonzero(activity[:,t] < 0)
        n_active = np.count_nonzero(activity[:,t] > 0)
        sum_activity.append(sum_at_t)
        range_activity.append(range_at_t)
        var_activity.append(variance_at_t)
        n_inhib_list.append(n_inhib)
        n_active_list.append(n_active)
    for neuron in range(np.shape(activity)[0]):
        rate_of_change = np.diff(activity[neuron,:])
        neuron_change_rates.append(rate_of_change)
    return sum_activity, range_activity, var_activity, n_inhib_list, n_active_list, neuron_change_rates

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
        mean_inhib.append(mean_n_inhib)
    return mean_inhib

def get_decision_time(sum_activity):
    '''
    takes a list of the sum of all neuron activity, 
    returns the index of when the sum of activity gets past a threshold (thresh) of the final sum
    '''
    dec_start = np.argmin(sum_activity)
    dec_end = dec_start
    while np.abs(sum_activity[dec_end]-sum_activity[dec_start]) < 0.5:
        dec_end+=1
        if dec_end == len(sum_activity):
            dec_end = dec_end -1
            break
    return dec_start, dec_end

def get_metric_mean_se(metric, sample_size):
    means = []
    ses = []
    n_values = len(metric) //sample_size
    for i in range(n_values):
        met_mean = np.mean(metric[i*sample_size:(i+1)*sample_size])
        met_se = np.std(metric[i*sample_size:(i+1)*sample_size])/np.sqrt(sample_size)
        means.append(met_mean)
        ses.append(met_se)
    return means, ses

def plot_metric(metrics,x,colors,labels,figurenum,size,title,xlabel,ylabel):
    '''
    A function which plots an aggregated metric over changing values of a parameter
    Parameters:
    metrics: a list of list of metrics with n samples per each value of the parameter we are measuring (the y axis)
    x: the values of the parameter we are measuring success over (the x axis)
    labels: the labels for each metric
    Expected output: 
    A line plot of the metric (y-axis) with standard error lines over x (x-axis)
    '''
    # first step: get the dimesions to line up
    # second step: plot it
    plt.figure(figsize=size)
    plt.figure(figurenum)
    n_values = len(x)
    sample_size = len(metrics[0])//n_values
    mean_metric_list = []
    for m in range(len(metrics)):
        mean_metric = []
        se_metric = []
        for value in range(n_values):
            sample_metric = metrics[m][value*sample_size:(value+1)*sample_size]
            mean_metric.append(np.mean(sample_metric))
            se_metric.append(np.std(sample_metric)/np.sqrt(sample_size))
        if m == 0 or m == (len(metrics)-1):
            plt.plot(x, mean_metric, color = colors[m], label = labels[int(bool(m))])
            plt.plot(x, np.add(mean_metric,se_metric), color = colors[m], label = 'standard error', linestyle = ':')
            plt.plot(x, np.subtract(mean_metric,se_metric), color = colors[m], linestyle = ':')
        else:
            plt.plot(x, mean_metric, color = colors[m])
            plt.plot(x, np.add(mean_metric,se_metric), color = colors[m], linestyle = ':')
            plt.plot(x, np.subtract(mean_metric,se_metric), color = colors[m], linestyle = ':')
        mean_metric_list.append(mean_metric)

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return figurenum+1, mean_metric_list


def plot_neurons(activity_df, figure, agg=True, tstart=0, tstop=0, N=100, start_neuron=0, end_neuron=100):
    '''
    A function which takes the activity of all the neurons in the ring attractor for one agent and plots them
    parameters: 
    activity_: a Nxt dataframe where N is the number of neurons and t is the number of timesteps in the simulation
    '''
    if tstop == 0:
        tstop = activity_df.shape[1]
    sample_size = activity_df.shape[0]//N
    activity_df = activity_df.iloc[:,tstart:tstop+1] # first slice the dataframe based on time interval so we don't have to worry about indexing for the interval later
    average_activity_list = []
    active_neuron_list = []
    # this gets a list of each neurons average activity averaged over all samples
    for neuron in range(start_neuron,end_neuron):
        activity_over_sims = []
        for sim in range(sample_size):
            mean_activity = np.mean(activity_df.iloc[neuron+(sim*N),:]) #mean activity for neuron in this sim over this time period
            activity_over_sims.append(mean_activity)
        mean_over_sims = np.mean(activity_over_sims)
        if mean_over_sims > 0:
            average_activity_list.append(mean_over_sims)
            active_neuron_list.append(neuron)
    print(active_neuron_list)
    # for each neuron we want to plot: 
    # iterate through each sim, plot first one with label, then plot rest without, use different color for different neurons
    colors = plt.cm.viridis(np.linspace(0,1,len(active_neuron_list)))
    if agg:
        for neuron in range(len(active_neuron_list)):
            # get mean activity of every 100th neuron starting at this one
            figure.plot(np.mean(activity_df.iloc[active_neuron_list[neuron]::N],axis=0),alpha = 0.6, label = f"neuron: {active_neuron_list[neuron]+start_neuron}", color=colors[neuron])
            #figure.legend()
                
    else:
        for neuron in range(len(active_neuron_list)):
            for sim in range(sample_size):
                if sim == 0:
                    figure.plot(activity_df.iloc[active_neuron_list[neuron]+((sim)*N)], alpha = 0.6/sample_size, label = f"neuron: {active_neuron_list[neuron]+start_neuron}", color=colors[neuron])
                else: 

                    figure.plot(activity_df.iloc[active_neuron_list[neuron]+((sim)*N)], alpha = 0.6/sample_size, color = colors[neuron])

        leg = figure.legend()
        for lh in leg.legend_handles: 
            lh.set_alpha(1.0)
    return None

def plot_sum_activity(activity_list,figure):
    '''
    A function that plots the sum of activity over ONE setting, with each sample a different line?
    '''
    min_len = 5000
    for item in activity_list:
        figure.plot(item)
        if len(item) < min_len:
            min_len = len(item)
    sum_list_truncated = []
    for item in activity_list:
        sum_list_truncated.append(item[:min_len])
    mean_list = np.mean(sum_list_truncated,axis=0)
    return mean_list

def plot_traj(xPos,yPos,targetsx,targetsy,sample_size,figure,start_ind=0,end_ind=0):
    '''
    A function that takes x trajectories and y trajectories and plots them over each other, with a low ish opacity so we can see overlap. 
    Designed to be used over the same simulation settings with a number s of samples.
    Need: to color by velocity
    '''
    if end_ind == 0:
        for sample in range(sample_size):
            deltax = np.diff(xPos[sample][start_ind:])
            deltay = np.diff(yPos[sample][start_ind:])
            dist = np.zeros(len(xPos[sample][start_ind:]))
            dist[0] =0
            dist[1:] = np.sqrt(deltax**2 + deltay**2)
            figure.scatter(xPos[sample][start_ind:],yPos[sample][start_ind:],c = dist, cmap = 'afmhot_r', alpha=0.4/sample_size,s=1)
        figure.scatter(targetsx,targetsy,color='red',s=5)
    else:
        for sample in range(sample_size):
            deltax = np.diff(xPos[sample][start_ind:end_ind])
            deltay = np.diff(yPos[sample][start_ind:end_ind])
            dist = np.zeros(len(xPos[sample][start_ind:end_ind]))
            dist[0] =0
            dist[1:] = np.sqrt(deltax**2 + deltay**2)
            figure.scatter(xPos[sample][start_ind:end_ind],yPos[sample][start_ind:end_ind],c = dist, cmap = 'afmhot_r', alpha=0.5,s=8)
    return None


# heat map plotting from vivek's code - very slow to run right now
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
    for i in range((tmax-window_size)):
        # calculate the window
        window_min = i
        window_max = i + window_size
        # get the x positions and y positions in the window and map them
        x = xPos[window_min:window_max+1]
        y = yPos[window_min:window_max+1]
        tmp_img = density_map(x, y)
        if i == 0:
            img = tmp_img
        else:
            img = np.fmax(tmp_img, img)
    return img
    


# TO DO: PLOT HEADING
# Will need to get heading from the repeated_sims in like a list of lists or smth
# should have all the indices of all neurons being inhibited, so we're good there

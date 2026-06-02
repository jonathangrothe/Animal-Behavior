import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# -------- Getting metrics --------

def get_destination_metrics(xPos, yPos, targetsx, targetsy , stopping_distance = 0.5):
    '''
    A function which takes the positional outputs from a simulation run and returns which target the agent reaches (-1 if no target) 
    and at what time the agent reaches that target

    Parameters:
    xPos: all the x positions of the agent(s)
    yPos: all the y positions of the agent(s)
    targetsx: all the x positions of the targets
    targetsy: all the y positions of the targets
    stopping_distance: the maximum distance between the agent and target at which the agent is considered to have reached the target

    Returns:
    final_target: the final target the agent ends up at (defined as being within stopping_distance of a target), zero indexed, -1 if the agent doesn't end up at a target
    time_target_reached: the time the agent got within the stopping_distance of the final target
    '''
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

    return target_reached, time_target_reached


def get_bifurcation_times(xPos,yPos,min_diff=1): # this function needs work...
    '''
    A function which takes the x positions and y positions and returns an approximate list of time steps where bifurcations occurred.
    Assumes that trajectories are approximately straight and fits a linear regression to approximate the trajectory. 
    A bifurcation is considered to have happened when the actual trajectory differs from the linear regression by a certain amount.

    Parameters:
    xPos: the x positions of the agent
    yPos: the y positions of the agent

    Returns: 
    dec_points: a list of time steps where bifurcations were found
    dec_pos: a list of tuples which are the coordinates where bifurcations occurred
    '''
    dec_points = []
    curr_index = len(xPos)-1
    complete = False
    while not complete:
        backtrack = curr_index - 30
        diff = 0
        while diff < min_diff:
            slope, intercept, r_value, p_value, std_err = stats.linregress(xPos[backtrack:curr_index], yPos[backtrack:curr_index])
            predicted_backtrack = intercept + slope*xPos[backtrack]
            diff = np.abs(predicted_backtrack - yPos[backtrack])
            if diff < min_diff: 
                backtrack = backtrack - 1
            if backtrack < 0:
                complete = True
                break
        if not complete:
            dec_points.append(backtrack)

        curr_index = backtrack
        if curr_index < 40:
            complete = True
    dec_points.reverse()
    dec_pos = []
    for item in dec_points:
        dec_pos.append((xPos[item],yPos[item]))
    return dec_points, dec_pos


def get_success_rate(target_list, sample_size, ntargets):
    '''
    A function which takes a list of targets over a number of samples and returns the probability of reaching a target for each sample

    Parameters:
    target_list: a list of targets reached by the agent in each simulation, arranged so that all runs with the same settings are consecutive
    sample_size: the number of samples taken for each unique set of settings
    ntargets: the number of targets

    Returns:
    target_probs: a list of size ntargets with the probability of reaching each target in consecutive order
    target_ses: a list of size ntargets with the standard error of reaching each target in consecutive order
    '''
    n_samples = len(target_list)//sample_size
    target_probs = []
    target_ses = []
    for s in range(n_samples):
        sample_list = target_list[s*sample_size:(s+1)*sample_size]
        probs = []
        ses = []
        for i in range (ntargets):
            times_reached = sample_list.count(i)
            prob = times_reached/sample_size
            se = np.sqrt((prob*(1-prob))/sample_size)
            probs.append(prob)
            ses.append(se)
        target_probs.append(probs)
        target_ses.append(ses)

    return target_probs, target_ses


def find_bumps(activity):
    '''
    A function which takes in the activity data for a simulation and returns a list of where indices that are the peak of bumps at each time point in that simulation
    
    Parameters:
    activity: a N x tsteps data frame of neuron activity over the entire simulation

    Returns: 
    bump_list: a list of length tsteps which every entry is the indices of the peak of a bump at that time 
    '''
    activity_to_insert = activity.iloc[0:4,1:]
    activity_plus = pd.concat([activity,activity_to_insert]).reset_index(drop=True)
    tsteps = len(activity.iloc[0,:])
    bump_list = []
    for i in range(tsteps):
        indices_bumps, _ = find_peaks(activity_plus.iloc[:,i])
        true_bumps = [x for x in indices_bumps if x <= 100]
        bump_list.append(true_bumps)
    return bump_list

def plot_metric(metrics,x,colors,labels,fig,title,xlabel,ylabel):
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
    #plt.figure(figsize=size)
    #plt.figure(figurenum)
    n_values = len(x)
    sample_size = len(metrics[0])//n_values
    mean_metric_list = []
    print(len(metrics))
    for m in range(len(metrics)):
        mean_metric = []
        se_metric = []
        for value in range(n_values):
            sample_metric = metrics[m][value*sample_size:(value+1)*sample_size]
            mean_metric.append(np.mean(sample_metric))
            se_metric.append(np.std(sample_metric)/np.sqrt(sample_size))
        if m == 0 or m == (len(metrics)-1):
            fig.plot(x, mean_metric, color = colors[m], label = labels[m])
            fig.plot(x, np.add(mean_metric,se_metric), color = colors[m], label = 'standard error', linestyle = ':')
            fig.plot(x, np.subtract(mean_metric,se_metric), color = colors[m], linestyle = ':')
        else:
            fig.plot(x, mean_metric, color = colors[m])
            fig.plot(x, np.add(mean_metric,se_metric), color = colors[m], linestyle = ':')
            fig.plot(x, np.subtract(mean_metric,se_metric), color = colors[m], linestyle = ':')
        mean_metric_list.append(mean_metric)

    fig.legend()
    fig.set_title(title)
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    return 0, mean_metric_list


def plot_traj(xPos,yPos,targetsx,targetsy,sample_size,dec_points,figure,plot_dec_point=False,start_ind=0,end_ind=0):
    '''
    A function that takes x trajectories and y trajectories and plots them over each other, with a low ish opacity so we can see overlap. 
    Designed to be used over the same simulation settings with a number s of samples.
    Need: to color by velocity
    return:
    '''
    if end_ind == 0:
        for sample in range(sample_size):
            figure.plot(xPos[sample][start_ind:],yPos[sample][start_ind:],color='blue',alpha=1/sample_size)
            if plot_dec_point:
                dec_time = dec_points[sample]
                figure.scatter(xPos[sample][dec_time],yPos[sample][dec_time], color="green",alpha = 0.1)
            else:
                figure.scatter(xPos[sample][0],yPos[sample][0], color='red',alpha=0)
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


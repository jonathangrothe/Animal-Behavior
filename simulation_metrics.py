import random
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# -------- Getting metrics --------

def get_destination_metrics(xPos, yPos, targetsx, targetsy , stopping_distance = 0.5):
    '''
    A function which takes the positional outputs from a simulation run and returns which target the agent reaches (-1 if no target) 
    and at what time the agent reaches that target

    Parameters:
    xPos: a list of lists of x positions of the agent where each sublist contains all the positions in one simulation run
    yPos: a list of lists of y positions of the agent where each sublist contains all the positions in one simulation run
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
    xPos: a list of lists of x positions of the agent where each sublist contains all the positions in one simulation run
    yPos: a list of lists of y positions of the agent where each sublist contains all the positions in one simulation run

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

def get_bump_type(initialxt,initialyt,xPos,yPos,activity,interval=10):
    '''
    A function that takes the targets position and the activity over a simulation and returns a list of probabilities of the bump behavior.
    Designed right now for three targets, expanding to more should be straightforward but requires close reading.
    Does not handle bifurcations specifically yet, but that could come from observing some 1 bump and some 2 bump behavior. 
    Still need to get that solid before doing more sweeps
    Paramters:
    initialxt: a list of initial x positions of the targets
    initialyt: a list of initial y positions of the targets
    xPos: a list of x positions over the simulation
    yPos: a list of y positions over the simulation
    activity: a dataframe of neuron activity over the simulation
    interval: how small of an interval to consider at a time. 
              Intervals are considered as a static point, so they should be small enough that in most cases the bump doesn't shift majorly over one interval
    returns: 
    proportion_list: a list of proportions of each bump outcome. The bump outcomes are defined as: 0 bumps, 1 bump, 2 bumps, 3 bumps, and more than 3 bumps.
                      The proportions are calculated as total number of intervals in which each bump state occurs over the total number of intervals.
                      This doesn't describe behavior exactly, but does offer a good idea of what kinds of behavior occur at each time. 

    '''
    activity = activity.dropna(axis=1)
    total_time = activity.shape[1]
    total_intervals = int(total_time/interval)
    ntargets = len(initialxt)
    bump1_list = []
    bump2_list = []
    bump3_list = []
    target_neurons_list = []
    rows = random.sample(range(0, 100), 5)
    npeaks_list = []
    oscilates = False
    for item in rows: 
        row = activity.iloc[item,:]
        peaks, indices = find_peaks(row)
        n_peaks = len(peaks)
        npeaks_list.append(n_peaks)
    if np.mean(npeaks_list) >= 5 and total_time >= 3000:
        oscilates = True
    for i in range(total_intervals):
        end_int = (i+1)*interval
        if end_int > total_time:
            end_int = total_time
        xPos_interval = xPos[i*interval:end_int]
        yPos_interval = yPos[i*interval:end_int]
        target_neurons = []
        for t in range(ntargets):
            angle = np.atan2(initialyt[t]-yPos_interval[0],initialxt[t]-xPos_interval[0])
            index = np.round(100*(angle/(2*np.pi)))
            if index < 0:
                index += 100
            target_neurons.append(index)
        target_neurons_list.append(target_neurons)
        activity_int = activity.iloc[target_neurons,i*interval:end_int]
        bump1 = np.mean(activity_int.iloc[0,:])
        bump2 = np.mean(activity_int.iloc[1,:])
        bump3 = np.mean(activity_int.iloc[2,:])
        bump1_list.append(bump1)
        bump2_list.append(bump2)
        bump3_list.append(bump3)
    counter_0 = 0
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_other = 0
    for a in range(len(bump1_list)):
        num_positive = sum(1 for item in [bump1_list[a],bump2_list[a],bump3_list[a]] if item > 0)
        end_int = (a+1)*interval
        if end_int > total_time:
            end_int = total_time
        if num_positive > 1:
            neuron_left = round(target_neurons_list[a][0])
            neuron_center = round(target_neurons_list[a][1])
            neuron_right = round(target_neurons_list[a][2])
            # if left and center are positive, check to see if between them there is a smaller value than the min between them
            if bump1_list[a] > 0 and bump2_list[a] > 0:
                # take the difference between the neurons to find the shorter interval
                # if the shorter interval doesn't work indexing wise, swap it around
                neuron_diff = neuron_left - neuron_center
                between_min = 0
                if 0 <= neuron_diff < 50:
                    between_min = np.min(activity.iloc[25:30,a*interval:end_int])
                elif neuron_diff >= 50:
                    combined = pd.concat([activity.iloc[:neuron_center,a*interval:end_int],activity.iloc[neuron_left:,a*interval:end_int]])
                    between_min = np.min(combined)
                else:
                    between_min = np.min(activity.iloc[neuron_left:neuron_center,a*interval:end_int])
                if between_min >= 0:
                    num_positive = num_positive -1
            if bump2_list[a] > 0 and bump3_list[a] > 0:
                neuron_diff = neuron_center - neuron_right
                between_min = 0
                if 0 <= neuron_diff < 50:
                    between_min = np.min(activity.iloc[(neuron_right+1):neuron_center,a*interval:end_int])
                elif neuron_diff >= 50:
                    combined = pd.concat([activity.iloc[:neuron_right,a*interval:end_int],activity.iloc[neuron_center:,a*interval:end_int]])
                    between_min = np.min(combined)
                else:
                    between_min = np.min(activity.iloc[neuron_center:neuron_right,a*interval:end_int])
                if between_min >= 0:
                    num_positive = num_positive -1
        if num_positive == 0:
            counter_0 += 1
        if num_positive == 1:
            if oscilates:
                counter_2 += 1
            else:
                counter_1 += 1
        if num_positive == 2:
            counter_2 += 1
        if num_positive == 3:
            counter_3 += 1
        if num_positive > 3:
            print(f"nc, time: {a*interval}: neuron at t1: {bump1_list[a]}, neuron at t2: {bump2_list[a]}, neuron at t3: {bump3_list[a]}, npos: {num_positive}, min: {min(bump1_list[a],bump2_list[a],bump3_list[a])}")
            counter_other += 1
    situation_sum = counter_0 + counter_1 + counter_2 + counter_3 + counter_other
    p_0 = counter_0/situation_sum
    p_1 = counter_1/situation_sum
    p_2 = counter_2/situation_sum
    p_3 = counter_3/situation_sum
    p_other = counter_other/situation_sum
    if p_other > 0.5:
        print("check this one, p other > 0.5")
    proportion_list = [p_0,p_1,p_2,p_3,p_other]
    #print(proportion_list)
    return proportion_list

def plot_metric(metrics,x,colors,labels,fig,title,xlabel,ylabel):
    '''
    A function which plots an aggregated metric over changing values of a parameter

    Parameters:
    metrics: a list of lists where each sublist is length sample_size*n_values and contains sample_size consecutive metric measurements
             for each time the simulation is run over a given parameter, p1. Each list of metrics corresponds to another parameter, p2.
             If metrics was an array, there would be p2 rows and sample_size*p1 columns, a,0:sample_size would correspond to one set of simulation settings
    x: the values of the parameter we are measuring our metric over (the x axis)
    colors: a list of size n_values which contains colors for each value of p2
    labels: the labels for each other parameter (p2), just the first and last labels are shown to avoid overcrowding the plot
    fig: the figure we are doing the plotting in
    title: the title of the plot
    xlabel: the label of the x axis
    ylabel: the label of the y axis

    Returns:
    mean_metric_list: a list of lists which contains the mean for each unique simulation setting. 
                      Same design as metrics, but instead of a,0 would correspond to the mean over a unique set of simulation settings, and a,1 a different set

    Expected output: 
    A line plot of metrics (y-axis) with standard error lines over x (x-axis)
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
    return mean_metric_list


def plot_traj(xPos,yPos,targetsx,targetsy,sample_size,dec_points,figure,plot_dec_point=False,start_ind=0,end_ind=0):
    '''
    A function that takes x trajectories and y trajectories and plots them over each other, with a low ish opacity so we can see overlap. 
    Designed to be used over the same simulation settings with a number s of samples.

    Parameters:
    xPos: a list of lists of x positions of the agent where each sublist contains all the positions in one simulation run
    yPos: a list of lists of y positions of the agent where each sublist contains all the positions in one simulation run
    targetsx: a list of x positions of the targets
    targetsy: a list of y positions of the targets
    sample_size: the number of repitions of the same exact simulation settings
    dec_points: a list of times where bifurcations are found, which will optionally plotted
    figure: the figure to plot the trajectories in
    plot_dec_point: whether or not to plot the bifurcation points
    start_ind: the time index to start plotting
    end_ind: the time index to stop plotting (if 0 plot until the final time step)
    
    returns: 
    None

    Expected output: 
    A line plot of the trajectories of the agent in blue that is darker on more commonly taken paths and lighter on less commonly taken paths. 
    The targets are plotted as small red dots, and if the bifurcation times are plotted then they will be larger, low opacity, green dots on the trajectory. 
    '''
    if end_ind == 0:
        for sample in range(sample_size):
            figure.plot(xPos[sample][start_ind:],yPos[sample][start_ind:],color='blue',alpha=1/sample_size)
            if plot_dec_point:
                dec_time = dec_points[sample]
                figure.scatter(xPos[sample][dec_time],yPos[sample][dec_time], color="green",alpha = 0.1)
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

def plot_phase_over_area(h0_list,sigma_list,target_list,acolor,figure):
    '''
    This function will take a set of simulations done over a constant area and plot a line plot with h0 as the x axis and sigma as the y axis
    each point will be highlighted in a different color or tick mark or something to note which phase its in (maybe one for bump and one for outcome)
    We'll first just start with reaching the target
    '''
    figure.plot(h0_list,sigma_list,c=acolor)
    #figure.scatter(x=h0_list,y=sigma_list,c=target_list,cmap="RdYlGn")



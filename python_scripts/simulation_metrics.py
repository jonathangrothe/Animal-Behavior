import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from helper_functions.helpers import get_estimated_angle

# -------- Getting metrics --------

def get_destination_metrics(xPos, yPos, targetsx, targetsy, maxtime = 5000):
    '''
    A function which finds the target the agent reached (if it reached a target) and the time it reached that target

    Parameters: 
        xPos: a numpy array of size nagents x tsteps of each agent's x position at each point in the simulation
        yPos: a numpy array of size nagents x tsteps of each agent's y position at each point in the simulation
        targetsx: a list of size ntargets of the targets x positions (this is assumed to be static )
        targetsy: a list of size ntargets of the targets y positions (this is assumed to be static)
        maxtime: the number of time steps will run without reaching a target

    Returns: 
        target_reached: the target (indexed as they are in the targetsx and targetsy list) the agent reaches
        time_target_reached: the number of time steps in the simulation, the length of xPos and yPos
    '''
    if np.shape(xPos)[0] == 1:
        xPos = xPos.ravel()
        yPos = yPos.ravel()
    if xPos.size != yPos.size:
        raise ValueError("arrays for x position and y position are different sizes")
    if len(targetsx) != len(targetsy):
        raise ValueError("target x positions and target y positions are different sizes")
    xPos = np.asarray(xPos)
    yPos = np.asarray(yPos)
    tsteps = len(xPos)
    target_reached = -1
    distance = (xPos[-1]-targetsx)**2+(yPos[-1]-targetsy)**2
    if tsteps <= maxtime:
        target_reached = np.argmin(distance)
    else:
        tsteps = maxtime+1
    return target_reached, tsteps


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
    if len(target_list) % sample_size != 0:
        raise ValueError("Sample size does not evenly divide total number of simulations")
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
        p_fail = sample_list.count(-1)/sample_size
        if sum(probs) + p_fail != 1:
            print(f"Warning: unexpected input in target list, counted as failure, target list: {sample_list}")
        target_probs.append(probs)
        target_ses.append(ses)
    return target_probs, target_ses


def get_bump_type(activity,maxtime=5000):
    '''
    A function that analyses the neural activity of a simulation in small intervals 
    and returns a number corresponding to the number of bumps that best describe the simulation.
    Designed to handle three targets only right now.

    Paramters:
        activity: a dataframe of neuron activity over the simulation
                Intervals are considered as a static point, so they should be small enough that in most cases the bump doesn't shift majorly over one interval

    Returns: 
        phase: a number to classify the number of bumps that best describes the simulation, possible outputs are 0, 1, 2, 3, or 4 (4 is undesireable).
               Usually the phase corresponds to the most common number of bumps present across all the intervals, but occasionally that is not the case.
    '''
    total_time = activity.shape[1]
    if maxtime < total_time:
        total_time = maxtime + 1
    counters = np.zeros(5, dtype=np.int32)
    for i in range(total_time):
        full_activity = activity[:,i]
        full_indices = [i for i, x in enumerate(full_activity) if x > 0]
        nbumps = 0
        if len(full_indices) > 0:
            nbumps += 1
            for index in range(len(full_indices)-1):
                if full_indices[index+1]-full_indices[index] > 1:
                    nbumps += 1
            if full_indices[0] == 0 and full_indices[-1] == 99:
                nbumps -= 1
            if nbumps >= 4:
                nbumps = 4
        counters[nbumps] += 1
        
    proportions = counters / sum(counters)
    phase = np.argmax(proportions)
    max_indices = np.where(proportions == proportions.max())[0]
    if len(max_indices) > 1:
        if total_time == maxtime+1:
            if proportions[2] >= 0.04:
                phase = 2
            elif 1 in max_indices:
                phase = 1
            elif 3 in max_indices:
                phase = 3
            else: 
                phase = 0
        else:
            if 1 in max_indices:
                phase = 1
            elif 3 in max_indices:
                phase = 3
            else:
                phase = 0

    if proportions[2] >= 0.04 and phase == 1 and total_time == maxtime+1:
        phase = 2
    return phase

def get_bifurcation_angle(xPos, yPos, headings, direction_thresh = np.pi/18, maxtime=5000,test=False):
    '''
    A function that takes the xpositions, ypositions, and headings from a trajectory, to determine the bifurcation points, using a threshold to determine if a turn is too insignificant to be counted. 
    Here a bifurcation is assumed to be a region of indices, from the index where the agent begins to turn to where it stops turning, 
    which gives us 4 points: the start point, the bifurcation start point, the bifurcation end point, and the end point. We assume there are straight lines between the start point and the bifurcation start point
    and the bifurcation end point and the end point. 
    The bifurcation angle is then calculated as the angle between where the lines meet (which is assumed to not be on the path the agent travels, but would be the angle if the agent turned all the way in one step)
    Parameters: 
    xPos: the x positions of the agent
    yPos: the y positions of the agent
    headings: the headings of the agent
    thresh: a constant from 0 to 1 (should generally be close to 0) which controls what proporition of the maximum change in direction we will consider to be part of a bifurcation
    maxtime: the maximum time allowed in the simulation
    test: a parameter to control whether or not we print out some values
    Returns:
    true_indices: the approximate indices where bifurcations occur. Taken as the mean between the start and end indices of each valid bifurcation. Length equal to the number of valid bifurcations
    theoretical_angles: the angles calculated as the angle between the line in the direction before the bifurcation and the line in the direction after the bifurcation. Length equal to the number of valid bifurcations.
    '''
    # VALIDITY CHECKS / SETUP
    if np.shape(xPos)[0] == 1:
        xPos = xPos.ravel()
        yPos = yPos.ravel()
    if xPos.size != yPos.size:
        raise ValueError("arrays for x position and y position are different sizes")
    if headings.size != xPos.size:
        raise ValueError("array for heading and arrays for position are different sizes")
    total_time = len(xPos)
    if maxtime <= total_time:
        total_time = maxtime
        xPos = xPos[:total_time]
        yPos = yPos[:total_time]
        headings = headings[:total_time]
    x0, y0 = xPos[0], yPos[0]
    targetx, targety = xPos[-1], yPos[-1] # this implies we should only call the function if a target was reached...

    # FINDING THE START 
    dist_to_targ = np.sqrt((x0-targetx)**2+(y0-targety)**2)
    movement_thresh = dist_to_targ*0.05
    found_thresh = False
    thresh_ind = 0
    increment_search = 20
    if increment_search > total_time-1:
        increment_search = total_time-1
    while not found_thresh: 
        ind_upper = min(thresh_ind + increment_search, total_time-1)
        dist_from_start = np.sqrt((xPos[ind_upper-1]-x0)**2+(yPos[ind_upper-1]-y0)**2)
        if dist_from_start > movement_thresh:
            dists_sq  = (xPos[thresh_ind:ind_upper]-x0)**2 + (yPos[thresh_ind:ind_upper]-y0)**2
            mask = dists_sq > movement_thresh**2
            for i in range(len(mask)):
                if mask[i]:
                    thresh_ind += i
                    found_thresh = True
                    break
        else:
            thresh_ind = ind_upper
            if ind_upper == total_time-1:
                thresh_ind = total_time-1
                break     
    thresh_ind -= 1 #difference in length between xPos and dheading
    #print(f"thresh_ind: {thresh_ind}")

    # FINDING JUMP STARTS AND ENDS
    headings_unwrapped = np.unwrap(headings)
    dheadings = np.abs(np.diff(headings_unwrapped))
    thresh_value = np.pi/720 # approximately good value, may want to parametrize in the future
    above = dheadings[thresh_ind:] >= thresh_value
    if test:
        print(f"THRESH VALUE: {thresh_value}")
        print(f"heading: {headings}")
        print(f"direction unwrapped: {headings_unwrapped}")
        print(f"ddirection: {dheadings}")
        print(f"above: {above}")
        plt.figure(2)
        plt.plot(dheadings[thresh_ind:])
        plt.axhline(y=thresh_value, color='r', linestyle='--', linewidth=2)
        plt.title("dheadings absolute value")
        plt.show()
    jump_ends = []
    jump_starts = []
    in_jump = False
    for i, val in enumerate(above):
        if val and not in_jump: 
            jump_starts.append(i + thresh_ind)
            in_jump = True
        elif not val and in_jump:
            jump_ends.append(i + thresh_ind - 1)
            in_jump = False
    jump_starts.append(total_time-1)
    

    # FINDING THE THEORETICAL ANGLES FROM THE BIFURCATION REGIONS
    # THREE STEP PROCESS: first we calculate the direction at the start of the bifurcation region
    # then we iterate through these directions and if there is a bifurcation such that the next direction is close to the current direction, we ignore that bifurcation point
    # then we calculate the theoretical angles and midpoints between the start and end of the bifurcation for all the start/end point pairs we actually flagged
    init_angle = np.atan2(yPos[jump_starts[0]]-y0,xPos[jump_starts[0]]-x0)
    candidate_directions = [init_angle]
    for ind, bif_end in enumerate(jump_ends):
        next_angle = np.atan2(yPos[jump_starts[ind+1]]-yPos[bif_end],xPos[jump_starts[ind+1]]-xPos[bif_end])
        candidate_directions.append(next_angle)

    candidate_directions = np.unwrap(candidate_directions)
    #print(f"candidate directions: {candidate_directions}")
    #print(f"jump starts: {jump_starts}")
    #print(f"jump ends: {jump_ends}")
    final_directions = False
    while not final_directions:
        new_directions = []
        new_starts = []
        new_ends = []
        for index in range(0,len(candidate_directions)-1):
            direction_diff = np.abs(candidate_directions[index]-candidate_directions[index+1])
            if direction_diff > direction_thresh: # if directions are off, apppend this one
                new_directions.append(candidate_directions[index])
                new_starts.append(jump_starts[index])
                new_ends.append(jump_ends[index])
        candidate_directions = new_directions
        jump_starts = new_starts
        jump_ends = new_ends
        if len(new_directions) == len(candidate_directions):
            final_directions = True
            
    jump_starts.append(total_time-1)
    theoretical_angles = []
    prev_angle = np.atan2(yPos[jump_starts[0]]-y0,xPos[jump_starts[0]]-x0)
    for ind, bif_end in enumerate(jump_ends):
        next_angle = np.atan2(yPos[jump_starts[ind+1]]-yPos[bif_end],xPos[jump_starts[ind+1]]-xPos[bif_end])
        theoretical_angle = get_estimated_angle(prev_angle,next_angle)
        #print(f"prev angle: {prev_angle}, next angle: {next_angle}, theoretical angle: {theoretical_angle}")
        theoretical_angles.append(theoretical_angle)
        prev_angle = next_angle
    true_indices = []
    for index in range(len(jump_ends)):
        mean_index = (jump_ends[index]+jump_starts[index])/2
        true_indices.append(int(round(mean_index)))
    return true_indices, theoretical_angles

def find_bumps(activity): # function not currently in use, will keep it for now
    '''
    A function which takes in the activity data for a simulation and returns a list of where indices that are the peak of bumps at each time point in that simulation
    
    Parameters:
        activity: a N x tsteps array of neuron activity over the entire simulation

    Returns: 
        bump_list: a list of length tsteps which every entry is the indices of the peak of a bump at that time 
    '''
    activity_to_insert = activity[0:4,1:]
    activity_plus = np.concatenate([activity,activity_to_insert])
    tsteps = len(activity.iloc[0,:])
    bump_list = []
    for i in range(tsteps):
        indices_bumps, _ = find_peaks(activity_plus[:,i])
        true_bumps = [x for x in indices_bumps if x <= 100]
        bump_list.append(true_bumps)
    return bump_list

    
def plot_metric(metrics,x,colors,labels,fig,title,xlabel,ylabel): # Function not currently in use, but I will keep it for now
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


def plot_traj(xPos,yPos,targetsx,targetsy,sample_size,figure,plot_dec_point=False,dec_points=None,start_ind=0,end_ind=0):
    '''
    A function that takes x trajectories and y trajectories and plots them over each other, with a low ish opacity so we can see overlap. 
    Designed to be used over the same simulation settings with a number s of samples.

    Parameters:
        xPos: a list of lists of x positions of the agent where each sublist contains all the positions in one simulation run
        yPos: a list of lists of y positions of the agent where each sublist contains all the positions in one simulation run
        targetsx: a list of x positions of the targets
        targetsy: a list of y positions of the targets
        sample_size: the number of repitions of the same exact simulation settings
        figure: the figure to plot the trajectories in
        plot_dec_point: whether or not to plot the bifurcation points
        dec_points: a list of times where bifurcations are found, which will optionally plotted
        start_ind: the time index to start plotting
        end_ind: the time index to stop plotting (if 0 plot until the final time step)

    Returns: 
        None

    Expected output: 
        A line plot of the trajectories of the agent in blue that is darker on more commonly taken paths and lighter on less commonly taken paths. 
        The targets are plotted as small red dots, and if the bifurcation times are plotted then they will be larger, low opacity, green dots on the trajectory. 
    '''
    if end_ind == 0:
        for sample in range(sample_size):
            figure.plot(xPos[sample][start_ind:],yPos[sample][start_ind:],color='blue',alpha=0.5/sample_size)
            if plot_dec_point:
                dec_time = dec_points[sample]
                figure.scatter(xPos[sample][dec_time],yPos[sample][dec_time], color="green",alpha = 0.8,s=5)
        figure.scatter(targetsx,targetsy,color='red',s=5)
    else:
        for sample in range(sample_size):
            deltax = np.diff(xPos[sample][start_ind:end_ind])
            deltay = np.diff(yPos[sample][start_ind:end_ind])
            dist = np.zeros(len(xPos[sample][start_ind:end_ind]))
            dist[0] =0
            dist[1:] = np.sqrt(deltax**2 + deltay**2)
            figure.scatter(xPos[sample][start_ind:end_ind],yPos[sample][start_ind:end_ind],c = dist, cmap = 'afmhot_r', alpha=0.5,s=8)
            if plot_dec_point:
                dec_time = dec_points[sample]
                figure.scatter(xPos[sample][dec_time],yPos[sample][dec_time], color="green",alpha = 0.1)
    return None




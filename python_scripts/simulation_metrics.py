import numpy as np
from scipy.signal import find_peaks

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


def get_bump_type(initialxt,initialyt,xPos,yPos,activity):
    # UPDATED FROM WHAT WE USED FOR HIGH RESOLUTION SWEEPS - NEEDS TESTING 
    '''
    A function that analyses the neural activity of a simulation in small intervals 
    and returns a number corresponding to the number of bumps that best describe the simulation.
    Designed to handle three targets only right now.

    Paramters:
        initialxt: a list of initial x positions of the targets
        initialyt: a list of initial y positions of the targets
        xPos: a list of x positions over the simulation
        yPos: a list of y positions over the simulation
        activity: a dataframe of neuron activity over the simulation
        interval: how small of an interval to consider at a time. 
                Intervals are considered as a static point, so they should be small enough that in most cases the bump doesn't shift majorly over one interval

    Returns: 
        phase: a number to classify the number of bumps that best describes the simulation, possible outputs are 0, 1, 2, 3, or 4 (4 is undesireable).
               Usually the phase corresponds to the most common number of bumps present across all the intervals, but occasionally that is not the case.
    '''
    if xPos.size != yPos.size:
        raise ValueError("arrays for x position and y position are different sizes")
    if len(initialxt) != len(initialyt):
        raise ValueError("target x positions and target y positions are different sizes")
    if xPos.size != activity.shape[1]:
        raise ValueError("position arrays and activity arrays are different sizes")
    
    total_time = activity.shape[1]
    rng = np.random.default_rng()
    intervals = []
    sum_intervals = 0
    while sum_intervals < total_time:
        rand_int = rng.integers(5, 16)
        intervals.append(rand_int)
        sum_intervals += rand_int

    counters = np.zeros(4, dtype=np.int32)
    start = 0
    for item in intervals:
        end = min(start+item,total_time)
        print(f"start: {start}, end: {end}")
        x0 = xPos[start]
        y0 = yPos[start]
        angles = np.atan2(initialyt - y0, initialxt - x0)
        indices = np.round(100 * (angles / (2 * np.pi))).astype(int) % 100
        act_int = activity[indices, start:end]
        bump_means = act_int.mean(axis=1) 
        npositive = int(np.sum(bump_means > 0))
        if npositive > 1:
            n0, n1, n2 = indices[0], indices[1], indices[2]
            b0, b1, b2 = bump_means[0], bump_means[1], bump_means[2]
            act_slice = activity[:, start:end]

            def between_min(na, nb):
                diff = na - nb
                if 0 <= diff < 50:
                    return act_slice[nb+1:na].min() if nb+1 < na else 0.0
                elif diff >= 50:
                    part = np.concatenate([act_slice[:nb], act_slice[na:]])
                    return part.min() if part.size > 0 else 0.0
                else:
                    return act_slice[n0+1:nb].min() if n0+1 < nb else 0.0 
                
            if b0 > 0 and b1 > 0:
                if between_min(n0, n1) >= 0:
                    npositive -= 1

            if b1 > 0 and b2 > 0:
                if between_min(n1, n2) >= 0:
                    npositive -= 1
        
        counters[npositive] += 1
        start += item
        
    proportions = counters / sum(counters)
    print(f"proportions: {proportions}")
    phase = int(np.argmax(proportions))
    return phase

def get_bifurcation_angle(xPos, yPos, targetx, targety, targ_angle):
    '''
    A function that calculates the local extrema of the angle between the agent and the target, and uses them to 
    find the ratio of the difference between the angle of the agent at the bifurcation and the most direct path to the target

    Parameters:
        xPos: a list of x positions from a simulation
        yPos: a list of y positions from a simulation
        targetx: the x position of the target that the agent reached, -1 is expected if no target was reached
        targety: the y position of the target that the agent reached, -1 is expected if no target was reached
        targ_angle: the angle between the targets, relative to the agents starting position

    Returns: 
        best_overall: a ratio representing the difference between the angle of the agent at the bifurcation and the most direct path. 
                    Calculated by taking both the local max and local mins of the angle, finding the first one where the agent has moved more than 1 total unit, 
                    then seeing if the difference between the most direct path and the first local max or first local min is bigger.
    '''
    if targetx < 0 or targety < 0:
        return 0
    xPos = np.asarray(xPos)
    yPos = np.asarray(yPos)
    angles_to_target = np.atan2(np.abs(targety-yPos),np.abs(targetx-xPos))
    initial_angle = angles_to_target[0]
    peaks_t, _ = find_peaks(angles_to_target)
    negative_peaks_t, _ = find_peaks(-angles_to_target)
    x0, y0 = xPos[0], yPos[0]
    best_positive = 0
    best_negative = 0

    def first_valid_ratio(peak_indices):
        if len(peak_indices) == 0:
            return 0
        dx_start = xPos[peak_indices] - x0
        dy_start = yPos[peak_indices] - y0
        dx_end   = xPos[peak_indices] - targetx
        dy_end   = yPos[peak_indices] - targety

        start_dist_sq = dx_start**2 + dy_start**2
        end_dist_sq   = dx_end**2   + dy_end**2

        valid = np.where((start_dist_sq > 1) & (end_dist_sq > 1))[0]
        if len(valid) == 0:
            return 0

        first = peak_indices[valid[0]]
        return np.abs(angles_to_target[first] - initial_angle) / targ_angle

    best_positive = first_valid_ratio(peaks_t)
    best_negative = first_valid_ratio(negative_peaks_t)
    best_overall = max(best_negative,best_positive)
    return best_overall

def find_bumps(activity): # function not currently use, will keep it for now
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

    Returns: 
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
            if plot_dec_point:
                dec_time = dec_points[sample]
                figure.scatter(xPos[sample][dec_time],yPos[sample][dec_time], color="green",alpha = 0.1)
    return None




import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# -------- Getting metrics --------

#so with what we returned from running the sim we can define a function to get metrics
def get_destination_metrics(xPos, yPos, targetsx, targetsy , stopping_distance = 0.1):
    '''
    A function which takes the outputs from the simulate_ringattractor code and gets a few metrics to quantify what happened in the simulation
    Inputs:
    distances: numpy array of distances at each point from the simulate_ringattractor code
    decision_precision: the distance cutoff for an agent to have considered "arrived" at a target
    Returns:
    final_target: the final target the agent ends up at (will expand to be a list for multiple agents), -1 if the agent doesn't end up at a target
    time_target_reached: the time the agent (expand to multiple) got within the decision_precision of the final target
    '''
    # TO DO: also return a starts moving time when the distance from the initial position is greater than decision precision
    # find which target the agent ends at
    final_x = xPos[0,-1]
    final_y = yPos[0,-1]
    final_target = -1
    ntargets = len(targetsx[:,0])
    for targ in range(ntargets):
        targ_x = targetsx[targ,-1]
        targ_y = targetsy[targ,-1]
        x_dist = np.abs(final_x - targ_x)
        y_dist = np.abs(final_y - targ_y)
        total_dist = np.sqrt((x_dist**2)+(y_dist**2))
        if total_dist <= stopping_distance:
            final_target = targ

    # find the first time the agent gets within the decision boundary for the target it ends up at
    # made a bit unnecessary if stop == True, but we'll keep it in for if stop == False

    time_target_reached = 0
    for index in range(len(xPos[0,:])):
        if final_target == -1:
            break
        agent_x = xPos[0,index]
        agent_y = yPos[0,index]
        target_x = targetsx[final_target,index]
        target_y = targetsy[final_target,index]
        distance = np.sqrt((agent_x - target_x)**2 + (agent_y-target_y)**2)
        if distance < stopping_distance:
            time_target_reached = index
            break

    # finding the time the agent starts moving
    # first calculate the initial closest distance and use that as decision boundary?
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
        
    return final_target, time_target_reached, movement_start

def get_direction_info(headings, n_peaks, start_step=200, dest_step=5000):
    '''
    A function which takes the agent's heading at each time step and determines when it turns towards one target
    This will depend on the initial geometry, so we might need to take that as a parameter
    First we'll try to just detect changes and see if that's adequate
    '''
    #need to add multiple agent functionality
    # need to find a way to handle multiple decisions
    # problem is that because there is some noise we can't just take the first time it turns
    # as the decision
    headings = headings[0,:]
    peaks, properties = find_peaks(headings, prominence=0)
    prominences = properties['prominences']
    top_n_indices = np.argsort(prominences)[-n_peaks:]
    top_n_peaks = peaks[top_n_indices]
    '''
    max_angle_diff = 0
    dec_ind = 0
    for i in range(start_step,dest_step): 
        if headings_diff[i] > max_angle_diff:
            max_angle_diff = headings_diff[i]
            dec_ind = i
    '''
    return top_n_peaks


def get_min_distance(xPos, yPos, targetXpos, targetYpos, even=True, better_ind=-1):
    tsteps = len(xPos[0,:])
    dists = np.sqrt((xPos[0, :] - targetXpos[:, :tsteps])**2 + 
                (yPos[0, :] - targetYpos[:, :tsteps])**2)
    sum_dists = dists.sum(axis=0)
    min_dists = dists[better_ind] if not even else dists.min(axis=0)
    min_over_sum = min_dists / sum_dists
    total_min = min_over_sum.min()
    return total_min

def plot_trajectories(xtraj, ytraj, targetsx, targetsy, colors, L, title):
    '''
    plots n different trajectories in n different colors
    xtraj: a n x t numpy array of x positions
    ytraj a n x t numpy array of y positions
    colors: a length n list of colors
    L: length of the grid
    '''
    # this works alright for getting an idea of how uneven geometries are failing if they're in between, but doesn't do great for even trajectories where
    # success should actually be somewhere in the middle most of the time
    # maybe instead want to plot 'maximum' trajectory (largest endpoint in y), 'minimum' trajectory (smallest endpoint in y) and 'median' trajectory (median endpoint in y)
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.scatter(
            targetsx,
            targetsy,
            s = 10,
            marker = 's',
            c = [[0.8, 0, 0.2]], 
        )
    for traj in range(len(xtraj[:,0])):
        plt.scatter(
            xtraj[traj,:],
            ytraj[traj,:],
            s = 10,
            color = colors[traj]
        )
    plt.title(title)
    plt.show(block=False)
    plt.pause(0.001)  


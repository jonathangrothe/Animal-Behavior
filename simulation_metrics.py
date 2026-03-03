import numpy as np

# -------- Getting metrics --------

#so with what we returned from running the sim we can define a function to get metrics
def get_destination_metrics(distances, decision_precision = 1):
    '''
    A function which takes the outputs from the simulate_ringattractor code and gets a few metrics to quantify what happened in the simulation
    Inputs:
    distances: numpy array of distances at each point from the simulate_ringattractor code
    decision_precision: the distance cutoff for an agent to have considered "arrived" at a target
    Returns:
    final_target: the final target the agent ends up at (will expand to be a list for multiple agents), -1 if the agent doesn't end up at a target
    time_target_reached: the time the agent (expand to multiple) got within the decision_precision of the final target
    '''
    # find which target the agent ends at
    final_distances = distances[-1,:]
    final_target = -1
    for index in range(len(final_distances)):
        if final_distances[index] < decision_precision:
            final_target = index


    # find the first time the agent gets within the decision boundary for the target it ends up at
    time_target_reached = 0
    for index in range(len(distances)):
        if distances[index, final_target] < decision_precision:
            time_target_reached = index
            break

   
    return final_target, time_target_reached

def get_distance_info(distances,xPos,yPos,targetsx,targetsy):

    nsteps = len(xPos[0,:])
    ntargets = len(targetsx[:,0])
    nagents = len(xPos[:,0])
    # using the positional information to quantify how the agent is moving relative to each target
    # still a work in progress to make this meaningful

    deltas = np.zeros((nsteps,(ntargets*2)))
    directions = np.zeros((nsteps, (ntargets*2)))
    for time in range(nsteps):
        for a in range(nagents):
            for targ in range(ntargets):
                delta_x = targetsx[targ,time]-xPos[a,time]
                delta_y = targetsy[targ,time]-yPos[a,time]
                mag = np.sqrt(delta_x**2+delta_y**2)
                deltas[time,targ*2] = delta_x
                deltas[time,targ*2+1] = delta_y
                directions[time,targ*2] = delta_x/mag
                directions[time,targ*2+1] = delta_y/mag
    return deltas, directions



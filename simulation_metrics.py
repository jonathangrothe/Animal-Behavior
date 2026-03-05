import numpy as np

# -------- Getting metrics --------

#so with what we returned from running the sim we can define a function to get metrics
def get_destination_metrics(xPos, yPos, targetsx, targetsy , decision_precision = 1):
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
    final_targets_x = targetsx[:,-1]
    final_targets_y = targetsy[:,-1]
    final_target = -1
    ntargets = len(targetsx[:,0])
    for index in range(ntargets):
        print(f"distance y: {np.abs(final_y - final_targets_y[index])}")
        if (np.abs(final_x - final_targets_x[index])) < decision_precision and (np.abs(final_y - final_targets_y[index]) < decision_precision):
            final_target = index


    # find the first time the agent gets within the decision boundary for the target it ends up at

    time_target_reached = 0
    for index in range(len(xPos[0,:])):
        if final_target == -1:
            break
        agent_x = xPos[0,index]
        agent_y = yPos[0,index]
        target_x = targetsx[final_target,index]
        target_y = targetsy[final_target,index]
        distance = np.sqrt((agent_x - target_x)**2 + (agent_y-target_y)**2)
        if distance < decision_precision:
            time_target_reached = index
            break

    # finding the time the agent starts moving. 
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

def get_direction_info(headings, delta_cutoff = 0.1):
    '''
    A function which takes the agent's heading at each time step and determines when it turns towards one target
    This will depend on the initial geometry, so we might need to take that as a parameter
    First we'll try to just detect changes and see if that's adequate
    '''
    #need to add multiple agent functionality, 
    #FILLER STATEMENT to get around that for now
    headings = headings[0,:]
    tsteps = len(headings)
    init_heading = headings[0]
    prev_angle = init_heading
    print(f"init angle: {prev_angle}")
    #before we mess with this too much we probably need to check two things:
    #A: the agent has actually started to move - we should get this in get_destination_metrics
    #B: the agent has not yet arrived at the target 
    for i in range(1,tsteps):
        angle = headings[i]
        if np.abs(angle-prev_angle) > delta_cutoff:
            x=1
        


    return "test"


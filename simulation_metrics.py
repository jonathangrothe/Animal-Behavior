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
    # TO DO: also return a starts moving time when the distance from the initial position is greater than decision precision
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
            print("change in angle flagged")
        


    return "test"


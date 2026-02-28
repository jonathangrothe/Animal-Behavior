import simulate_ringattractor
import numpy as np
import matplotlib.pyplot as plt

# --------  PARAMETERS --------

# --- Setting up the grid ---

#number of neurons in the alpha ring
N = 100

#total size of the grid
L = 100

# --- Geometry-based parameters to change ---

#number of targets
ntargets = 2

#number of agents: 
nagents = 1

#setting up the agent
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = L/2 
    initialy[a] = L/2

#setting up the targets
#radius = 20
#initialxt, initialyt = simulate_ringattractor.create_grid(ntargets, 4, L)
initialxt = [L-30, L-30]
initialyt = [L/2+15, L/2-15]

# --- Setting up the simulation ---

#number of time steps
T = 5000

#rego
rEgo = 0

#in the original code the simulation is run 5 times with [0, 0.5, 1, 2, 4] as values
rEgoTarget = 0

#egonumber
Egonumber = 1

#collision avoidance:
hColl = -10
rColl = 0

#in the original code the simulation runs it on 0 once and then 1 five times
distf = 1

#this determines the "characteristic length of signal decay",
#original code iterates through [1, 1, 2, 4, 8, 16]
adistf = 1

#step size
dt = 0.1

#v0: velocity I believe
v0 = 0.05

#v0t: velocity of targets I believe
#in the original code it is initialized to zero but that doesn't really make sense to me
v0t = np.zeros(ntargets)
for i in range(ntargets):
    v0t[i] = 0

nu = 0.5
theta = np.linspace(0,2*np.pi,N+1)
theta = theta[:-1]
J = np.zeros((N,N))
for i in range(N):
    deltah = np.abs(theta - theta[i])
    deltah = np.pi-np.abs(np.pi-deltah)
    J[i,:] = np.cos(np.pi*(deltah/np.pi)**nu)
    J[i,i] = 0.0
    J = np.squeeze(J)



# --- Details of the simulation to change ---

#allocentric flag
allocentricFlag = [0]

#attraction
h0s = [0.4]

#hbase
h_b = [0.2]

#width of the gauss bump 
sigma = [0.2]

#noise parameter 
beta = [100]

#ego distance? (don't remember exactly which one this is)

# -------- Getting metrics --------

#so with what we returned from running the sim we can define a function to get metrics
def get_metrics(distances, xPos, yPos, targetsx, targetsy, decision_precision = 1):
    # first find where the agent ended and if it is 'close' to a target
    # if yes,
    # iterate through distances, get index of biggest jump  
    # have to do this for each comparison between targets 
    # ie: for 3 targets: when did agent decide between A and B, A and C, and B and C
    ntargets = len(targetsx[:,0])
    nagents = len(xPos[:,0])
    #first we'll find out which target we ended up at
    final_xpos = xPos[:,-1]
    final_ypos = yPos[:,-1]
    final_target_xpos = targetsx[:,-1]
    final_target_ypos = targetsy[:,-1]
    final_target = -1
    for a in range(nagents):
        for t in range(ntargets):
            if ((final_xpos[a] - final_target_xpos[t]) < decision_precision) and ((final_ypos[a] - final_target_ypos[t]) < decision_precision):
                final_target = t

    #at what time does the agent reach the target?
    time_target_reached = 0
    for index in range(len(distances)):
        if distances[index, t] < decision_precision:
            time_target_reached = index
            break

            
    
    # first we will iterate through and find all the changes in distances (index 0 is change from t0 to t1) 
    # this will help us check to see when the agent stops moving towards a target or when it accelerates towards a target
    # important to note that the indcies are NOT evenly spaced - there will be a lot more movement at the beginning     
    distance_diff = np.zeros((T-1,ntargets))
    for time in range(len(distances)-1):
        prev_distances = distances[time-1,:]
        curr_distances = distances[time,:]
        for target in range(ntargets):
            change = curr_distances[target] - prev_distances[target]
            distance_diff[time,target] = change
   
   # finding the time point when the agent never moves towards a target,
   # this can be considered the elimination point of that target
   # we will only go up to the time the target is reached, 
   # as there is a lot of random noisy movement after the agent reaches the target
    eliminated_indices = np.zeros(ntargets)
    for index in range(time_target_reached):
        if index > time_target_reached - 10:
            print(f"index: {index}, distance differences: {distance_diff[index,:]}, distances: {distances[index-1,:]}")
        for t in range(ntargets):
            #need to check if we're within a certain absolute distance of the target
            #WANT: change the distance cutoff to something else to reduce how this is affected by noise...
            #Still really hard to define exactly when the decision happens because there are so many time steps close to the 'decision'
            if (distance_diff[index][t] > 0) and (eliminated_indices[t] == 0):
                eliminated_indices[t] = index+1
            if distance_diff[index][t] < 0:
                eliminated_indices[t] = 0


    '''
    max_towards_target = 0
    decision_point = -1
    for index in range(len(distance_diff)):
        #check if one diff is negative all else are positive
        #store the distance in the negative direction
        #check to see if the
        n_negative = 0
        negative_index = -1
        for distance in range(len(distance_diff[index])):
            if distance_diff[index][distance] < 0:
                n_negative += 1
                negative_index = distance
        if n_negative == 1:
            if index < 5:
                print(f"entered at index{index}")
                print(distance_diff[index])
            negative_distance = distance_diff[index][negative_index]
            if negative_distance < max_towards_target:
                max_towards_target = distance_diff[index][negative_index]
                decision_point = index+1
                '''
    return final_target, time_target_reached, eliminated_indices


# -------- Running the simulation --------

#run it with both allo and ego centric orientations
for orientation in range(len(allocentricFlag)):
    #run it with each level of attraction
    for h in range(len(h0s)):
        h0 = np.zeros(ntargets+nagents)
        for i in range(ntargets):
            h0[i] = h0s[h]
        for a in range(nagents):
            h0[ntargets+a] = h0s[h]
        #run it with each h base level
        for hb in range(len(h_b)):
            #run it with each sigma level
            for s in range(len(sigma)):
                #run it with each beta level
                for b in range(len(beta)):
                    distances, xPos, yPos, targetsx, targetsy = simulate_ringattractor.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag[orientation],rEgo,rEgoTarget,Egonumber,
                                distf,adistf,J,beta[b],h0,h_b[hb],dt,v0,v0t,sigma[s],hColl,rColl,
                                initialx,initialy,initialxt,initialyt)
                    plt.figure(2)
                    plt.plot(distances)
                    plt.ylabel("Absolute value of the difference between distance agent and each target")
                    plt.xlabel("Time step")
                    plt.title(f"Difference in distance between agent and each target over time\n allocentric:{allocentricFlag[orientation]}, h0: {h0}, beta : {beta[b]}")
                    plt.show()
                    final_decision, time_reached, eliminated = get_metrics(distances, xPos, yPos, targetsx, targetsy)
                    print(f"ended at target: {final_decision}")
                    print(f"the time when the agent reached that target: {time_reached}")
                    print(f"the time when each target is eliminated: {eliminated}")
                    



'''
next steps: 
-take out plotting each time (for now)
-run it repeatedly (50 times?) with the same settings
-record performance metrics for each run 
-do this again with slight changes to parameters
-repeat everything with a slight change in geometry 

Overall workflow:
Even spacing: 
choose 10 (?) sets of parameters
run the sim with each set of parameters 50 (?) times
(~500 sims)

Move a target in one direction: 
run the same settings again
(~500 sims)

Move the target in the same direction further 1-2 more times

Move the target in a different direction 3-4 times

Repeat for as many directions as possible

For each sim we will get 5ish performance metrics
for each set of parameters we will get 50ish datapoints (50x5)
for each geometry we will have 10 sets of parameters (10x50x5)
for each directional comparison we will get 4-5 levels (5 levels x 10 parameter sets x 50 simulations x 5 metrics)
'''

'''
How do I want to implement this scale? 
We definitely can easily create a function that repeats the simulate_ringattractor function x times, 
returns a dataframe or np array with our metrics - like get_metrics or something
We then can create a different function that does this for over changing parameters 
(kind of like the nested for loop I have going now) 
We can then create a different function that runs the previous function over similar geometries
Maybe we input an initial geometry, a direction to change and a unit to change and a number of sims to run?
To ensure that we are making as small of a change in geometry at each step?


Once we have all of this, we can see which set of parameters at each geometry perform best over the 50 sims we run at that level
and which parameters produce the biggest changes all else being equal (and maybe interaction of parameters too)
'''





#produce unequal positions, see how often it chooses the "best option"
#proportion of time going to the right option
#geometry and number of decisions affect accuracy
#to what extent model behaves like empirical data

#variables to look at:
#p(decision is made) 
#time to decision making 
#time spent close to an option 
#min distance to option
#complexity measures: entropy
#trajectories as functions of parameters

#important: track distance between each (or at least top 2) target and agent, see when agent clearly favors one
#have a metric to determine if no decision was made (difference in distance between top 2 < cutoff)
#entropy/complexity of geometry might be harder, look into that more 

#other problems to solve: 
#if we plot everything this will take a super long time to run 
#generate uneven geometries efficeintly
#storing our derived metrics (most importantly for now distance) and plotting them

#parameters
#hbase
#sigma
#beta
#allo/ego
#egocentric distance
#h0s
#distf: distance until switch to ego
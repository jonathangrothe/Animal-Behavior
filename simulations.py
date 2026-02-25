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
ntargets = 3

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
initialxt = [L-30, L-30, L/2-15]
initialyt = [L/2+15, L/2-15,L/2+8]

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
allocentricFlag = [0,1]

#attraction
h0s = [0.4]

#hbase
h_b = [0.2]

#width of the gauss bump 
sigma = [0.2]

#noise parameter 
beta = [100]

#ego distance? (don't remember exactly which one this is)

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
                #run it with each beeta level
                for b in range(len(beta)):
                    distances = simulate_ringattractor.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag[orientation],rEgo,rEgoTarget,Egonumber,
                                distf,adistf,J,beta[b],h0,h_b[hb],dt,v0,v0t,sigma[s],hColl,rColl,
                                initialx,initialy,initialxt,initialyt)
                    #now we can go through the distances and find when the "decision" happens
                    #now we need to extend this to the case of multiple targets
                    diff_list = []
                    for dist in range(len(distances)):
                        dist_meas = min(distances[dist])
                        #print(f"time step: {dist},distances: {distances[dist]}, difference: {dist_diff}")
                        diff_list.append(dist_meas)
                    plt.figure(2)
                    plt.plot(diff_list)
                    plt.ylabel("Absolute value of the difference between distance from agent to target A and distance from agent to target B")
                    plt.xlabel("Time step")
                    plt.title(f"Difference in distance between agent and each target over time\n allocentric:{allocentricFlag[orientation]}, h0: {h0}, beta : {beta[b]}")
                    plt.show()

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
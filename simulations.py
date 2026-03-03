import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor
import simulation_metrics

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
allocentricFlag = 0,1

#attraction
h0s = 0.4

#hbase
h_b = 0.2

#width of the gauss bump 
sigma = 0.2

#noise parameter 
beta = 100

# data we will collect each time we run the simulation
targets_reached = []
time_to_target = []
allo = []
attraction = []

# number of times to run the simulation
n_samples = 10

# -------- Running the simulation --------

# first we will sweep the parameter space to at least get an interesting set of parameters to work with

adequate = False
iterations = 0
while adequate == False:
    for sample in range(n_samples):
        h0 = np.zeros(ntargets+nagents)
        for i in range(ntargets):
            h0[i] = h0s
        distances, xPos, yPos, targetsx, targetsy = simulate_ringattractor.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag,rEgo,rEgoTarget,Egonumber,
                                    distf,adistf,J,beta,h0,h_b,dt,v0,v0t,sigma,hColl,rColl,
                                    initialx,initialy,initialxt,initialyt,True)
        final_decision, time_reached = simulation_metrics.get_destination_metrics(distances)
        print(final_decision)
        print(time_reached)
        targets_reached.append(final_decision)
        time_to_target.append(time_reached)
        notreached = 0
    for item in targets_reached:
        if item == -1:
            notreached += 1
    print(notreached)
    if notreached <= 2:
        adequate = True
    else:
        #can increase h0 - attraction - mean of gaussian
        #and adjust hb correspondingly - base
        #can adjust sigma - variance of gaussian
        #can adjust beta - noise but we're already pretty high so I think we're good here
        #we'll try just alternating by slightly increasing h0 and slightly decreasing sigma
        if iterations % 2 == 0:
            for i in range(ntargets):
                h0[i] = h0[i] + 0.02
        else:
            sigma = sigma -0.02

print(f"h0: {h0}")
print(f"sigma: {sigma}")

#ok now it just only goes up IDK why, 
# kinda want stronger attraction because even though it's reaching a target,
# the bifuraction is really late
# to get this to work I will find the angle of the agent relative to targets.
# hopefully this will give me moment of decision. 
# while doing this I will make the data the simulation code collects more efficient.


#this is running the sim with a bunch of changes in variables - 
# we'll do this later and I imagine we will focus more on geometry than on changing parameters
'''
for sim in range(n_samples):
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
                                    initialx,initialy,initialxt,initialyt,False)
                        
                        plt.figure(2)
                        plt.plot(distances)
                        plt.ylabel("Distance between agent and each target")
                        plt.xlabel("Time step")
                        plt.title(f"Difference in distance between agent and each target over time\n allocentric:{allocentricFlag[orientation]}, h0: {h0}, beta : {beta[b]}")
                        plt.show(block=False)
                        
                        final_decision, time_reached = simulation_metrics.get_destination_metrics(distances)
                        targets_reached.append(final_decision)
                        time_to_target.append(time_reached)
                        allo.append(allocentricFlag[orientation])
                        attraction.append(h0s[h])
    
sim_data = {'Final target': targets_reached,'Time to target': time_to_target, 'Allocentric or egocentric': allo, 'Attraction': attraction}  
sim_df = pd.DataFrame(sim_data)
print(sim_df)   
'''   


# goals: rework distances so that we are only getting raw data, getting distances in a different step
# sweep the parameter space based on the metrics: ie: if we're not getting close to a target, update the parameters so that we do

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
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
ntargets = 4 # sims 1 and 2
# ntargets = 5 # sim 3

#number of agents: 
nagents = 1

#setting up the agent
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    '''
    initialx[a] = 5 # sims 2 and 3
    initialy[a] = 5
    '''
    initialx[a] = L/2 - 30 # sim 1
    initialy[a] = L/2
    
    

#setting up the targets
'''
initialxt = [10, 45, 75, 70, 95] # sim 3
initialyt = [70, 75, 45, 10, 85]

initialxt = [12, 45, 75, 70] # sim 2
initialyt = [70, 75, 45, 10]
'''
initialxt = [L-15, L-15, L-15, L-15] # sim 1
initialyt = [L/2+30, L/2-30, L/2+20, L/2-20]



# --- Setting up the simulation ---

#number of time steps
T = 5000

#periodic flag
periodicflag = 0

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
# h0s = [[0.85,0.85,0.85,0.85,0.15]] # sim 3
# h0s = [[0.4,0.5,0.6,0.2,0]] # sim 2
h0s = [[0.47,0.47,-0.05,0.01,0], [0.67,0.47,-0.2,0.01,0]] # sim 1

#hbase
h_b = [0.2]

#width of the gauss bump 
# sigma = [0.6] # sim 2
sigma = [0.5]

#noise parameter 
beta = [100]

# data we will collect each time we run the simulation
targets_reached = []
time_to_target = []
movement_starts = []

decision1_time = []
decision2_time = []

allo = []
attraction = []
h_bs = []
sigmas = []

# number of times to run the simulation
n_samples = 3

# -------- Running the simulation --------

#this is running the sim with a bunch of changes in variables

for sim in range(n_samples):
    for orientation in range(len(allocentricFlag)):
        for h in range(len(h0s)):
            h0 = h0s[h]
            for hb in range(len(h_b)):
                for s in range(len(sigma)):
                    for b in range(len(beta)):
                        headings, xPos, yPos, targetsx, targetsy = simulate_ringattractor.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag[orientation],periodicflag,rEgo,rEgoTarget,Egonumber,
                                    distf,adistf,J,beta[b],h0,h_b[hb],dt,v0,v0t,sigma[s],hColl,rColl,
                                    initialx,initialy,initialxt,initialyt,True,True)
                        
                        final_decision, time_reached, movement_start = simulation_metrics.get_destination_metrics(xPos,yPos,targetsx,targetsy)
                        decision_indices = simulation_metrics.get_direction_info(headings,movement_start,time_reached,2)
                        movement_starts.append(movement_start)
                        targets_reached.append(final_decision)
                        time_to_target.append(time_reached)
                        if len(decision_indices) > 0:
                            decision1_time.append(decision_indices[0])
                        else:
                            decision1_time.append(-1)
                        if len(decision_indices) > 1:
                            decision2_time.append(decision_indices[1])
                        else:
                            decision2_time.append(-1)
                        allo.append(allocentricFlag[orientation])
                        attraction.append(h0s[h])
                        h_bs.append(h_b[hb])
                        sigmas.append(sigma[s])

          

sim_data = {'Final target': targets_reached,'Time to target': time_to_target, 'Movement starts': movement_starts, 'First decision time': decision1_time, 'Second decision time': decision2_time, 'Allocentric or egocentric': allo, 'Attraction': attraction, 'H_b': h_bs, 'sigma': sigmas}
sim_df = pd.DataFrame(sim_data)
pd.set_option('display.max_columns', None)
sim_df.to_csv("simulation_resuts.csv") 



# goals: rework distances so that we are only getting raw data, getting distances in a different step
# sweep the parameter space based on the metrics: ie: if we're not getting close to a target, update the parameters so that we do

#produce unequal positions, see how often it chooses the "best option"
#proportion of time going to the right option
#geometry and number of decisions affect accuracy
#to what extent model behaves like empirical data

#variables to look at:
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
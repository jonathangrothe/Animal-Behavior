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
ntargets = 12

#number of agents: 
nagents = 1

#setting up the agent
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = L/2 
    initialy[a] = L/2

#setting up the targets
radius = 20
initialxt, initialyt = simulate_ringattractor.create_grid(ntargets, 4, L)

# --- Setting up the simulation ---

#number of time steps
T = 1000

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

#dt: step size I believe
dt = 0.1

#v0: velocity I believe
v0 = 0.05

#v0t: velocity of targets I believe
#in the original code it is initialized to zero but that doesn't really make sense to me
v0t = np.zeros(ntargets)
for i in range(ntargets):
    v0t[i] = 0.01

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
h0s = [0.4,0.45,0.5]

#hbase
h_b = [0,0.1,0.2]

#width of the gauss bump 
sigma = [0.1,0.2,0.3]

#noise parameter 
beta = [40,60,80,100]

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
                    simulate_ringattractor.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag[orientation],rEgo,rEgoTarget,Egonumber,
                                distf,adistf,J,beta[b],h0,h_b[hb],dt,v0,v0t,sigma[s],hColl,rColl,
                                initialx,initialy,initialxt,initialyt)



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
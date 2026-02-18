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
ntargets = 30

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
initialxt, initialyt = simulate_ringattractor.create_circle(ntargets, radius, L)

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
allocentricFlag = 0

#attraction
h0s = [0.4]

#hbase
h_b = 0

#width of the gauss bump 
sigma = 0.2

#noise parameter 
beta = 80

#ego distance? (don't remember exactly which one this is)

for h in range(len(h0s)):
    h0 = np.zeros(ntargets+nagents)
    for i in range(ntargets):
        h0[i] = h0s[h]
    for a in range(nagents):
        h0[ntargets+a] = h0s[h]
    simulate_ringattractor.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag,rEgo,rEgoTarget,Egonumber,
                            distf,adistf,J,beta,h0,h_b,dt,v0,v0t,sigma,hColl,rColl,
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

#parameters
#hbase
#sigma
#beta
#allo/ego
#egocentric distance
#h0s
#distf: distance until switch to ego
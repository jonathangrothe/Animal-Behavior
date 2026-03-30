import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor
import simulation_metrics
import time

# --------  PARAMETERS --------

# --- Setting up the grid ---

# number of neurons in the alpha ring
N = 100

# total size of the grid
L = 100

# --- Geometry-based parameters to change ---

ntargets = 2
 
nagents = 1

initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 20
    initialy[a] = 50
    
initialxt = [80,80]
initialyt = [20,80]

# --- Setting up the simulation ---

# number of time steps
T = 5000

periodicflag = 0

rEgo = 0

# in the original code the simulation is run 5 times with [0, 0.5, 1, 2, 4] as values
rEgoTarget = 0

Egonumber = 1

# collision avoidance:
hColl = -10
rColl = 0

# in the original code the simulation runs it on 0 once and then 1 five times
distf = 1

# original code iterates through [1, 1, 2, 4, 8, 16]
adistf = 1

dt = 0.1

v0 = 0.05

# not moving targets for now
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

# 1 is allo, 0 is ego
allocentricFlag = [1]

# attraction vector, first are attraction for targets, then agents
h0s = [[0.45,0.45]]

h_b = np.linspace(0,0.5,num=20)

# width of the gauss bump 
sigma = [0.5]

# noise parameter 
beta = [100]

# data we will collect each time we run the simulation
# currently not collecting this
targets_reached = []
time_to_target = []
movement_starts = []
mean_distance = []

decision1_time = []
decision2_time = []

# number of times to run the simulation
n_samples = 50

# -------- Running the simulation --------

# this is running the sim with a bunch of changes in variables


for orientation in range(len(allocentricFlag)):
    for h in range(len(h0s)):
        h0 = h0s[h]
        for hb in range(len(h_b)):
            print(f"hb: {hb}")
            for s in range(len(sigma)):
                for b in range(len(beta)):
                    for sim in range(n_samples):
                        min_distance = []
                        headings, xPos, yPos, targetsx, targetsy = simulate_ringattractor.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag[orientation],periodicflag,rEgo,rEgoTarget,Egonumber,
                                    distf,adistf,J,beta[b],h0,h_b[hb],dt,v0,v0t,sigma[s],hColl,rColl,
                                    initialx,initialy,initialxt,initialyt,False,False)
                        success_measure = simulation_metrics.get_min_distance(xPos, yPos, targetsx, targetsy, False)
                        min_distance.append(success_measure)
                    mean_distance.append(np.mean(min_distance))


# plot x axis as param of interest, plot y axis as success measure
plt.figure(1)
plt.plot(h_b, mean_distance)
plt.title("Success over different base attraction values (even attraction, allocentric)")
plt.xlabel("Base attraction")
plt.ylabel("Smallest distance to closer target (over sum of distances)")
plt.show()

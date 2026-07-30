# goal for this: 
# get an idea of the relationship between angle and sigma and distance and h0
# start with a basic circle and then see how that generalizes

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from . import simulation_metrics as sim_met
from . import repeated_sims
from . import simulate_ringattractor as sim_ra

start_time = time.perf_counter()
r = 20
L = 100 
ntargets = 10
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 50
    initialy[a] = 50
initialxt = []
initialyt = []
evenly_spaced = np.linspace(0,2*np.pi,ntargets+1)
indices_of_interest = [3,4,7,8]
for n in range(ntargets):
    x = 50 + r*np.cos(evenly_spaced[n])
    y = 50 + r*np.sin(evenly_spaced[n])
    if n in indices_of_interest:
        initialxt.append(x)
        initialyt.append(y)
print(f"initialxt: {initialxt}")
ntargets = len(initialxt)
print(f"ntargets: {ntargets}")
T = 5000
periodicflag = 0
rEgo = 0 
rEgoTarget = 0
Egonumber = 1
hColl = -10
rColl = 0
distf = 0
adistf = 1
dt = 0.1
v0 = 0.05
v0t = np.zeros(ntargets)
for i in range(ntargets):
    v0t[i] = 0

N = 100
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

allocentricFlag = 1
h0s = [0.25]*10
h_b = 0.2
sigmas = np.linspace(0.29,0.33,num=10)
beta = 100

for item in sigmas: 
    headings, xPos, yPos, targetXPos, targetYPos, uArray = sim_ra.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag,periodicflag,
                                                                                          rEgo,rEgoTarget,Egonumber,distf,adistf,J,beta,h0s,h_b,dt,
                                                                                          v0,v0t,item,hColl,rColl,[50],[50],initialxt,initialyt,True,False)
    # do the same get_bump_type logic to see how many bumps there are...
    total_time = len(headings[0])
    nbumps_eachstep= []
    for i in range(total_time):
        full_activity = uArray[:,0,i]
        full_indices = [i for i, x in enumerate(full_activity) if x > 0]
        nbumps = 0
        if len(full_indices) > 0:
            nbumps += 1
            for index in range(len(full_indices)-1):
                if full_indices[index+1]-full_indices[index] > 1:
                    nbumps += 1
            if full_indices[0] == 0 and full_indices[-1] == 99:
                nbumps -= 1
        nbumps_eachstep.append(nbumps)
    print(f"sigma: {item}, mean bumps: {np.mean(nbumps_eachstep)}, sd bumps: {np.std(nbumps_eachstep)}")
    plt.figure(2)
    plt.imshow(uArray[:,0,:],aspect='auto')
    plt.show()


# so, it does make sense, but it seems like aggregation occurs when sigma^2 > difference in neurons apart/100
# lets see if that holds when we move one of the points back? - hypothesis: this will get messed up immediately when we move towards the other one
        

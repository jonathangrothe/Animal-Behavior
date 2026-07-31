# goal for this: 
# get an idea of how it aggregates different points that are relatively close
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
L = 100 
ntargets = 2
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 50
    initialy[a] = 50
initialxt = [50,50]
initialyt = [25,75]
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
h0s = [0.25,0.26]
h_b = 0.2
sigma = 0.25
h0s_better = np.linspace(0.251,0.27,num=30)
h0s_test = []
for h in h0s_better:
    h0s_test.append([0.25,h])
beta = 100

p_more_attractive = [] 
for item in h0s_test:
    sum_t1 = 0
    sum_t0 = 0
    sum_none = 0
    for i in range(50):
        headings, xPos, yPos, targetXPos, targetYPos, uArray = sim_ra.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag,periodicflag,
                                                                                          rEgo,rEgoTarget,Egonumber,distf,adistf,J,beta,item,h_b,dt,
                                                                                          v0,v0t,sigma,hColl,rColl,[50],[50],initialxt,initialyt,False,True)
        target, time_reached = sim_met.get_destination_metrics(xPos,yPos,targetXPos,targetYPos)
        if target == 1:
            sum_t1 += 1
        if target == 0:
            sum_t0 += 1
        if target == -1:
            sum_none += 1


    
      

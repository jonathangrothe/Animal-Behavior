import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor
import simulation_metrics
import repeated_sims

# --------  PARAMETERS --------

L = 100 # width of grid
ntargets = 2
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 20
    initialy[a] = 50
initialxt = [80,80]
initialyt = [20,80]

# number of time steps
T = 5000
periodicflag = 0
rEgo = 0 # radius to switch to egocentric (for when close to another agent)
rEgoTarget = 0 # radius for switch near target 
Egonumber = 1

# collision avoidance:
hColl = -10
rColl = 0

# signal decay
distf = 1
adistf = 1

dt = 0.1
v0 = 0.05

v0t = np.zeros(ntargets)
for i in range(ntargets):
    v0t[i] = 0

N = 100 # number of neurons
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

allocentricFlag = 0 # 1 is allo, 0 is ego
h0s = [0.45,0.45,0] # attraction vector, first are attraction for targets, then agents
h_b = 0.2
sigma = 0.5
beta = 100


# -------- Running the simulation --------

# first we set out base parameters using the initializations above

base = {'N':N,
        'L':L,
        'ntargets':ntargets,
        'nagents':nagents,
        'initialx':initialx,
        'initialy':initialy,
        'initialxt':initialxt,
        'initialyt':initialyt,
        'T':T,
        'periodicFlag':periodicflag,
        'rEgo':rEgo,
        'rEgoTarget':rEgoTarget,
        'Egonumber':Egonumber,
        'hColl':hColl,
        'rColl':rColl,
        'distf':distf,
        'adistf':adistf,
        'dt':dt,
        'v0':v0,
        'v0t':v0t,
        'nu':nu,
        'J':J,
        'allocentricFlag':allocentricFlag,
        'h0':h0s,
        'h_b':h_b,
        'sigma':sigma,
        'beta':beta}

change = {'h0': [[0.45,0.45],[0.45,0.46]]}

n_samples = 3

success, x_trajs, y_trajs = repeated_sims.sample_sims(base,change,n_samples)
print(f"success: {success}")


# plotting the trajectories over a certain number of samples - NEEDS WORK
'''
plt.figure(1)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.viridis(np.linspace(0, 1, 5)),L,
                                     "Average trajectories across attraction values from 0.64 to 1 from 25 samples, even attraction, allocentric")
'''

# plot x axis as param of interest, plot y axis as success measure
# this should be its own function in simulation_metrics (That file def needs to be looked over)
'''
plt.figure(2)
plt.plot(attrac_for_plot, success)
plt.title("Success over different attraction values (even attraction, allocentric)")
plt.xlabel("Attraction of both targets")
plt.ylabel("Smallest distance to closer target (over sum of distances)")
plt.show()
'''


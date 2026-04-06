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

allocentricFlag = 1 # 1 is allo, 0 is ego
h0s = [0.45,0.45] # attraction vector, first are attraction for targets, then agents
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

change = {'h0':[[0.68,0.68],[0.71,0.71],[0.74,0.74],[0.77,0.77],[0.8,0.8]]}

n_samples = 30

success, x_trajs, y_trajs = repeated_sims.sample_sims(base,change,n_samples)

plt.figure(1)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Mean trajectories for allocentric, even attraction. h0:0.68(blue)->0.8(red)',5, 'mean')

plt.figure(2)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Min trajectories for allocentric, even attraction. h0:0.68(blue)->0.8(red)',5, 'min')

plt.figure(3)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Max trajectories for allocentric, even attraction. h0:0.68(blue)->0.8(red)',5, 'max')
plt.show(block=False)

# plot x axis as param of interest, plot y axis as success measure
x_label = "h0"
y_label = "Smallest distance to closer target (over sum of distances)"
success_title = "Success over different h0 values, allocentric, even attraction"
plt.figure(4)
simulation_metrics.plot_metric(success,[0.68,0.71,0.74,0.77,0.8],success_title,x_label,y_label)
plt.show()



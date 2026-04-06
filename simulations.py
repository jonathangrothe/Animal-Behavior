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

change = {'h_b':np.linspace(0.05,0.25,num=5)}

n_samples = 30

success, x_trajs, y_trajs = repeated_sims.sample_sims(base,change,n_samples)

plt.figure(1)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Mean trajectories for egocentric, even attraction. hb:0.05(blue)->0.25(red)',5, 'mean')

plt.figure(2)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Min trajectories for egocentric, even attraction. beta:0.05(blue)->0.25(red)',5, 'min')

plt.figure(3)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Max trajectories for egocentric, even attraction. beta:0.05(blue)->0.25(red)',5, 'max')
plt.show(block=False)

# plot x axis as param of interest, plot y axis as success measure
x_label = "h_b"
y_label = "Smallest distance to closer target (over sum of distances)"
success_title = "Success over different h_b values, egocentric, even attraction"
plt.figure(4)
simulation_metrics.plot_metric(success,change['h_b'],success_title,x_label,y_label)
plt.show()



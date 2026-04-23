import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
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
distf = 0
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
h0s = [0.25,0.25] # attraction vector, first are attraction for targets, then agents
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

# plotting settings: controls what kind of simulations we're running
plot_neurons = True
plot_trajs = [True, 'scatter']
uneven = True
sample_size = 10

base_value = 0.249
h0_range= [0,0.001,0.002,0.003]
h0_for_plot = h0_range


h0_list = []
for item in h0_range:
    h0_list.append([[base_value+item,base_value+item+0.01]])
print(h0_list)

xpositions = []
ypositions = []
neuron_activity = []
mean_decision_points = []
sum_total_activity = []
range_total_activity = []
for item in h0_list:
    change = {'h0':item}
    success_list, target_list, time_list, decision_points, sum_activity_list, range_activity_list, range_argmin_list, activity_df, x_list, y_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
    xpositions += (x_list)
    ypositions += (y_list)
    neuron_activity.append(activity_df)
    sum_total_activity += sum_activity_list
    range_total_activity += range_activity_list
    sum_s = 0
    sum_e = 0
    for item in decision_points:
        sum_s+=item[0]
        sum_e+=item[1]
    mean_decision_points.append([round(sum_s/sample_size),round(sum_e/sample_size)])

figure_num = 1

    
# SINGLE SETTING PLOTS

# general plotting settings
n_plots = len(h0_range)

# neuron heat maps
plt.figure(figsize=(10,7))
fig, ax = plt.subplots(n_plots,1,num=figure_num)
for s in range(n_plots):
    rolled= np.roll(neuron_activity[s].values, shift=50, axis=0)
    print(rolled.shape)
    ax[s].imshow(rolled,cmap='viridis',aspect='auto')
    ax[s].axvline(x=mean_decision_points[s][0], color='red', linestyle='--')
    ax[s].axvline(x=mean_decision_points[s][1], color='red', linestyle='--')
fig.suptitle("Neuron activity")
fig.supxlabel("Time")
fig.supylabel("Neuron (samples are stacked, 50, 150, etc. are neuron 0)")
figure_num += 1

# TO DO: compare different measures of the 'decision' interval: right now we're just using the min and the max of the sum of activity
# some possible candidates: 
# min/max of the range of activity (taken over a reasonable interval near)
# Decision end: (much easier I think)
# decision end as when the sum/range of activity gets within a certain amount of the stable activity level
# minimum speed (besides the start)
# Decision start: these need a bit more exploring but they could be interesting
# when neuron 0 is no longer the most active neuron ?
# number of neurons active goes down or up ?


# sum of all activity plots
plt.figure(figsize=(10,7))
fig, ax = plt.subplots(n_plots,1,num=figure_num)
for s in range(n_plots):
    mean_list = sim_met.plot_sum_activity(sum_total_activity[s*sample_size:(s+1)*sample_size],ax[s])
fig.suptitle(f"Sum of all neuron activity over time, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
fig.supxlabel("Time")
fig.supylabel("Sum of neuron activity")
figure_num+=1

# range activity plots - modify this - probably don't need to plot everything...
plt.figure(figsize=(10,7))
fig, ax = plt.subplots(n_plots,1,num=figure_num)
for s in range(n_plots):
    mean_list = sim_met.plot_sum_activity(range_total_activity[s*sample_size:(s+1)*sample_size],ax[s])
fig.suptitle(f"Range of neuron activity over time, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
fig.supxlabel("Time")
fig.supylabel("Range of neuron activity")
figure_num+=1

#trajectories

if plot_trajs[0]:
    n_plots = len(h0_range)
    plt.figure(figsize=(12,6))
    fig, ax = plt.subplots(1,n_plots,num=figure_num)
    if plot_trajs[1] == 'scatter':
        for s in range(n_plots):
            sim_met.plot_traj(xpositions[s*sample_size:(s+1)*sample_size],ypositions[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,ax[s],0,0)
    fig.suptitle(f"Trajectory plot, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figsize=(12,6))
    fig, ax = plt.subplots(1,n_plots,num=figure_num)
    if plot_trajs[1] == 'scatter':
        for s in range(n_plots):
            sim_met.plot_traj(xpositions[s*sample_size:(s+1)*sample_size],ypositions[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,ax[s],mean_decision_points[s][0],mean_decision_points[s][1])
    fig.suptitle(f"Trajectory plot during the decision, h0: {base_value}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")

plt.show()
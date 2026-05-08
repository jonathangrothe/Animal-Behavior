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

allocentricFlag = 1 # 1 is allo, 0 is ego 
h0s = [0.25,0.25] # attraction vector, first are attraction for targets, then agents
h_b = 0.2
sigma = 0.5
beta = 9.5


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
min_list = []
max_list = []
min_range_list = []
max_range_list = []

'''
for i in range(5):
    min_low = 0.19- 0.04 + 0.01*np.random.rand()
    min_high = 0.19 + 0.04 + 0.01*np.random.rand()
    max_low = 0.31 - 0.04 + 0.01*np.random.rand()
    max_high = 0.31 + 0.04 + 0.01*np.random.rand()
    min_est,range_min = repeated_sims.boundary_search(base,min_low,min_high,10,True,0.5,5) # probably also want to return the final step size to get an idea of the scale of the boundary point
    max_est, range_max = repeated_sims.boundary_search(base,max_low,max_high,10,False,0.5,5)
    print(f"boundaries, sim: {i}: {min_est,max_est}")
    min_list.append(min_est)
    max_list.append(max_est)
    min_range_list.append(range_min)
    max_range_list.append(range_max)

print(f"min list: {min_list}")
print(f"max_list: {max_list}")
min_val = np.mean(min_list)
max_val = np.mean(max_list)
min_range = np.mean(min_range_list)
max_range = np.mean(max_range_list)
'''

h0_range= [0.215,0.23,0.23,0.26,0.26,0.29,0.29,0.31,0.31] #0.215 - 0.32
h0_for_plot = [h0_range]


h0_list = [[[0.215,0.215]],[[0.23,0.230001]],[[0.23,0.230005]],[[0.26,0.260001]],[[0.26,0.260005]],[[0.29,0.290001]],[[0.29,0.290005]],[[0.31,0.310001]],[[0.31,0.310005]]]
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
    print(target_list)
    xpositions += (x_list)
    ypositions += (y_list)
    neuron_activity.append(activity_df)
    sum_total_activity += sum_activity_list
    range_total_activity += range_activity_list
    sum_s = 0
    sum_e = 0
    for item in decision_points:
        sum_s+=item[0]
        sum_e+=item[1]+5
    mean_decision_points.append([round(sum_s/sample_size),round(sum_e/sample_size)])

figure_num = 1

    
# SINGLE SETTING PLOTS

# general plotting settings
n_plots = len(h0_range)
ncols = 3
nrows = 3
# neuron heat maps - maybe just do one example for each?
plt.figure(figsize=(14.4375,7))
fig, ax = plt.subplots(nrows*2,3,num=figure_num)
axes_flat = ax.flatten()
for s in range(n_plots):
    # figure out way to plot neuron heatmap directly below corresponding trajectory
    sim_met.plot_traj(xpositions[s*sample_size:(s+1)*sample_size],ypositions[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,axes_flat[s],0,0)
    axes_flat[s].set_title(f"h0: {h0_list[s][0]}")
    rolled= np.roll(neuron_activity[s].dropna().iloc[0:100,:].values, shift=50, axis=0)
    col_min = np.min(neuron_activity[s].iloc[:,40:])
    col_max = np.max(neuron_activity[s].iloc[:,40:])
    axes_flat[s].imshow(rolled,cmap='viridis',aspect='auto',vmin=col_min,vmax=col_max)
    axes_flat[s].set_title(f"h0: {h0_list[s][0]}")
#fig.suptitle(f"Example neuron activity from one simulation, {"allocentric" if allocentricFlag==1 else "egocentric"}, beta: {beta}, hb: {h_b}, sigma: {sigma}") 
#fig.supxlabel("Time")
#fig.supylabel("Neuron (50 is forward)")
figure_num += 1



#trajectories
''''
if plot_trajs[0]:
    n_plots = len(h0_range)
    plt.figure(figsize=(14.475,7))
    fig, ax = plt.subplots(2,5,num=figure_num)
    axes_flat = ax.flatten()
    if plot_trajs[1] == 'scatter':
        for s in range(n_plots):
            sim_met.plot_traj(xpositions[s*sample_size:(s+1)*sample_size],ypositions[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,axes_flat[s],0,0)
            axes_flat[s].set_title(f"h0: {h0_list[s][0]}")
    #fig.suptitle(f"Trajectory plot, {"allocentric" if allocentricFlag==1 else "egocentric"}, beta: {beta}, hb: {h_b}, sigma: {sigma}")
    figure_num += 1
'''

plt.show()
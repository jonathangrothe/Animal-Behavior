import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
import repeated_sims
from scipy.optimize import curve_fit

# --------  PARAMETERS --------

L = 100 # width of grid
ntargets = 3
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 50
    initialy[a] = 50
initialxt = [50-(15*np.sqrt(3)),50,50+(15*np.sqrt(3))]
initialyt = [35,80,35]

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
h0s = [0.25,0.25,0.25] # attraction vector, first are attraction for targets, then agents
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
# not plotting trajectories or neurons in this file
plot_trajs = True
include_neurons = True
sample_size = 1

base_sigma = np.linspace(0.05,1,num=100)
start = 0.15
finish = 0.35
n_h0 = 100
h0_range= np.linspace(start,finish,num=n_h0)
h0_list = []
for item in h0_range:
    h0_list.append([item,item,item])

subfigs, axs = plt.subplots(nrows=1,ncols=3, figsize = (16,7))


target_grid = []
phase_grid = []
time_grid = []
for sigma_index in range(len(base_sigma)):
    sigma_list = [base_sigma[sigma_index]]*n_h0
    change= {'h0':h0_list,
             'sigma':sigma_list}
    target_list, time_list, decision_points, decision_pos, activity_df, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=include_neurons)
    p_target, se_target = sim_met.get_success_rate(target_list, sample_size, ntargets)
    target_reached = [1 if x >= 0 else x for x in target_list]
    phases = []
    for i in range(len(target_list)):
        curr_activity = activity_df.iloc[i*100:(i+1)*100]
        curr_xpos = x_list[i:(i+1)][0]
        curr_ypos = y_list[i:(i+1)][0]
        probabilities = sim_met.get_bump_type(initialxt,initialyt,curr_xpos,curr_ypos,curr_activity,5)
        phase = np.argmax(probabilities)
        if target_reached == -1 and probabilities[3]>=0.3:
            phase = 3
        phases.append(phase) # can add to this later with bifurcation stuff when c4 is bigger and c2 is smaller in the c2 case
    grid_phases = []
    grid_targets = []
    grid_times = []
    for s in range(n_h0):
        sim_phases = phases[s*sample_size:(s+1)*sample_size]
        sim_targets = target_reached[s*sample_size:(s+1)*sample_size]
        sim_times = time_list[s*sample_size:(s+1)*sample_size]
        n0 = sim_phases.count(0)
        n1 = sim_phases.count(1)
        n2 = sim_phases.count(2)
        n3 = sim_phases.count(3)
        nOther = sim_phases.count(4)
        nreach = sim_targets.count(1)
        nfail = sim_targets.count(-1)
        reach = 0
        if nreach > nfail:
            reach = 1
        if nreach < nfail:
            reach = -1
        grid_phases.append(np.argmax([n0,n1,n2,n3,nOther]))
        grid_targets.append(reach)
        grid_times.append(np.mean(sim_times))
    phase_grid.append(grid_phases)
    target_grid.append(grid_targets)
    time_grid.append(grid_times)

target_df = pd.DataFrame(target_grid,columns=h0_range,index=base_sigma)
print(target_df)
phase_df = pd.DataFrame(phase_grid,columns=h0_range,index=base_sigma)
print(phase_df)
time_df = pd.DataFrame(time_grid,columns=h0_range,index=base_sigma)
print(time_df)
axs[0].imshow(target_df, cmap='bwr_r',origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
axs[1].imshow(phase_df,cmap='viridis',origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
axs[2].imshow(time_df,cmap='inferno_r',origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
axs[0].set_title("Reaching a target heatmap")
axs[1].set_title("Phase heatmap (purple: 0 bumps -> yellow: 3 bumps)")
axs[2].set_title("Time to target heatmap (lighter is fastere)")
subfigs.supxlabel('h0')
subfigs.supylabel('sigma')
'''
ax.set_xticks(np.arange(len(target_df.columns)))
ax.set_xticklabels(target_df.columns)
ax.set_yticks(np.arange(len(target_df.index)))
ax.set_yticklabels(target_df.index)

area_list = [0.05,0.2,0.35,0.5]
cmap = plt.get_cmap('cividis')
colors_area = cmap(np.linspace(0, 1, len(area_list)))
for area_index in range(len(area_list)):
    sigma_list = repeated_sims.area_helper(area_list[area_index],'h0',h0_range)
    sim_met.plot_phase_over_area(h0_range,sigma_list,target_reached,colors_area[area_index],axs[0])
'''
plt.show()

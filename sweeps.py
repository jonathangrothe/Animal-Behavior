import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

base_sigma = np.linspace(0.05,1,num=10)
start = 0.15
finish = 0.35
n_h0 = 10
h0_range= np.linspace(start,finish,num=n_h0)
h0_list = []
for item in h0_range:
    h0_list.append([item,item,item])

subfigs, axs = plt.subplots(nrows=1,ncols=3, figsize = (20,5))


target_grid = []
phase_grid = []
time_grid = []
shift_grid = []
switchs_grid = []
for sigma_index in range(len(base_sigma)):
    sigma_list = [base_sigma[sigma_index]]*n_h0
    change= {'h0':h0_list,
             'sigma':sigma_list}
    target_list, time_list, decision_points, decision_pos, activity_df, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=include_neurons)
    p_target, se_target = sim_met.get_success_rate(target_list, sample_size, ntargets)
    target_reached = [1 if x >= 0 else x for x in target_list]
    phases = []
    shift = []
    switch = []
    for i in range(len(target_list)):
        curr_activity = activity_df.iloc[i*100:(i+1)*100]
        curr_xpos = x_list[i:(i+1)][0]
        curr_ypos = y_list[i:(i+1)][0]
        probabilities = sim_met.get_bump_type(initialxt,initialyt,curr_xpos,curr_ypos,curr_activity,5)
        phase = np.argmax(probabilities)
        if target_reached == -1 and probabilities[3]>=0.3:
            phase = 3
        phases.append(phase) # can add to this later with bifurcation stuff when c4 is bigger and c2 is smaller in the c2 case
        print(f"phase: {phase}")
        print(f"reaches target: {target_list[i]}")
    grid_phases = []
    grid_targets = []
    grid_times = []
    grid_shift = []
    grid_switch = []
    for s in range(n_h0):
        sim_phases = phases[s*sample_size:(s+1)*sample_size]
        sim_targets = target_reached[s*sample_size:(s+1)*sample_size]
        sim_times = time_list[s*sample_size:(s+1)*sample_size]
        sim_shifts = shift[s*sample_size:(s+1)*sample_size]
        sim_switchs = switch[s*sample_size:(s+1)*sample_size]
        n0 = sim_phases.count(0)
        n1 = sim_phases.count(1)
        n2 = sim_phases.count(2)
        n3 = sim_phases.count(3)
        nOther = sim_phases.count(4)
        nreach = sim_targets.count(1)
        reach = 0
        if nreach/sample_size >= 0.75:
            reach = 1
        elif nreach/sample_size <= 0.25:
            reach = -1
        else:
            reach = 0
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

bwr_r = plt.colormaps['bwr_r']
discrete_bwrr = bwr_r.resampled(3)(np.linspace(0,1,3))
viridis = plt.colormaps['viridis']
discrete_viridis = viridis.resampled(5)(np.linspace(0,1,5))
cmap1 = mcolors.ListedColormap(discrete_bwrr)
cmap2 = mcolors.ListedColormap(discrete_viridis)
cmap3 = 'inferno_r'
boundaries_tar = np.arange(-1,3) - 0.5
norm_tar = mcolors.BoundaryNorm(boundaries_tar,cmap1.N)
boundaries_phase = np.arange(5) - 0.5
norm_phase = mcolors.BoundaryNorm(boundaries_phase,cmap2.N)



categories_tar = ['Fails to reach', 'Both', 'Reaches']
categories_phase = ['0 bumps', '1 bump', '2 bumps', '3 bumps']

im1 = axs[0].imshow(target_df, cmap=cmap1,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im2 = axs[1].imshow(phase_df,cmap=cmap2,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im3 = axs[2].imshow(time_df,cmap=cmap3,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
cbar1 = plt.colorbar(im1, ticks=np.arange(-1,2))
cbar1.ax.set_yticklabels(categories_tar)
cbar2 = plt.colorbar(im2, ticks=np.arange(4))
cbar2.ax.set_yticklabels(categories_phase)
cbar3 = plt.colorbar(im3)
cbar3.set_label('Time to target')

subfigs.supxlabel('h0')
subfigs.supylabel('sigma')
subfigs.suptitle(f"Heatmaps for 2π/3 between targets, average of {sample_size} samples")
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
plt.tight_layout()
plt.show()

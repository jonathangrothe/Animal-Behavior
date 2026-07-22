import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from . import simulation_metrics as sim_met
from . import repeated_sims


L = 100
ntargets = 3
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 50
    initialy[a] = 50
initialxt = [35,50,65]
initialyt = [50+15*np.sqrt(3),80,50+15*np.sqrt(3)]

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
h0s = [0.21903,0.21903,0.21904]
h_b = 0.2
sigma = 0.25 
beta = 100 

# -------- Running the simulation --------

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

plot_neurons = True
plot_trajs = True
sample_size = 1
h0_list = [[0.212,0.212,0.212],[0.21,0.21,0.21],[0.212,0.212,0.212],[0.213,0.213,0.213]]
sigma_list = [0.16,0.23,0.24,0.26]
change = {'sigma':sigma_list, 'h0':h0_list}
target_list, time_list, activity_list, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)

phases = []
angles = []
bif_indices = []
for i in range(len(target_list)):
    curr_activity = activity_list[i]
    curr_headings = headings_list[i][0]
    curr_xpos = x_list[i:(i+1)][0]
    curr_ypos = y_list[i:(i+1)][0]
    target = target_list[i]
    targx = 'n'
    targy = 'n'
    indices = [0]
    bifurcation_angles = [np.pi/2]
    if target != -1:
        targx = initialxt[target]
        targy = initialyt[target]
        bif_start = time.perf_counter()   
        indices, bifurcation_angles = sim_met.get_bifurcation_angle(curr_xpos,curr_ypos,curr_headings,0.25,5000)
        bif_end = time.perf_counter()
        bif_time = bif_end - bif_start
        print(f"bif time: {bif_time:.6f} seconds")
    start_time = time.perf_counter()
    phase = sim_met.get_bump_type(curr_activity)
    end_time = time.perf_counter()
    bump_time = end_time-start_time
    print(f"time to get bump type: {bump_time:.6f} seconds")
    angles.append(bifurcation_angles)
    bif_indices.append(indices)
    phases.append(phase)

n_plots = len(h0_list)
ncols = 4
nrows = 1
fig = plt.figure(layout='constrained',figsize=(ncols*3.5,nrows*6))
subfigs = fig.subfigures(2,1, wspace=0.1)
axs0 = subfigs[0].subplots(nrows,ncols)
axs0 = axs0.flatten()
axs1 = subfigs[1].subplots(nrows,ncols)
axs1 = axs1.flatten()
#axs2 = subfigs[2].subplots(nrows,ncols)
#axs2 = axs2.flatten()
grey_to_blue = ["#D3D3D3", "#A9A9A9", "#708090", "#4682B4", "#000080"]
cmap = mcolors.LinearSegmentedColormap.from_list("GreyBlue", grey_to_blue)

for s in range(n_plots):
    sim_met.plot_traj(x_list[s*sample_size:(s+1)*sample_size],y_list[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,axs0[s],True,bif_indices[s*sample_size:(s+1)*sample_size],0,0)
    axs0[s].set_title(f"sigma: {round(sigma_list[s],4)}, h0: {round(h0_list[s][0],4)}, beta: {beta}")
    axs0[s].set_aspect('equal', adjustable='box')
    sim_activity = activity_list[s*sample_size]
    col_min = np.min(sim_activity[:,40:])
    col_max = np.max(sim_activity[:,40:])
    axs1[s].imshow(sim_activity,cmap=cmap,aspect='auto',vmin=col_min,vmax=col_max)
    sim_phases = phases[s*sample_size:(s+1)*sample_size]
    sim_angles = angles[s*sample_size:(s+1)*sample_size]
    sim_indices = bif_indices[s*sample_size:(s+1)*sample_size]
    n0 = sim_phases.count(0)
    n1 = sim_phases.count(1)
    n2 = sim_phases.count(2)
    n3 = sim_phases.count(3)
    n4 = sim_phases.count(4)
    print(f"Plot: {s}")
    print(f"outcome: 0: {n0}, 1: {n1}, 2: {n2}, 3: {n3}, other: {n4}")
    print(f"overall: {np.argmax([n0,n1,n2,n3,n4])}")
    print(f"target list: {target_list[s*sample_size:(s+1)*sample_size]}")
    print(f"angles: {sim_angles}")
    print(f"indices: {sim_indices}")


plt.show()
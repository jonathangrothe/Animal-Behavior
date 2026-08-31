import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from . import simulation_metrics as sim_met
from . import repeated_sims

start_time = time.perf_counter()

L = 100
ntargets = 2
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 50
    initialy[a] = 50
initialxt = [25,75]
initialyt = [75,75]

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
h0s = [0.23,0.23]
h_b = 0.2
sigma = 0.2
beta = 100

# -------- Running the simulation --------
u0 = np.zeros((N,1))
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
        'beta':beta,
        'u0':u0}

include_pos = True
include_neurons = True
sample_size = 1
v0_start = 0.03
v0_finish = 0.929
n_v0 = 75
beta_start = 25
beta_finish = 200
n_beta = 75
base_v0 = np.linspace(v0_start,v0_finish,num=n_v0)
base_beta = np.linspace(beta_start,beta_finish,num=n_beta)
v0_list = []
beta_list = []
for b in range(n_beta):
    for v in range(n_v0):
        v0_list.append(base_v0[v])
        beta_list.append(base_beta[b])

init_list = []
for i in range(6):
    u0 = np.zeros((N,1))
    start_ind = (i*5)+50
    end_ind = (i*5)+76
    if end_ind < 100:
        u0[start_ind:end_ind] = 0.2
    else:
        u0[start_ind:] = 0.2
    init_list.append(u0)

change= {'beta':beta_list,
         'v0':v0_list}
sample_time = time.perf_counter()
fig_init = plt.figure(layout='constrained',figsize = (20,5.5),num=1)
subfigs_init = fig_init.subfigures(2,3)
for u in range(6):
    base['u0'] = init_list[u]
    target_list, time_list, activity_list, x_list, y_list, headings_list, init_heading  = repeated_sims.sample_sims(base,change,sample_size,include_pos,include_neurons)
    end_sample_time = time.perf_counter()
    sampling_time = end_sample_time - sample_time
    print(f"Sampling time: {sampling_time:.6f} seconds")
    analysis_time = time.perf_counter()
    t_reached = []
    row_ind = 1 if u>2 else 0
    for item in target_list:
        if item == -1:
            t_reached.append(0)
        elif item != row_ind:
            t_reached.append(1)
        else:
            t_reached.append(-1)
        
    grid = np.zeros((n_v0,n_beta))
    for p in range(n_beta):
        row = t_reached[p*n_beta:(p+1)*n_beta]
        grid[p,:] = row
    print(grid)
    bwr_r = plt.colormaps['bwr_r']
    axs0 = subfigs_init[row_ind][u%3].subplots(1,2)
    axs0[0].imshow(grid,cmap=bwr_r,aspect='auto',extent=[base_v0[0], base_v0[-1], base_beta[-1], base_beta[0]])
    axs0[0].set_xlabel("v0")
    axs0[0].set_ylabel("beta")
    start_angle = ((u*5+50)/100)*np.pi*2
    end_angle = (((u*5+76)+25)/100)*np.pi*2 
    theta = np.linspace(start_angle, end_angle, 200)
    to_finish = np.linspace(end_angle,2*np.pi,600)
    to_start = np.linspace(0,start_angle,200)
    x_plot = np.cos(theta)
    y_plot = np.sin(theta)
    x_finish = np.cos(to_finish)
    y_finish = np.sin(to_finish)
    x_start = np.cos(to_start)
    y_start = np.sin(to_start)
    axs0[1].plot(x_plot,y_plot,c='red')
    axs0[1].plot(x_finish,y_finish, c='blue')
    axs0[1].plot(x_start,y_start, c='blue')
end_analysis_time = time.perf_counter()
analyzing_time = end_analysis_time - analysis_time
print(f"Analysis time: {analyzing_time:.6f} seconds")



end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
import repeated_sims
import matplotlib.colors as mcolors

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
h0s = [0.27,0.27005] 
h_b = 0.2
sigma = 0.5 # narrower sigma means more bifurcations in 3 (+?) target case
beta = 20


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
plot_trajs = True
sample_size = 1


h0_list = [[0.23,0.23005],[0.23,0.23001],[0.25,0.25002],[0.25,0.250005]]
allo_list = [0,1]

for i in range(2):
    base['allocentricFlag'] = allo_list[i] # add title to reflect allocentricflag

    change = {'h0':h0_list}
    target_list, time_list, decision_points, decision_pos, activity_df, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
        
    # look at activity, get the times of bifurcation and max activated values around them ?

    one_sample = activity_df.dropna(axis=1).iloc[0:100,:]
    max_act = 0.9
    tanh_val = max_act*2-1
    u_cutoff = (0.5*np.log((1+tanh_val)/(1-tanh_val)))/beta
    print(f"u cutoff: {u_cutoff}")
    for x in range(len(one_sample.columns)):
        max_u = np.max(one_sample.iloc[:,x]) 
        if x % 5 == 0:
            print(f"tstep: {x}, max activ: {max_u}")
        if max_u < u_cutoff:
            print(f"low activity alert! tstep: {x}, max_activ: {max_u}")

    n_plots = len(h0_list)
    ncols = len(h0_list)
    nrows = 1

    fig = plt.figure(layout='constrained',figsize=(10,5),num=i+1)
    subfigs = fig.subfigures(2,1, wspace=0.1)
    axs0 = subfigs[0].subplots(nrows,ncols)
    axs1 = subfigs[1].subplots(nrows,ncols)

    grey_to_blue = ["#D3D3D3", "#A9A9A9", "#708090", "#4682B4", "#000080"]
    cmap = mcolors.LinearSegmentedColormap.from_list("GreyBlue", grey_to_blue)

    for s in range(n_plots):
        dec_points = []
        dec_positions = []
        sim_met.plot_traj(x_list[s*sample_size:(s+1)*sample_size],y_list[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,[0],axs0[s],False,0,0)
        axs0[s].set_title(f"{"allocentric" if allo_list[i]==1 else "egocentric"},h0: {h0_list[s]}")

        rolled= np.roll(activity_df.dropna(axis=1).iloc[s*100*sample_size:s*100*sample_size+100,:].values, shift=50, axis=0)
        col_min = np.min(activity_df.iloc[:,40:])
        col_max = np.max(activity_df.iloc[:,40:])
        axs1[s].imshow(rolled,cmap=cmap,aspect='auto',vmin=col_min,vmax=col_max)


plt.show()
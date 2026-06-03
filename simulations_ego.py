import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
import repeated_sims
import matplotlib.colors as mcolors

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
initialyt = [65,80,65]

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
h0s = [0.205,0.205,0.205] # attraction vector, first are attraction for targets, then agents
h_b = 0.2
sigma = 0.2 #0.25 works for three target (sometimes)
beta = 100 #100 works for double


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
uneven = True
sample_size = 30

# turn this into a sweep of sweeps, get the range each time
# get a list of betas/sigmas we want to sweep over 
'''
sigma_sweep_list = np.linspace(0.175,0.55,num=10)
sigma_h0range_list = []
sigma_extrema = []

min_low = 0.2- 0.04 + 0.01*np.random.rand()
min_high = 0.2 + 0.04 + 0.01*np.random.rand()
max_low = 0.25 - 0.04 + 0.01*np.random.rand()
max_high = 0.25 + 0.04 + 0.01*np.random.rand()


for item in sigma_sweep_list:
    base['sigma'] = item
    # first run a mini sweep to see if there might be something worthwhile? IDK
    min_est,range_min = repeated_sims.boundary_search(base,min_low,min_high,10,True,'h0',0.2)
    max_est, range_max = repeated_sims.boundary_search(base,max_low,max_high,10,False,'h0',0.2)
    print(f"boundaries, sigma: {item}: min: {min_est} (range: {range_min}), max: {max_est} (range: {range_max})")
    sigma_h0range_list.append(max_est-min_est)
    sigma_extrema.append((min_est,max_est))
    min_low = min_est - 0.04 + 0.01*np.random.rand()
    min_high = min_est + 0.04 + 0.01*np.random.rand()
    max_low = max_est - 0.04 + 0.01*np.random.rand()
    max_high = max_est + 0.04 + 0.01*np.random.rand()

print(f"range list: {sigma_h0range_list}")
ind_best_sigma = np.argmax(sigma_h0range_list)
best_sigma = sigma_sweep_list[ind_best_sigma]

base['sigma'] = best_sigma
min_val = sigma_extrema[ind_best_sigma][0]+0.01
max_val = sigma_extrema[ind_best_sigma][1]-0.01
int_val = round((min_val+max_val)/2,5)
'''
min_val = 0.2
max_val = 0.22
int_val = (min_val+max_val)/2
#h0_list = [[0.19,0.19,0.19],[0.195,0.195,0.195],[0.2,0.2,0.2],[0.205,0.205,0.205],[0.21,0.21,0.21],[0.215,0.215,0.215]]
#print(h0_list)
sigma_list = [0.18,0.23,0.28,0.33,0.38,0.43]
#beta_list = [[11],[15],[20],[40],[90],[250]]
#allo_list = [[0],[1]]


change = {'sigma':sigma_list}
target_list, time_list, decision_points, decision_pos, activity_df, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
print(target_list)

print(np.shape(activity_df))

n_plots = len(sigma_list)
ncols = len(sigma_list)
nrows = 1
# neuron heat maps - maybe just do one example for each? 
fig = plt.figure(layout='constrained',figsize=(10,10))
#fig, ax = plt.subplots(nrows,ncols,num=figure_num)
subfigs = fig.subfigures(2,1, wspace=0.1)
axs0 = subfigs[0].subplots(nrows,ncols)
axs1 = subfigs[1].subplots(nrows,ncols)
#axs2 = subfigs[2].subplots(nrows,ncols)

grey_to_blue = ["#D3D3D3", "#A9A9A9", "#708090", "#4682B4", "#000080"]
cmap = mcolors.LinearSegmentedColormap.from_list("GreyBlue", grey_to_blue)

for s in range(n_plots):
    sim_met.plot_traj(x_list[s*sample_size:(s+1)*sample_size],y_list[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,[0],axs0[s],False,0,0)
    axs0[s].set_title(f"{"allocentric" if allocentricFlag==1 else "egocentric"}, sigma: {sigma_list[s]}, beta: {beta}")

    rolled= np.roll(activity_df.dropna(axis=1).iloc[s*100*sample_size:s*100*sample_size+100,:].values, shift=50, axis=0)
    col_min = np.min(activity_df.iloc[:,40:])
    col_max = np.max(activity_df.iloc[:,40:])
    axs1[s].imshow(rolled,cmap=cmap,aspect='auto',vmin=col_min,vmax=col_max)



plt.show()
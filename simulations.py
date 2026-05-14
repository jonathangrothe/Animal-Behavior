import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
import repeated_sims

# --------  PARAMETERS --------

L = 100 # width of grid
ntargets = 3
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 20
    initialy[a] = 50
initialxt = [65,90,65]
initialyt = [20,50,80]

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
sigma = 0.25 #0.25 works for three target (sometimes)
beta = 200 #100 works for double


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
    max_low = 0.32 - 0.04 + 0.01*np.random.rand()
    max_high = 0.32 + 0.04 + 0.01*np.random.rand()
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
int_val = round((min_val+max_val)/2,5)
'''

# p(reaching each target|current path)
# p(t=0), p(t=1), p(t=2) 
# p(t=0|go to top), p(t=0|go to bottom), ....
# p(t=0|top, back) p(t=0|bottom, back), ....
# to get where it bifurcates pre decison we need a range of where it slows (we do this in traj code) and then to get its position at that time

h0_list = [[[0.22,0.22,0.22]],[[0.2205,0.2205,0.2205]],[[0.221,0.221,0.221]]]
print(h0_list)

xpositions = []
ypositions = []
neuron_activity = []
mean_decision_points = []
sum_total_activity = []
range_total_activity = []
indices_list_forheatmap = []
for item in h0_list:
    change = {'h0':item}
    success_list, target_list, time_list, decision_points, sum_activity_list, range_activity_list, range_argmin_list, activity_df, x_list, y_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
    print(target_list)
    # always get the top target
    for t in range(len(target_list)):
        if target_list[t] == 0:
            indices_list_forheatmap.append(t)
            break

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
#print(indices_list_forheatmap)
figure_num = 1

    
# SINGLE SETTING PLOTS
# general plotting settings
n_plots = len(h0_list)
ncols = 3
nrows = 1
# neuron heat maps - maybe just do one example for each? 
plt.figure(layout='constrained',figsize=(10,5))
fig, ax = plt.subplots(nrows,ncols,num=figure_num)
axes_flat = ax.flatten()
for s in range(n_plots): #UPDATE INDICES WITH INDICES_FORHEATMAP List
    dec_points_list = sim_met.plot_traj(xpositions[s*sample_size:(s+1)*sample_size],ypositions[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,axes_flat[s],0,0)
    print(pd.DataFrame(dec_points_list))
    # work with the coordinates from here, cluster them to find approximate true decision point, drop outliers
    # then we can calculate probabilities of reaching each point given the path traveled
    # which will then allow us to sweep the three target case
    '''
    for item in minima_indices:
        print(f"index: {item}")
        print(f"xpos at index: {xpositions[s*sample_size:(s+1)*sample_size][0][item]}")
        print(f"ypos at index: {ypositions[s*sample_size:(s+1)*sample_size][0][item]}")
    '''
    axes_flat[s].set_title(f"h0: {h0_list[s][0]}")
figure_num += 1
#plt.savefig('trajectories_beta20.png')

for a in range(n_plots):
    plt.figure(figsize=(5,2.5))
    plt.figure(figure_num)
    rolled= np.roll(neuron_activity[a].dropna(axis=1).iloc[0:100,:].values, shift=50, axis=0)
    col_min = np.min(neuron_activity[a].iloc[:,40:])
    col_max = np.max(neuron_activity[a].iloc[:,40:])
    #print(f"min: {col_min}")
    #print(f"max: {col_max}")
    plt.imshow(rolled,cmap='viridis',aspect='auto',vmin=col_min,vmax=col_max)
    plt.title(f"h0: {h0_list[a][0]}")
    #plt.savefig(f'heatmap_{h0_list[a][0]}_beta20.png')
    figure_num+=1


plt.show()
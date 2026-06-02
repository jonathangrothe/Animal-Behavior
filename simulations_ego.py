import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
import repeated_sims
import ego_data_analysis as egodata
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
h0s = [0.295,0.295001] # attraction vector, first are attraction for targets, then agents
h_b = 0.2
sigma = 0.5 #0.25 works for three target (sometimes)
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
sample_size = 1
min_list = []
max_list = []
min_range_list = []
max_range_list = []

'''
for i in range(5):
    min_low = 0.22- 0.04 + 0.01*np.random.rand()
    min_high = 0.22 + 0.04 + 0.01*np.random.rand()
    max_low = 0.28 - 0.04 + 0.01*np.random.rand()
    max_high = 0.28 + 0.04 + 0.01*np.random.rand()
    min_est,range_min = repeated_sims.boundary_search(base,min_low,min_high,10,True,'h0',0.2) # probably also want to return the final step size to get an idea of the scale of the boundary point
    max_est, range_max = repeated_sims.boundary_search(base,max_low,max_high,10,False,'h0',0.2)
    print(f"boundaries, sim: {i}: {min_est,max_est}")
    min_list.append(min_est)
    max_list.append(max_est)
    min_range_list.append(range_min)
    max_range_list.append(range_max)
'''
print(f"min list: {min_list}")
print(f"max_list: {max_list}")
min_val = np.mean(min_list)
max_val = np.mean(max_list)
min_range = np.mean(min_range_list)
max_range = np.mean(max_range_list)
int_val = round((min_val+max_val)/2,5)

#h0_list = [[[min_val,min_val]],[[int_val,int_val]],[[max_val,max_val]]]
#print(h0_list)
#sigma_list = [[0.25],[0.3],[0.35],[0.4],[0.45],[0.5]]
#beta_list = [[11],[15],[20],[40],[90],[250]]
allo_list = [[0],[1]]
xpositions = []
ypositions = []
headings = []
neuron_activity = []
mean_decision_points = []
decision_points_total = []
decision_pos_total = []
indices_list_forheatmap = []
for item in allo_list:
    change = {'allocentricFlag':item}
    target_list, time_list, decision_points, decision_pos, activity_df, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
    # always get the top target
    '''
    for t in range(len(target_list)):
        if target_list[t] == 0:
            indices_list_forheatmap.append(t)
            break
    '''
    xpositions += (x_list)
    ypositions += (y_list)
    headings += (headings_list)
    neuron_activity.append(activity_df)
    #decision_points_total += decision_points
    #decision_pos_total += decision_pos

print(np.shape(neuron_activity[0]))

positions_df = pd.DataFrame(ypositions[1],xpositions[1])
headings_df = pd.DataFrame(headings[1])
y_diff = np.zeros(len(ypositions[1]))
y_diff[1:] = np.diff(ypositions[1])
x_diff = np.zeros(len(xpositions[1]))
x_diff[1:] = np.diff(xpositions[1])
positions_df['x diff'] = x_diff
positions_df['y diff'] = y_diff
figure_num = 1

#positions_df.to_csv("positions_search_ego_df.csv")
#neuron_activity[1].to_csv("activity_ego_df.csv")
#headings_df.to_csv("headings_ego_df.csv",index=False)

# SINGLE SETTING PLOTS
# general plotting settings
n_plots = len(allo_list)
ncols = 2
nrows = 1
# neuron heat maps - maybe just do one example for each? 
fig = plt.figure(layout='constrained',figsize=(10,10))
#fig, ax = plt.subplots(nrows,ncols,num=figure_num)
subfigs = fig.subfigures(2,1, wspace=0.1)
axs0 = subfigs[0].subplots(nrows,ncols)
ax0_labels = ['A','C','E','G','I','K']
axs1 = subfigs[1].subplots(nrows,ncols)
ax1_labels = ['B','D','F','H','J','L']
#axs2 = subfigs[2].subplots(nrows,ncols)
#ax2_labels = ['C','F','I','L']
grey_to_blue = ["#D3D3D3", "#A9A9A9", "#708090", "#4682B4", "#000080"]
cmap = mcolors.LinearSegmentedColormap.from_list("GreyBlue", grey_to_blue)

for s in range(n_plots):
    sim_met.plot_traj(xpositions[s*sample_size:(s+1)*sample_size],ypositions[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,decision_points_total[s*sample_size:(s+1)*sample_size],axs0[s],0,0)
    axs0[s].set_title(f"{"allocentric" if allo_list[s][0]==1 else "egocentric"}, h0: {h0s}, beta: {beta}")
    #axs0[s].annotate(xy=(0.1, 0.9), xycoords="axes fraction")

    rolled= np.roll(neuron_activity[s].dropna(axis=1).iloc[0:100,:].values, shift=50, axis=0)
    col_min = np.min(neuron_activity[s].iloc[:,40:])
    col_max = np.max(neuron_activity[s].iloc[:,40:])
    axs1[s].imshow(rolled,cmap=cmap,aspect='auto',vmin=col_min,vmax=col_max)
    #axs1[s].annotate(xy=(0.1, 0.9), xycoords="axes fraction")
    #axs1[s].set_title("egocentric" if allo_list[s][0]==0 else "allocentric")

    '''
    # at time something plot bumps
    time_100 = neuron_activity[s].iloc[0:100,100]
    axs1[s].plot(time_100)
    axs1[s].set_title("activity at time 100") 

    time_600 = neuron_activity[s].iloc[0:100,600]
    axs2[s].plot(time_600)
    axs2[s].set_title("activity at time 600") 
    '''
    #egodata.plot_area(neuron_activity[s],axs2[s],(-1.2,11.5))
    #axs2[s].annotate(ax2_labels[s], xy=(0.1, 0.9), xycoords="axes fraction")



plt.show()
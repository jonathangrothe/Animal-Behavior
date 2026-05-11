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
beta = 25


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
plot_trajs = [False, 'scatter'] 
plot_neurons = False
uneven = True
sample_size = 15
start = 0
finish = 0.00002
h0_range= np.linspace(start,finish,num=10)
h0_for_plot = h0_range
h0_total_list = []
beta_list = [12] # 9.5: 0.215-0.32, 10: 0.208-0.325, 12: 0.189-0.339, 20: 0.177-0.338, 52: 0.182-0.325, 180: 0.189-0.326 ? 
base_h0_list = np.linspace(0.28,0.31,num=16) # 0.22 to 0.31 - 0.1925-0.335
n_plots = len(beta_list)
n_lines = len(base_h0_list)
for n in range(n_lines):
    for p in range(n_plots):
        h0_list = []
        for item in h0_range:
            h0_list.append([base_h0_list[n],base_h0_list[n]+item])
        h0_total_list.append(h0_list)
# defining our metrics of interest: 
time_total_list = []
correct_total_list = []
correct_se_list = []
range_total_list = []
range_total_se_list = []
jnd_list1 = []
jnd_list2 = []


for index in range(len(h0_total_list)):
    change= {'h0':h0_total_list[index]}
    base['beta'] = beta_list[index%n_plots ] 
    print(f'beta: {base['beta']}')
    print(f'base h0: {base_h0_list[int(index*(1/n_plots))]}') 
    success_list, target_list, time_list, decision_points, sum_activity_list, range_activity_list, range_argmin_list, activity_df, x_list, y_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
    p_success, se_success, p_correct, se_correct = sim_met.get_success_rate(target_list, sample_size, uneven, 1)
    range_mean, range_se = sim_met.get_metric_mean_se(range_argmin_list,sample_size)
    time_total_list.append(time_list)
    correct_total_list.append(p_correct)
    correct_se_list.append(se_correct)
    range_total_list.append(range_mean)
    range_total_se_list.append(range_se)
    # here take the jnd using a few measures
    ind_jnd1 = -1
    ind_jnd2 = -1
    jnd2_counter = 0
    for index in range(len(p_correct)):
        if ind_jnd1 == -1 and p_correct[index] == 1:
            ind_jnd1 = index
        if ind_jnd2 == -1 and p_correct[index] >= 0.9:
            jnd2_counter += 1 
            if jnd2_counter >= 3:
                ind_jnd2 = index
    jnd_list1.append(ind_jnd1)
    jnd_list2.append(ind_jnd2)
    
        
print(jnd_list1)
print(jnd_list2)

correct_df = pd.DataFrame(correct_total_list)
correct_df.index = base_h0_list
correct_df.columns = h0_range


# compile the data we want from each run into a dataframe
# we want: p_correct and time, and we would prefer if they are labeled with simulation settings


figure_num = 1


# overall plotting settings
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, n_lines))
labels = []
for item in base_h0_list:
    labels.append(f'base h0: {item}') 
    
for p in range(n_plots):
    x_label = "difference in atraction"
    time_y_label = "Average time to target"
    time_agg_title = f"Average time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, beta: {beta_list[0]}"
    figure_num, mean_time = sim_met.plot_metric(time_total_list[p::n_plots],h0_for_plot,colors,labels,figure_num,(8,8),
                        time_agg_title,x_label,time_y_label)
    time_df = pd.DataFrame(mean_time)
    time_df.index = base_h0_list
    time_df.columns = h0_range

    p_success_y = "Probability of reaching top target"
    p_success_title = f"Probability of reaching top target, {"allocentric" if allocentricFlag==1 else "egocentric"}, beta: {beta_list[0]}"

    plt.figure(figsize=(8,8))
    fig, ax = plt.subplots(4,4,num=figure_num)
    axes_flat = ax.flatten()
    current_correct_list = correct_total_list[p::n_plots]
    current_se_list = correct_se_list[p::n_plots]
    for i in range(len(current_correct_list)):
        axes_flat[i].plot(h0_for_plot,current_correct_list[i],color=colors[i], label=labels[i])
        axes_flat[i].plot(h0_for_plot, np.add(current_correct_list[i],current_se_list[i]), color = colors[i], linestyle = ':')
        axes_flat[i].plot(h0_for_plot, np.subtract(current_correct_list[i],current_se_list[i]), color = colors[i], linestyle = ':')
        axes_flat[i].set_title(f"{round(base_h0_list[i],5)}")

    fig.supxlabel(x_label)
    fig.supylabel(p_success_y)
    fig.suptitle(p_success_title)
    figure_num += 1
    plt.tight_layout()

time_df.to_csv(f"time_df_beta{beta_list[0]}.csv")
correct_df.to_csv(f"pcorrect_df_beta{beta_list[0]}.csv")
plt.show()

'''
range_y = "Time step with the smallest activation range"
range_title = f"Time step where the difference between least and most active neuron was smallest: {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction "
plt.figure(figsize=(10,5))
plt.figure(figure_num)
for i in range(len(range_total_list)):
    plt.plot(h0_for_plot,range_total_list[i],color=colors[i],label=labels[i])
    plt.plot(h0_for_plot, np.add(range_total_list[i],range_total_se_list[i]), color = colors[i], label = 'standard error', linestyle = ':')
    plt.plot(h0_for_plot, np.subtract(range_total_list[i],range_total_se_list[i]), color = colors[i], linestyle = ':')
plt.legend()
plt.xlabel(x_label)
plt.ylabel(range_y)
plt.title(range_title)
plt.show()
'''
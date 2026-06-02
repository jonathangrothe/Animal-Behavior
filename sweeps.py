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

allocentricFlag = 0 # 1 is allo, 0 is ego
h0s = [0.25,0.25] # attraction vector, first are attraction for targets, then agents
h_b = 0.2
sigma = 0.5
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
# not plotting trajectories or neurons in this file
plot_trajs = False
plot_neurons = False
uneven = True
sample_size = 2
start = 0.18
finish = 0.33
n_h0 = 2
h0_range= np.linspace(start,finish,num=n_h0)
#h0_for_plot = h0_range
h0_total_list = []
# 9.5: 0.215-0.32, 10: 0.208-0.325, 12: 0.189-0.339, 20: 0.177-0.338, 52: 0.182-0.325, 180: 0.189-0.326 ? 
beta_list = [40] # 0.22 to 0.31 - 0.1925-0.335
h0_list = []
for item in h0_range:
    h0_list.append([item,item])

# defining our metrics of interest: 
time_total_list = []
prob_total_list = []
prob_se_list = []
decision_points_total = []
decision_pos_total = []
decision_pos_x_total = []

# want to get the x position of the last decision point where y was close to 50

# we want to sweep over sigmas, so each time we set the h0 and the beta, and have sigma as change
for index in range(len(beta_list)):
        #print(f"h0: {h0_lis}")
        print(f"beta: {beta_list[index]}")
        change= {'h0':h0_list}
        base['beta'] = beta_list[index]
        target_list, time_list, decision_points, decision_pos, activity_df, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
        p_target, se_target = sim_met.get_success_rate(target_list, sample_size, ntargets)
        time_total_list.append(time_list)
        prob_total_list.append(p_target)
        prob_se_list.append(se_target)
        print(p_target)
        print(se_target)
        decision_points_total.append(decision_points)
        decision_pos_total.append(decision_pos)
        xpos_list = []
        for item in decision_pos:
            if len(item) == 0:
                xpos_list.append(80)
            if len(item) == 1:
                xpos_list.append(item[0][0])
            if len(item) > 1:
                index_first_dec = 0
                for index in range(len(item)):
                    if np.abs(item[index][1]-50) >= 3:
                        index_first_dec = index - 1
                xpos_list.append(item[index_first_dec][0])
        decision_pos_x_total.append(xpos_list)
    
correct_df = pd.DataFrame(prob_total_list)
#correct_df.index = h0_list
#correct_df.columns = beta_list

# compile the data we want from each run into a dataframe
# we want: p_correct and time, and we would prefer if they are labeled with simulation settings


figure_num = 1


# overall plotting settings
cmap = plt.get_cmap('viridis')
colors_time = cmap(np.linspace(0, 1, len(beta_list)))
labels_time = []
for item in beta_list:
    labels_time.append(f'beta: {item}') 

colors_prob = ['blue','red']
#labels_prob = ['lower target','middle target','upper target']
    
fig, axs = plt.subplots(1, 2, figsize=(14, 7),num=1)

x_label = "h0"
time_y_label = "Average time to target"
time_agg_title = f"Average time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, sigma: {sigma}"
figure_num, mean_time = sim_met.plot_metric(time_total_list,h0_range,colors_time,labels_time,axs[0],
                            time_agg_title,x_label,time_y_label)
    
time_df = pd.DataFrame(mean_time)
#time_df.index = beta_list
#time_df.columns = h0_range

    
dec_posx_y_label = "Average x position at decision"
decision_posx_agg_title = f"Average x position at bifurcation, {"allocentric" if allocentricFlag==1 else "egocentric"}, sigma: {sigma}"
figure_num, mean_dec_time = sim_met.plot_metric(decision_pos_x_total,h0_range,colors_time,labels_time,axs[1],
                                                    decision_posx_agg_title,x_label,dec_posx_y_label)

'''
p_success_y = "Probability of reaching each target (green middle)"
p_success_title = f"Probability of reaching each target, {"allocentric" if allocentricFlag==1 else "egocentric"}, beta: {beta}"

plt.figure(figsize=(8,8))
fig, ax = plt.subplots(2,2,num=figure_num)
axes_flat = ax.flatten()
current_prob_list = np.array(prob_total_list)
current_se_list = np.array(prob_se_list)
print(f"current: {current_prob_list}")
for i in range(len(current_prob_list)):
    for p in range(ntargets):
        axes_flat[i].plot(h0_range,current_prob_list[i][:,p],color=colors_prob[p])
        axes_flat[i].plot(h0_range, np.add(current_prob_list[i][:,p],current_se_list[i][:,p]), color = colors_prob[p], linestyle = ':')
        axes_flat[i].plot(h0_range, np.subtract(current_prob_list[i][:,p],current_se_list[i][:,p]), color = colors_prob[p], linestyle = ':')
        axes_flat[i].set_title(f"{round(beta_list[i],5)}")

fig.supxlabel(x_label)
fig.supylabel(p_success_y)
fig.suptitle(p_success_title)
figure_num += 1
plt.tight_layout()
'''
#time_df.to_csv(f"time_df_beta{beta_list[0]}_egocentric.csv")
#correct_df.to_csv(f"pcorrect_df_beta{beta_list[0]}_egocentric.csv")
plt.show()

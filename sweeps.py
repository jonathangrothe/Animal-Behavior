import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
import repeated_sims

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
plot_trajs = [False, 'scatter'] 
plot_neurons = False
n_lines = 3
uneven = True
sample_size = 150

base_value = 0.2
h0_range= np.linspace(1,1.00011,num=100)
h0_for_plot = h0_range
h0_total_list = []

for n in range(n_lines):
    h0_list = []
    for item in h0_range:
        h0_list.append([base_value+n*0.03,(base_value+n*0.03)*item])
    h0_total_list.append(h0_list)

# defining our metrics of interest: 
time_total_list = []
correct_total_list = []
correct_se_list = []
range_total_list = []
range_total_se_list = []
for item in h0_total_list:
    change= {'h0':item}
    success_list, target_list, time_list, decision_points, sum_activity_list, range_activity_list, range_argmin_list, activity_df, x_list, y_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
    p_success, se_success, p_correct, se_correct = sim_met.get_success_rate(target_list, sample_size, uneven, 1)
    range_mean, range_se = sim_met.get_metric_mean_se(range_argmin_list,sample_size)
    time_total_list.append(time_list)
    correct_total_list.append(p_correct)
    correct_se_list.append(se_correct)
    range_total_list.append(range_mean)
    range_total_se_list.append(range_se)

figure_num = 1

# overall plotting settings
colors = ['green','blue','red']
labels = ['base: 0.2','base: 0.25','base: 0.3']
x_label = "difference between targets"
time_y_label = "Average time to target"
time_agg_title = f"Average time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction, stopping distance = 0.5"
figure_num = sim_met.plot_metric(time_total_list,h0_for_plot,colors,labels,figure_num,(10,5),
                    time_agg_title,x_label,time_y_label)

    
p_success_y = "Probability of reaching top target"
p_success_title = f"Probability of getting within 0.5 units of top target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
plt.figure(figsize=(10,5))
plt.figure(figure_num)
if uneven: 
    for i in range(len(correct_total_list)):
        plt.plot(h0_for_plot,correct_total_list[i],color=colors[i], label=labels[i])
        plt.plot(h0_for_plot, np.add(correct_total_list[i],correct_se_list[i]), color = colors[i], label = 'standard error', linestyle = ':')
        plt.plot(h0_for_plot, np.subtract(correct_total_list[i],correct_se_list[i]), color = colors[i], linestyle = ':')
plt.legend()
plt.xlabel(x_label)
plt.ylabel(p_success_y)
plt.title(p_success_title)
figure_num += 1
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
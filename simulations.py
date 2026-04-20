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

allocentricFlag = 0 # 1 is allo, 0 is ego
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

plot_neurons = True
plot_trajs = [True, 'scatter']
plot_sweep = not (plot_neurons or plot_trajs[0])
sigma_for_sim = [0.4]
change = {'sigma':sigma_for_sim}
uneven = h0s[0] != h0s[1]

sample_size = 25

success_list, target_list, time_list, inhib_list, inhib_times_list, sum_activity_list, activity_df, x_list, y_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
p_success, se_success, p_correct, se_correct = sim_met.get_success_rate(target_list, sample_size, uneven, 1)
mean_inhib = sim_met.get_inhibition_rates(inhib_list, sample_size)

figure_num = 1

# SWEEP PLOTS

if plot_sweep:
    x_label = "h0"
    dist_y_label = "Average distance to target"
    dist_title = f" Distance to target over last quarter of simulation, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(success_list,sigma_for_sim,dist_title,x_label,dist_y_label,False)
    figure_num += 1

    dist_agg_title = f"Average distance to target over last quarter of simulation, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(success_list,sigma_for_sim,dist_agg_title,x_label,dist_y_label,True)
    figure_num += 1

    time_y_label = "Average time to target"
    time_title = f"Time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction, stopping distance = 0.5"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(time_list,sigma_for_sim,time_title,x_label,time_y_label,False)
    figure_num += 1

    time_agg_title = f"Average time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction, stopping distance = 0.5"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(time_list,sigma_for_sim,time_agg_title,x_label,time_y_label,True)
    figure_num += 1

    p_success_y = "Probability of reaching a target"
    p_success_title = f"Probability of getting within 0.5 units of a target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    if uneven: 
        plt.plot(sigma_for_sim,p_correct,color='Green', label="Correct target")
        plt.plot(sigma_for_sim, np.add(p_correct,se_correct), color = 'green', label = 'standard error for correct', linestyle = ':')
        plt.plot(sigma_for_sim, np.subtract(p_correct,se_correct), color = 'green', linestyle = ':')
        plt.plot(sigma_for_sim,p_success,color='Blue', label="Any target")
        plt.plot(sigma_for_sim, np.add(p_success,se_success), color = 'blue', label = 'standard error for any target', linestyle = ':')
        plt.plot(sigma_for_sim, np.subtract(p_success,se_success), color = 'blue', linestyle = ':')
        plt.legend()
    if not uneven:
        plt.plot(sigma_for_sim,p_success,color='Blue')
        plt.plot(sigma_for_sim, np.add(p_success,se_success), color = 'blue', label = 'standard error', linestyle = ':')
        plt.plot(sigma_for_sim, np.subtract(p_success,se_success), color = 'blue', linestyle = ':')
    plt.xlabel(x_label)
    plt.ylabel(p_success_y)
    plt.title(p_success_title)
    figure_num += 1

    
    n_inhib_y = "Average number of steps where every neuron was inhibited"
    n_inhib_title = f"Average number of steps where every neuron was inhibited, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    plt.plot(sigma_for_sim,mean_inhib,color='red')
    plt.xlabel(x_label)
    plt.ylabel(n_inhib_y)
    plt.title(n_inhib_title)
    figure_num += 1

# SINGLE SETTING PLOTS

# getting an approximation of the decision point ?

# this is only going off of the first sample, we should probably change this in repeated_sims
# so it just returns a one d list? and a df? 


if plot_neurons:   
    index_sum = 0
    for sim in range(len(sum_activity_list)):
        dec_index = sim_met.get_decision_time(sum_activity_list[sim])
        index_sum += dec_index
    mean_index = index_sum/len(sum_activity_list)
    print(f"mean decision index: {mean_index}")

    plt.figure(figure_num)
    mean_list = sim_met.plot_sum_activity(sum_activity_list)
    plt.title(f"Sum of all neuron activity over time, h0: {base['h0']}, sigma: {sigma_for_sim} {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    plt.xlabel("Time")
    plt.ylabel("Sum of neuron activity")
    figure_num+=1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,True,0,dec_index,100)
    plt.title(f"Average activity of all on average active neurons pre decision, h0: {base['h0']}, sigma: {sigma_for_sim}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,False,0,dec_index,100)
    plt.title(f"Activity of all on average active neurons pre decision, h0: {base['h0']}, sigma: {sigma_for_sim} {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,True,dec_index,0,100)
    plt.title(f"Average activity of all on average active neurons post decision, h0: {base['h0']}, sigma: {sigma_for_sim}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,False,dec_index,0,100)
    plt.title(f"Activity of all on average active neurons post decision, h0: {base['h0']}, sigma: {sigma_for_sim} {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1


if plot_trajs[0]:
    plt.figure(figsize=(7,7))
    plt.figure(figure_num)
    if plot_trajs[1] == 'heat':
        img = sim_met.plot_from_density(x_list, y_list, 30)
        plt.imshow(img)
    if plot_trajs[1] == 'scatter':
        sim_met.plot_traj(x_list,y_list,initialxt,initialyt,sample_size)
    plt.title(f"Trajectory plot, h0: {base['h0']}, sigma: {sigma_for_sim}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    

plt.show()
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
h0_for_plot = np.linspace(0.249,0.252,num=15)
h0_for_uneven_sim = []
h0_for_even_sim = []
for item in h0_for_plot:
    h0_for_uneven_sim.append([item,item+0.01])
    h0_for_even_sim.append([item,item])
change_uneven = {'h0':h0_for_uneven_sim}
change_even = {'h0':h0_for_even_sim}
change = {'h0':h0_for_uneven_sim}
uneven = True

sample_size = 20

if plot_sweep:
    success_list_u, target_list_u, time_list_u, inhib_list_u, inhib_times_list_u, decision_points_u, sum_activity_list_u, activity_df_u, x_list_u, y_list_u  = repeated_sims.sample_sims(base,change_uneven,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
    p_success_uneven, se_success_uneven, p_correct_uneven, se_correct_uneven = sim_met.get_success_rate(target_list_u, sample_size, uneven, 1)
    mean_inhib_u = sim_met.get_inhibition_rates(inhib_list_u, sample_size)

    success_list_e, target_list_e, time_list_e, inhib_list_e, inhib_times_list_e, decision_points_e, sum_activity_list_e, activity_df_e, x_list_e, y_list_e  = repeated_sims.sample_sims(base,change_even,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
    p_success_even, se_success_even, p_correct_even, se_correct_even = sim_met.get_success_rate(target_list_e, sample_size, uneven, 1)
    mean_inhib_e = sim_met.get_inhibition_rates(inhib_list_e, sample_size)
else:
    success_list, target_list, time_list, inhib_list, inhib_times_list, decision_points, sum_activity_list, activity_df, x_list, y_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
    p_success, se_success, p_correct, se_correct = sim_met.get_success_rate(target_list, sample_size, uneven, 1)
    mean_inhib = sim_met.get_inhibition_rates(inhib_list, sample_size)

figure_num = 1

# SWEEP PLOTS

if plot_sweep:
    x_label = "h0"
    '''
    dist_y_label = "Average distance to target"
    dist_title = f" Distance to target over last quarter of simulation, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(success_list,h0_for_plot,dist_title,x_label,dist_y_label,False)
    figure_num += 1

    dist_agg_title = f"Average distance to target over last quarter of simulation, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(success_list,h0_for_plot,dist_agg_title,x_label,dist_y_label,True)
    figure_num += 1
    '''
    time_y_label = "Average time to target"
    '''
    time_title = f"Time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, stopping distance = 0.5"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(time_list,h0_for_plot,time_title,x_label,time_y_label,False)
    figure_num += 1
    '''

    time_agg_title = f"Average time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction, stopping distance = 0.5"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(time_list_u,h0_for_plot,'green','Top target is better',True)
    sim_met.plot_metric(time_list_e,h0_for_plot,'blue', 'Targets are equal', True)
    plt.title(time_agg_title)
    plt.xlabel(x_label)
    plt.ylabel(time_y_label)
    figure_num += 1

    p_success_y = "Probability of reaching top target"
    p_success_title = f"Probability of getting within 0.5 units of top target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    if uneven: 
        plt.plot(h0_for_plot,p_correct_uneven,color='Green', label="Top target is more attractive")
        plt.plot(h0_for_plot, np.add(p_correct_uneven,se_correct_uneven), color = 'green', label = 'standard error', linestyle = ':')
        plt.plot(h0_for_plot, np.subtract(p_correct_uneven,se_correct_uneven), color = 'green', linestyle = ':')
        plt.plot(h0_for_plot,p_correct_even,color='Blue', label="Targets are even")
        plt.plot(h0_for_plot, np.add(p_correct_even,se_correct_even), color = 'blue', label = 'standard error', linestyle = ':')
        plt.plot(h0_for_plot, np.subtract(p_correct_even,se_correct_even), color = 'blue', linestyle = ':')
        plt.legend()
    '''
    if not uneven:
        plt.plot(h0_for_plot,p_success,color='Blue')
        plt.plot(h0_for_plot, np.add(p_success,se_success), color = 'blue', label = 'standard error', linestyle = ':')
        plt.plot(h0_for_plot, np.subtract(p_success,se_success), color = 'blue', linestyle = ':')
    '''
    plt.xlabel(x_label)
    plt.ylabel(p_success_y)
    plt.title(p_success_title)
    figure_num += 1

    '''
    n_inhib_y = "Average number of steps where every neuron was inhibited"
    n_inhib_title = f"Average number of steps where every neuron was inhibited, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    plt.plot(h0_for_plot,mean_inhib,color='red')
    plt.xlabel(x_label)
    plt.ylabel(n_inhib_y)
    plt.title(n_inhib_title)
    figure_num += 1
    '''
    
# SINGLE SETTING PLOTS

# need to shift this so it plots a heatmap and maybe less of this arbitrary stuff
if plot_neurons:  
    sum_start_ind = 0
    sum_end_ind = 0
    for item in decision_points:
        sum_start_ind += item[0]
        sum_end_ind += item[1]
    mean_start_index = round(sum_start_ind/len(decision_points))
    mean_end_index = round(sum_end_ind/len(decision_points))
    diff = mean_end_index-mean_start_index
    print(mean_start_index)
    print(mean_end_index) 
    print(diff)
    plt.figure(figsize=(12,7))
    plt.figure(figure_num)
    rolled= np.roll(activity_df.values, shift=50, axis=0)
    plt.imshow(rolled,cmap='viridis',aspect='auto')
    plt.axvline(x=mean_start_index, color='red', linestyle='--')
    plt.axvline(x=mean_end_index, color='red', linestyle='--')
    plt.title("Neuron activation over time example")
    plt.colorbar()
    figure_num += 1
    plt.figure(figure_num)
    mean_list = sim_met.plot_sum_activity(sum_activity_list)
    plt.title(f"Sum of all neuron activity over time, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    plt.xlabel("Time")
    plt.ylabel("Sum of neuron activity")
    figure_num+=1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,True,0,mean_start_index,100)
    plt.title(f"Average activity of all on average active neurons pre decision, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,False,0,mean_start_index,100)
    plt.title(f"Activity of all on average active neurons pre decision, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,True,mean_start_index,mean_end_index,100)
    plt.title(f"Average activity of all on average active neurons during decision, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,False,mean_start_index,mean_end_index,100)
    plt.title(f"Activity of all on average active neurons during decision, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,True,mean_end_index,0,100)
    plt.title(f"Average activity of all on average active neurons post decision, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,False,mean_end_index,0,100)
    plt.title(f"Activity of all on average active neurons post decision, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1


if plot_trajs[0]:
    plt.figure(figsize=(7,7))
    plt.figure(figure_num)
    if plot_trajs[1] == 'heat':
        img = sim_met.plot_from_density(x_list, y_list, 30)
        plt.imshow(img)
    if plot_trajs[1] == 'scatter':
        sim_met.plot_traj(x_list,y_list,initialxt,initialyt,sample_size)
    plt.title(f"Trajectory plot, h0: {h0_for_plot}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figsize=(7,7))
    plt.figure(figure_num)
    if plot_trajs[1] == 'scatter':
        sim_met.plot_traj(x_list,y_list,initialxt,initialyt,sample_size,mean_start_index,mean_end_index)
    plt.title(f"Trajectory plot, from time: {mean_start_index} to time: {mean_end_index} h0: {h0_for_uneven_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

plt.show()